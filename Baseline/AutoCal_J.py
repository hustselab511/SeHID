
import numpy as np
from scipy.signal import remez, filtfilt, find_peaks

# --- TIM-based J-peak detection parameters ---
# The original paper uses 250 Hz. This implementation keeps the public
# interface fs=125 and rescales time-related parameters internally.
P_SAMPLES_AT_250HZ = 30
K_FACTOR = 2.0
T_EPSILON_MS = 40
DELTA_T_DEFAULT_MS = 80
LPF_CUTOFF_HZ = 1.8
LPF_TAPS = 200
BPF_ORDER = 1024
CALIBRATION_DURATION_SEC = 60


def _to_numpy_1d(x):
    if hasattr(x, "detach"):
        x = x.detach()
    if hasattr(x, "cpu"):
        x = x.cpu().numpy()
    x = np.asarray(x).squeeze()
    if x.ndim != 1:
        raise ValueError("bcg_signal must be 1-D after squeeze().")
    return x.astype(float, copy=False)


def _safe_zscore(x):
    std = np.std(x)
    if std < 1e-12:
        return np.zeros_like(x)
    return (x - np.mean(x)) / std


def _design_bandpass_fir(fs, numtaps=BPF_ORDER):
    nyq = fs / 2.0
    # Remez is used here as a practical approximation of the
    # Parks-McClellan FIR design described in the paper.
    bands = [0.0, 1.5, 2.0, 14.0, 14.5, nyq]
    desired = [0.0, 1.0, 0.0]
    weights = [1.0, 1.0, 1.0]
    return remez(numtaps, bands, desired, weight=weights, fs=fs)


def _design_lowpass_fir(fs, cutoff_hz=LPF_CUTOFF_HZ, numtaps=LPF_TAPS):
    nyq = fs / 2.0
    stop = min(cutoff_hz + 0.4, nyq - 1e-3)
    if stop <= cutoff_hz:
        stop = min(cutoff_hz * 1.2, nyq - 1e-3)
    bands = [0.0, cutoff_hz, stop, nyq]
    desired = [1.0, 0.0]
    weights = [1.0, 1.0]
    return remez(numtaps, bands, desired, weight=weights, fs=fs)


def tim_preprocess(bcg_signal, fs):
    """
    Placeholder preprocessing function.

    Signal preprocessing is assumed to be completed upstream.
    This function is kept only to preserve the original code structure.
    """
    return _to_numpy_1d(bcg_signal)


def calculate_x_det(bcg_preprocessed, fs, lpf_cutoff=LPF_CUTOFF_HZ, lpf_taps=LPF_TAPS):
    """
    Compute the detection signal x_DET:

        x_DET[n] = sum_k b[k] * x_BCG^2[n-k]
    """
    bcg_preprocessed = _to_numpy_1d(bcg_preprocessed)
    bcg_squared = np.square(bcg_preprocessed)

    if len(bcg_squared) < lpf_taps * 3:
        return bcg_squared.copy()

    fir_lpf = _design_lowpass_fir(fs=fs, cutoff_hz=lpf_cutoff, numtaps=lpf_taps)
    return filtfilt(fir_lpf, [1.0], bcg_squared)


def _extract_positive_intervals(x_sqr):
    intervals = []
    start = None
    for i, v in enumerate(x_sqr):
        if v > 0 and start is None:
            start = i
        elif v <= 0 and start is not None:
            intervals.append([start, i - 1])
            start = None
    if start is not None:
        intervals.append([start, len(x_sqr) - 1])
    return intervals


def calculate_x_sqr(x_det, fs, p_duration_ms=None, k=K_FACTOR):
    """
    Generate binary signal x_SQR using the adaptive thresholding rule:

        x_DET[i] >= mu(i-p:i) + k * sigma(i-p:i)

    If p_duration_ms is None, the paper setting p=30 at 250 Hz is
    rescaled proportionally to the current sampling rate.
    """
    x_det = _to_numpy_1d(x_det)
    n = len(x_det)
    x_sqr = -np.ones(n, dtype=np.int8)

    if p_duration_ms is None:
        p_samples = max(1, int(round(P_SAMPLES_AT_250HZ * fs / 250.0)))
    else:
        p_samples = max(1, int(round(p_duration_ms * fs / 1000.0)))

    for i in range(p_samples, n):
        window = x_det[i - p_samples:i]
        mu_local = np.mean(window)
        sigma_local = np.std(window)
        if x_det[i] >= mu_local + k * sigma_local:
            x_sqr[i] = 1

    return x_sqr, _extract_positive_intervals(x_sqr)


def _interval_duration_stats(intervals):
    if not intervals:
        return None, None
    durations = np.array([end - start + 1 for start, end in intervals], dtype=float)
    median = np.median(durations)
    mad = np.median(np.abs(durations - median))
    robust_std = 1.4826 * mad if mad > 0 else max(1.0, np.std(durations))
    return median, robust_std


def _negative_intervals_from_positive(positive_intervals, n_samples):
    negatives = []
    prev_end = -1
    for start, end in positive_intervals:
        if start - prev_end > 1:
            negatives.append([prev_end + 1, start - 1])
        prev_end = end
    if prev_end < n_samples - 1:
        negatives.append([prev_end + 1, n_samples - 1])
    return negatives


def refine_candidate_intervals(x_det, x_sqr, candidate_intervals, fs, base_k=K_FACTOR):
    """
    Approximate the second-pass refinement described in the paper:
    1) split overly long positive intervals using local valleys;
    2) inspect overly long negative intervals with a reduced threshold.
    """
    x_det = _to_numpy_1d(x_det)
    x_sqr = np.array(x_sqr, copy=True)
    intervals = [list(itv) for itv in candidate_intervals]
    if not intervals:
        return x_sqr, intervals

    pos_med, pos_std = _interval_duration_stats(intervals)
    if pos_med is None:
        return x_sqr, intervals

    refined = []
    for start, end in intervals:
        dur = end - start + 1
        if dur > pos_med + 2.0 * pos_std and dur > 3:
            seg = x_det[start:end + 1]
            valley_rel = int(np.argmin(seg))
            valley = start + valley_rel
            left_len = valley - start
            right_len = end - valley
            if left_len >= max(2, int(0.25 * pos_med)) and right_len >= max(2, int(0.25 * pos_med)):
                x_sqr[valley] = -1
                refined.append([start, valley - 1])
                refined.append([valley + 1, end])
            else:
                refined.append([start, end])
        else:
            refined.append([start, end])

    intervals = [itv for itv in refined if itv[0] <= itv[1]]
    neg_intervals = _negative_intervals_from_positive(intervals, len(x_det))
    neg_med, neg_std = _interval_duration_stats(neg_intervals)

    if neg_med is not None:
        p_samples = max(1, int(round(P_SAMPLES_AT_250HZ * fs / 250.0)))
        reduced_k = max(0.5, base_k * 0.5)
        for start, end in neg_intervals:
            dur = end - start + 1
            if dur <= neg_med + 2.0 * neg_std:
                continue
            for i in range(max(start, p_samples), end + 1):
                window = x_det[i - p_samples:i]
                mu_local = np.mean(window)
                sigma_local = np.std(window)
                if x_det[i] >= mu_local + reduced_k * sigma_local:
                    x_sqr[i] = 1

    return x_sqr, _extract_positive_intervals(x_sqr)


def _local_max_index(signal, start, end):
    seg = signal[start:end + 1]
    if seg.size == 0:
        return None
    peaks, _ = find_peaks(seg)
    if len(peaks) == 0:
        return start + int(np.argmax(seg))
    return start + int(peaks[np.argmax(seg[peaks])])


def calibrate_delta_t(
    bcg_preprocessed,
    x_det,
    candidate_intervals,
    fs,
    t_epsilon=T_EPSILON_MS,
    default_delta_t_ms=DELTA_T_DEFAULT_MS,
):
    """
    Estimate the subject-specific temporal offset t_J,DET as the median
    distance between local maxima in x_DET and corresponding BCG peaks.
    """
    bcg_preprocessed = _to_numpy_1d(bcg_preprocessed)
    x_det = _to_numpy_1d(x_det)
    t_epsilon_samples = max(1, int(round(t_epsilon * fs / 1000.0)))
    default_delta_t_samples = int(round(default_delta_t_ms * fs / 1000.0))

    delta_t_list = []
    j_pos_list = []
    for start, end in candidate_intervals:
        det_max_pos = _local_max_index(x_det, start, end)
        if det_max_pos is None:
            continue

        search_start = max(start, det_max_pos - t_epsilon_samples)
        search_end = min(end, det_max_pos + t_epsilon_samples)
        j_pos = _local_max_index(bcg_preprocessed, search_start, search_end)
        if j_pos is None:
            continue

        delta_t_list.append(det_max_pos - j_pos)
        j_pos_list.append(j_pos)

    if len(delta_t_list) == 0:
        delta_t_j_det = default_delta_t_samples
    else:
        delta_t_j_det = int(round(np.median(delta_t_list)))

    return delta_t_j_det, np.array(sorted(set(j_pos_list)), dtype=int)


def locate_j_peaks(
    bcg_preprocessed,
    x_det,
    candidate_intervals,
    delta_t_j_det,
    fs,
    t_epsilon=T_EPSILON_MS,
    default_delta_t_ms=DELTA_T_DEFAULT_MS,
):
    """
    Locate J-peaks by first finding x_DET local maxima inside each
    candidate interval, then searching around the calibrated offset.
    """
    bcg_preprocessed = _to_numpy_1d(bcg_preprocessed)
    x_det = _to_numpy_1d(x_det)
    t_epsilon_samples = max(1, int(round(t_epsilon * fs / 1000.0)))

    raw_j_peaks = []
    for start, end in candidate_intervals:
        det_max_pos = _local_max_index(x_det, start, end)
        if det_max_pos is None:
            continue

        j_expected_pos = det_max_pos - delta_t_j_det
        search_start = max(start, j_expected_pos - t_epsilon_samples)
        search_end = min(end, j_expected_pos + t_epsilon_samples)
        j_pos = _local_max_index(bcg_preprocessed, search_start, search_end)
        if j_pos is not None:
            raw_j_peaks.append(j_pos)

    if not raw_j_peaks:
        return np.array([], dtype=int)
    return np.array(sorted(set(raw_j_peaks)), dtype=int)


def post_process_j_peaks(j_peaks, bcg_preprocessed, x_det, fs):
    """
    Post-process detected J-peaks by:
    - removing overly close false positives;
    - inserting one weak candidate inside overly long intervals.
    """
    j_peaks = np.asarray(j_peaks, dtype=int)
    if len(j_peaks) < 2:
        return j_peaks

    bcg_preprocessed = _to_numpy_1d(bcg_preprocessed)
    x_det = _to_numpy_1d(x_det)

    rr_like = np.diff(j_peaks)
    median_jj = np.median(rr_like)
    if median_jj <= 0:
        return j_peaks

    min_sep = max(1, int(round(0.3 * median_jj)))

    kept = [int(j_peaks[0])]
    for j in j_peaks[1:]:
        if j - kept[-1] >= min_sep:
            kept.append(int(j))
        else:
            if bcg_preprocessed[j] > bcg_preprocessed[kept[-1]]:
                kept[-1] = int(j)

    refined = [kept[0]]
    long_thr = 1.5 * median_jj
    for j in kept[1:]:
        prev = refined[-1]
        gap = j - prev
        if gap > long_thr:
            mid_start = prev + int(round(0.25 * gap))
            mid_end = j - int(round(0.25 * gap))
            if mid_end > mid_start + 2:
                det_mid = _local_max_index(x_det, mid_start, mid_end)
                if det_mid is not None:
                    local_half = max(1, int(round(T_EPSILON_MS * fs / 1000.0)))
                    cand_start = max(mid_start, det_mid - local_half)
                    cand_end = min(mid_end, det_mid + local_half)
                    mid_j = _local_max_index(bcg_preprocessed, cand_start, cand_end)
                    if mid_j is not None and (mid_j - prev) >= min_sep and (j - mid_j) >= min_sep:
                        refined.append(int(mid_j))
        refined.append(int(j))

    return np.array(sorted(set(refined)), dtype=int)


def tim_bcg_jpeak_detect(bcg_signal, fs=125):
    """
    Main function for TIM-based J-peak detection from BCG.

    This implementation preserves the original public interface:
        input  -> 1-D BCG signal
        output -> J-peak indices at the original sampling rate

    Signal preprocessing is assumed to be completed upstream.
    This function focuses on the core detection pipeline described in the paper:
    1) compute x_DET = LPF(x_BCG^2);
    2) extract candidate positive intervals using x_SQR;
    3) perform second-pass refinement;
    4) estimate subject-specific temporal offset t_J,DET;
    5) locate J-peaks inside candidate intervals;
    6) apply post-processing.
    """
    bcg_signal = _to_numpy_1d(bcg_signal)
    if len(bcg_signal) == 0:
        return np.array([], dtype=int)

    bcg_preprocessed = tim_preprocess(bcg_signal, fs=fs)
    x_det = calculate_x_det(bcg_preprocessed, fs=fs)

    x_sqr, candidate_intervals = calculate_x_sqr(x_det, fs=fs, p_duration_ms=None, k=K_FACTOR)
    if len(candidate_intervals) == 0:
        return np.array([], dtype=int)

    x_sqr, candidate_intervals = refine_candidate_intervals(
        x_det, x_sqr, candidate_intervals, fs=fs, base_k=K_FACTOR
    )
    if len(candidate_intervals) == 0:
        return np.array([], dtype=int)

    calibration_len = min(len(bcg_preprocessed), int(round(CALIBRATION_DURATION_SEC * fs)))
    calib_intervals = [[s, e] for s, e in candidate_intervals if s < calibration_len and e >= 0]
    calib_intervals = [[s, min(e, calibration_len - 1)] for s, e in calib_intervals if s <= calibration_len - 1]
    if len(calib_intervals) == 0:
        calib_intervals = candidate_intervals

    delta_t_j_det, _ = calibrate_delta_t(
        bcg_preprocessed[:calibration_len],
        x_det[:calibration_len],
        calib_intervals,
        fs=fs,
        t_epsilon=T_EPSILON_MS,
        default_delta_t_ms=DELTA_T_DEFAULT_MS,
    )

    raw_j_peaks = locate_j_peaks(
        bcg_preprocessed,
        x_det,
        candidate_intervals,
        delta_t_j_det,
        fs=fs,
        t_epsilon=T_EPSILON_MS,
        default_delta_t_ms=DELTA_T_DEFAULT_MS,
    )
    if len(raw_j_peaks) == 0:
        return np.array([], dtype=int)

    final_j_peaks = post_process_j_peaks(raw_j_peaks, bcg_preprocessed, x_det, fs=fs)
    final_j_peaks = final_j_peaks[(final_j_peaks >= 0) & (final_j_peaks < len(bcg_signal))]
    return np.array(sorted(set(final_j_peaks.tolist())), dtype=int)


if __name__ == "__main__":
    x = np.random.rand(1, 1, 625)
    signal = x[0, 0]
    peaks = tim_bcg_jpeak_detect(signal, fs=125)

    print("Input shape:", x.shape)
    print("Detected J-peaks:", peaks)
    print("Number of peaks:", len(peaks))
