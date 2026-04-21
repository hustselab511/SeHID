
import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks, hilbert, resample_poly


# ---------------------------------------------------------------------
# Bed-only implementation of the BSPC 2024 J-wave detector.
#
# Notes:
# 1) The paper uses a spline-CWT whose mother wavelet is the first
#    derivative of a 4th-order B-spline. pywt does not expose that exact
#    spline wavelet directly, so this implementation uses 'gaus1', which
#    the paper itself states is similar to the intended wavelet.
# 2) The user requested that signal preprocessing is handled upstream.
#    Therefore this file reproduces the core detection pipeline only.
# 3) Input/output interface is preserved: the public function accepts a
#    1D BCG signal and returns J-peak indices at the original sampling
#    rate.
# ---------------------------------------------------------------------


TARGET_FS = 1000
CWT_SCALE = 5
CWT_WAVELET = "gaus1"

MIN_ENV_DISTANCE_MS = 480
FIRST_STAGE_PAIR_LIMIT_MS = 400

BACK_SEARCH_MIN_MS = 200
FWD_SEARCH_MIN_MS = 200
SECOND_SEARCH_BACK_MS = 250
SECOND_SEARCH_OFFSET_MS = 80

MIN_VALID_JJ_MS = 556
MAX_VALID_JJ_MS = 1352


@dataclass
class Candidate:
    j_pos: int
    p_wn: int
    p_wp: int
    f_wn: float
    f_wp: float
    overwritten_from_best_candidate: bool = False
    p_wn_alt: Optional[int] = None
    p_wp_alt: Optional[int] = None
    wn_alt: Optional[float] = None
    wp_alt: Optional[float] = None


def _to_numpy_1d(x) -> np.ndarray:
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    elif hasattr(x, "cpu"):
        x = x.cpu().numpy()
    x = np.asarray(x).squeeze()
    if x.ndim != 1:
        raise ValueError("bcg_signal must be convertible to a 1D array.")
    return x.astype(float, copy=False)


def _upsample_to_1khz(x: np.ndarray, fs: int) -> np.ndarray:
    if fs == TARGET_FS:
        return x
    return resample_poly(x, TARGET_FS, fs)


def _downsample_indices(indices: Sequence[int], fs: int, n_orig: int) -> np.ndarray:
    idx = np.asarray(indices, dtype=float)
    if idx.size == 0:
        return np.array([], dtype=int)
    idx = np.round(idx * fs / TARGET_FS).astype(int)
    idx = idx[(idx >= 0) & (idx < n_orig)]
    return np.unique(np.sort(idx))


def _compute_cwt_and_envelope(x_1k: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Approximation of the paper's spline-CWT using the first derivative of a
    # Gaussian-like kernel. This follows the paper's remark that the selected
    # spline wavelet is similar to the first derivative of a Gaussian.
    sigma = float(CWT_SCALE)
    cwt = gaussian_filter1d(x_1k, sigma=sigma, order=1, mode="nearest")
    env = np.abs(hilbert(cwt))
    min_distance = int(MIN_ENV_DISTANCE_MS / 1000 * TARGET_FS)
    m_env, _ = find_peaks(-env, distance=max(min_distance, 1))
    return cwt, env, m_env


def _local_extremum_before(x: np.ndarray, center: int, mode: str, max_back: int) -> Optional[int]:
    start = max(0, center - max_back)
    seg = x[start:center]
    if seg.size < 3:
        return None
    if mode == "max":
        peaks, _ = find_peaks(seg)
    else:
        peaks, _ = find_peaks(-seg)
    if len(peaks) == 0:
        return None
    return start + peaks[np.argmax(seg[peaks]) if mode == "max" else np.argmax((-seg)[peaks])]


def _local_extremum_after(x: np.ndarray, center: int, mode: str, max_forward: int) -> Optional[int]:
    end = min(len(x), center + max_forward)
    seg = x[center:end]
    if seg.size < 3:
        return None
    if mode == "max":
        peaks, _ = find_peaks(seg)
    else:
        peaks, _ = find_peaks(-seg)
    if len(peaks) == 0:
        return None
    return center + peaks[np.argmax(seg[peaks]) if mode == "max" else np.argmax((-seg)[peaks])]


def _argmax_between(x: np.ndarray, left: int, right: int) -> Optional[int]:
    left, right = int(left), int(right)
    if left > right:
        left, right = right, left
    left = max(0, left)
    right = min(len(x) - 1, right)
    if right - left < 1:
        return None
    return left + int(np.argmax(x[left:right + 1]))


def _build_general_ensemble(
    bcg: np.ndarray, cwt: np.ndarray, m_env: np.ndarray, half_window_ms: int = 300
):
    half_window = int(half_window_ms / 1000 * TARGET_FS)
    valid = [p for p in m_env if p - half_window >= 0 and p + half_window < len(cwt)]
    if not valid:
        return None

    cwt_stack = np.stack([cwt[p - half_window:p + half_window] for p in valid], axis=0)
    bcg_stack = np.stack([bcg[p - half_window:p + half_window] for p in valid], axis=0)

    avg_cwt = cwt_stack.mean(axis=0)
    avg_bcg = bcg_stack.mean(axis=0)
    center = half_window

    pg_wn1 = center
    pg_wp1 = _local_extremum_after(avg_cwt, pg_wn1, "max", int(0.25 * TARGET_FS))
    pg_wp2 = _local_extremum_before(avg_cwt, pg_wn1, "max", int(0.25 * TARGET_FS))
    pg_wn2 = _local_extremum_before(avg_cwt, pg_wp2 if pg_wp2 is not None else pg_wn1, "min", int(0.25 * TARGET_FS))

    if pg_wp1 is None or pg_wp2 is None or pg_wn2 is None:
        return None

    # Estimate whether the ensemble-average J-wave lies forward to Wn1.
    j_forward = bool(_argmax_between(avg_bcg, pg_wn1, pg_wp1) is not None)
    if j_forward:
        jp_est = _argmax_between(avg_bcg, pg_wn1, pg_wp1)
        j_forward = jp_est is not None and jp_est >= pg_wn1
    else:
        jp_est = _argmax_between(avg_bcg, pg_wn2, pg_wp2)

    sd_wn2 = float(np.std(cwt_stack[:, pg_wn2]))
    sd_wp1 = float(np.std(cwt_stack[:, pg_wp1]))

    # Fig. 7 rules.
    if j_forward:
        if avg_cwt[pg_wp1] > avg_cwt[pg_wp2]:
            thd_n = 0.6 + 0.5 * sd_wn2
            thd_p = 0.6 - 0.5 * sd_wp1
        else:
            thd_n = 0.5 + 0.5 * sd_wn2
            thd_p = 0.5 - 0.5 * sd_wp1
    else:
        thd_n = 0.5 + 0.5 * sd_wn2
        thd_p = 0.7 - 0.5 * sd_wp1

    return {
        "avg_cwt": avg_cwt,
        "avg_bcg": avg_bcg,
        "pg_wn1": pg_wn1,
        "pg_wn2": pg_wn2,
        "pg_wp1": pg_wp1,
        "pg_wp2": pg_wp2,
        "j_forward": j_forward,
        "thd_n": float(thd_n),
        "thd_p": float(thd_p),
    }


def _first_stage_candidate(
    bcg: np.ndarray,
    cwt: np.ndarray,
    m_env_idx: int,
    thd_n: float,
    thd_p: float,
) -> Optional[Candidate]:
    wn1_pos = int(m_env_idx)
    wn1_val = float(cwt[wn1_pos])

    wp1_pos = _local_extremum_after(cwt, wn1_pos, "max", int(0.25 * TARGET_FS))
    wp2_pos = _local_extremum_before(cwt, wn1_pos, "max", int(0.25 * TARGET_FS))
    if wp1_pos is None or wp2_pos is None:
        return None

    wn2_pos = _local_extremum_before(cwt, wp2_pos, "min", int(0.25 * TARGET_FS))
    if wn2_pos is None:
        return None

    wp1_val = float(cwt[wp1_pos])
    wp2_val = float(cwt[wp2_pos])
    wn2_val = float(cwt[wn2_pos])

    pair_limit = int(FIRST_STAGE_PAIR_LIMIT_MS / 1000 * TARGET_FS)

    if wp1_val > wp2_val:
        dominant = (
            wn2_val < 0.85 * wn1_val
            and wp2_val < 0.85 * wp1_val
            and (wp1_pos - wn1_pos) < pair_limit
        )
        if dominant:
            p_wn, p_wp = wn2_pos, wp1_pos
        else:
            if wn2_val < wn1_val * thd_n and wp2_val > 0.89 * wp1_val:
                p_wn, p_wp = wn2_pos, wp2_pos
            else:
                p_wn, p_wp = wn1_pos, wp1_pos
    else:
        dominant = (
            wn2_val < 0.8 * wn1_val
            and wp2_val > 0.8 * wp1_val
            and (wp1_pos - wn1_pos) < pair_limit
        )
        if dominant:
            p_wn, p_wp = wn2_pos, wp1_pos
        else:
            if wn2_val > wn1_val * thd_n and wp1_val > 0.89 * wp2_val:
                p_wn, p_wp = wn1_pos, wp1_pos
            else:
                p_wn, p_wp = wn2_pos, wp2_pos

    j_pos = _argmax_between(bcg, p_wn, p_wp)
    if j_pos is None:
        return None

    return Candidate(
        j_pos=j_pos,
        p_wn=p_wn,
        p_wp=p_wp,
        f_wn=float(cwt[p_wn]),
        f_wp=float(cwt[p_wp]),
    )


def _mean_prev_jj(accepted_peaks: Sequence[int]) -> Optional[float]:
    if len(accepted_peaks) < 2:
        return None
    jj = np.diff(np.asarray(accepted_peaks))
    if len(jj) >= 5:
        return float(np.mean(jj[-5:]))
    return float(np.mean(jj))


def _search_backward_pair(cwt: np.ndarray, ref_pos: int) -> Tuple[Optional[int], Optional[int]]:
    start = max(0, ref_pos - int(SECOND_SEARCH_BACK_MS / 1000 * TARGET_FS))
    wn3 = _local_extremum_before(cwt, ref_pos - int(SECOND_SEARCH_OFFSET_MS / 1000 * TARGET_FS), "min", ref_pos - start)
    if wn3 is None or wn3 < start:
        wn3 = _local_extremum_after(cwt, start, "min", ref_pos - start)
    if wn3 is None:
        return None, None
    wp3 = _local_extremum_after(cwt, wn3, "max", int(0.20 * TARGET_FS))
    return wn3, wp3


def _search_forward_pair(cwt: np.ndarray, ref_pos: int) -> Tuple[Optional[int], Optional[int]]:
    offset = int(SECOND_SEARCH_OFFSET_MS / 1000 * TARGET_FS)
    search_start = min(len(cwt) - 1, ref_pos + offset)
    wn4 = _local_extremum_after(cwt, search_start + int(FWD_SEARCH_MIN_MS / 1000 * TARGET_FS), "min", int(0.25 * TARGET_FS))
    if wn4 is None:
        return None, None
    wp4 = _local_extremum_after(cwt, wn4, "max", int(0.20 * TARGET_FS))
    return wn4, wp4


def _better_candidate(
    bcg: np.ndarray,
    cwt: np.ndarray,
    cand: Candidate,
    accepted_peaks: Sequence[int],
) -> Candidate:
    if len(accepted_peaks) < 2:
        return cand

    prev_jj_mean = np.mean(np.diff(np.asarray(accepted_peaks[-4:]))) if len(accepted_peaks) >= 4 else np.mean(np.diff(np.asarray(accepted_peaks)))
    current_ijjp = cand.j_pos - accepted_peaks[-1]
    prev_jj = accepted_peaks[-1] - accepted_peaks[-2]

    # Backward candidate, Eq. (3)
    if current_ijjp > prev_jj_mean:
        wn3, wp3 = _search_backward_pair(cwt, cand.j_pos)
        if wn3 is not None and wp3 is not None:
            jpb = _argmax_between(bcg, wn3, wp3)
            if jpb is not None:
                cond = (
                    cwt[wn3] < cand.f_wn * 0.5
                    and cwt[wp3] > 0.54 * cand.f_wp
                    and current_ijjp > 0.86 * prev_jj_mean
                    and bcg[jpb] > 0.86 * bcg[cand.j_pos]
                )
                if cond:
                    cand = Candidate(
                        j_pos=jpb,
                        p_wn=wn3,
                        p_wp=wp3,
                        f_wn=float(cwt[wn3]),
                        f_wp=float(cwt[wp3]),
                        overwritten_from_best_candidate=True,
                        p_wn_alt=wn3,
                        p_wp_alt=wp3,
                        wn_alt=float(cwt[wn3]),
                        wp_alt=float(cwt[wp3]),
                    )

    # Forward candidate, Eq. (4)
    if current_ijjp < prev_jj:
        wn3, wp3 = _search_forward_pair(cwt, cand.j_pos)
        if wn3 is not None and wp3 is not None:
            jpb = _argmax_between(bcg, wn3, wp3)
            if jpb is not None:
                cond = (
                    cwt[wn3] < 0.4 * cand.f_wn
                    and cwt[wp3] > 0.45 * cand.f_wp
                    and current_ijjp < 1.35 * prev_jj_mean
                    and bcg[cand.j_pos] > 1.5 * bcg[jpb]
                )
                if cond:
                    cand = Candidate(
                        j_pos=jpb,
                        p_wn=wn3,
                        p_wp=wp3,
                        f_wn=float(cwt[wn3]),
                        f_wp=float(cwt[wp3]),
                        overwritten_from_best_candidate=True,
                        p_wn_alt=wn3,
                        p_wp_alt=wp3,
                        wn_alt=float(cwt[wn3]),
                        wp_alt=float(cwt[wp3]),
                    )
    return cand


def _recognition_pass(
    cand: Candidate,
    accepted_peaks: Sequence[int],
    u1: Optional[float],
    u2: Optional[float],
) -> Tuple[bool, Optional[float], Optional[float], Optional[float]]:
    if u1 is None or u2 is None or len(accepted_peaks) < 2:
        return True, cand.f_wp, cand.f_wn, None

    inter_jj = _mean_prev_jj(accepted_peaks)
    if inter_jj is None:
        return True, cand.f_wp, cand.f_wn, None

    ijjp = cand.j_pos - accepted_peaks[-1]
    ok = (
        cand.f_wn < 0.35 * u2
        and cand.f_wp > 0.35 * u1
        and ijjp < 1.35 * inter_jj
        and ijjp > 0.7 * inter_jj
    )
    if ok:
        new_u1 = 0.6 * u1 + 0.4 * cand.f_wp
        new_u2 = 0.6 * u2 + 0.4 * cand.f_wn
        return True, new_u1, new_u2, inter_jj
    return False, u1, u2, inter_jj


def _second_search(
    bcg: np.ndarray,
    cwt: np.ndarray,
    cand: Candidate,
    accepted_peaks: Sequence[int],
    u1: float,
    u2: float,
    inter_jj: float,
) -> Optional[Candidate]:
    ijjp = cand.j_pos - accepted_peaks[-1]
    umn = 0.7
    umx = 1.3

    lmin = ijjp > umn * inter_jj
    lmax = ijjp < umx * inter_jj

    if lmin and lmax:
        # The paper keeps the same zone and relaxes through Eq. (10)
        if cand.overwritten_from_best_candidate and cand.p_wn_alt is not None and cand.p_wp_alt is not None:
            p_wn4, p_wp4 = cand.p_wn_alt, cand.p_wp_alt
        else:
            p_wn4, p_wp4 = cand.p_wn, cand.p_wp
    elif not lmin:
        p_wn4, p_wp4 = _search_forward_pair(cwt, cand.j_pos)
    else:
        p_wn4, p_wp4 = _search_backward_pair(cwt, cand.j_pos)

    if p_wn4 is None or p_wp4 is None:
        p_wn4, p_wp4 = cand.p_wn, cand.p_wp

    jp2 = _argmax_between(bcg, p_wn4, p_wp4)
    if jp2 is not None:
        ijjp2 = jp2 - accepted_peaks[-1]
        cond10 = (
            ijjp2 > umn * inter_jj
            and ijjp2 < umx * inter_jj
            and cwt[p_wn4] < 0.35 * u2
            and cwt[p_wp4] > 0.35 * u1
        )
        if cond10:
            return Candidate(
                j_pos=jp2,
                p_wn=p_wn4,
                p_wp=p_wp4,
                f_wn=float(cwt[p_wn4]),
                f_wp=float(cwt[p_wp4]),
            )

    # Final relaxed rule described at the end of Section 2.3.5
    relaxed = (
        MIN_VALID_JJ_MS <= ijjp <= MAX_VALID_JJ_MS
        and cand.f_wn < 0.5 * u2
        and cand.f_wp > 0.5 * u1
    )
    if relaxed:
        return cand

    # 20% widening for later arrhythmia
    umn *= 0.8
    umx *= 1.2
    if jp2 is not None:
        ijjp2 = jp2 - accepted_peaks[-1]
        if (
            ijjp2 > umn * inter_jj
            and ijjp2 < umx * inter_jj
            and cwt[p_wn4] < 0.35 * u2
            and cwt[p_wp4] > 0.35 * u1
        ):
            return Candidate(
                j_pos=jp2,
                p_wn=p_wn4,
                p_wp=p_wp4,
                f_wn=float(cwt[p_wn4]),
                f_wp=float(cwt[p_wp4]),
            )

    return None


def bed_bcg_jpeak_detect(bcg_signal, fs: int = 125) -> np.ndarray:
    """
    Bed-only J-peak detector based on the BSPC 2024 paper.

    The user requested that preprocessing be done upstream, so this
    function assumes the input has already been prepared consistently
    with the rest of the user's pipeline.

    Parameters
    ----------
    bcg_signal : 1D array-like
        Single BCG trace.
    fs : int, default=125
        Original sampling frequency of the provided trace.

    Returns
    -------
    np.ndarray
        J-peak indices at the original sampling rate.
    """
    x = _to_numpy_1d(bcg_signal)
    n_orig = len(x)
    if n_orig == 0:
        return np.array([], dtype=int)

    x_1k = _upsample_to_1khz(x, fs)

    # Bed recordings are inverted in the paper.
    x_1k = -x_1k

    cwt, env, m_env = _compute_cwt_and_envelope(x_1k)
    if len(m_env) == 0:
        return np.array([], dtype=int)

    ensemble = _build_general_ensemble(x_1k, cwt, m_env)
    thd_n = ensemble["thd_n"] if ensemble is not None else 0.6
    thd_p = ensemble["thd_p"] if ensemble is not None else 0.5
    _ = thd_p  # retained for fidelity to the paper's notation

    accepted: List[int] = []
    u1: Optional[float] = None
    u2: Optional[float] = None

    for idx in m_env:
        cand = _first_stage_candidate(x_1k, cwt, int(idx), thd_n=thd_n, thd_p=thd_p)
        if cand is None:
            continue

        cand = _better_candidate(x_1k, cwt, cand, accepted)

        ok, u1_new, u2_new, inter_jj = _recognition_pass(cand, accepted, u1, u2)
        if ok:
            accepted.append(cand.j_pos)
            u1, u2 = u1_new, u2_new
            continue

        if u1 is None or u2 is None or inter_jj is None:
            accepted.append(cand.j_pos)
            u1, u2 = cand.f_wp, cand.f_wn
            continue

        cand2 = _second_search(x_1k, cwt, cand, accepted, u1, u2, inter_jj)
        if cand2 is not None:
            accepted.append(cand2.j_pos)
            u1 = 0.6 * u1 + 0.4 * cand2.f_wp
            u2 = 0.6 * u2 + 0.4 * cand2.f_wn

    return _downsample_indices(accepted, fs=fs, n_orig=n_orig)


if __name__ == "__main__":
    x = np.random.rand(1, 1, 625)
    signal = x[0, 0]
    peaks = bed_bcg_jpeak_detect(signal, fs=125)
    print("Input shape:", x.shape)
    print("Detected J-peaks:", peaks)
    print("Number of peaks:", len(peaks))
