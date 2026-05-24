#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Standardized Manual Review Module for ECG-BCG Signal Quality Assessment
=======================================================================

Official implementation of the standardized manual review process for 
ECG-BCG signal segment validation, as described in the manuscript.

Key validation criteria:
1. QRS truncation detection at segment boundaries
2. QRS width validation (50ms - 160ms physiological range)
3. BCG signal quality assessment for motion artifacts
4. Cross-modal synchronization verification (R-peak vs J-peak)

Usage example:
    >>> from manual_review import standardized_manual_review
    >>> is_valid, reason = standardized_manual_review(
    ...     ecg_signal, bcg_signal, qrs_mask, r_peaks, j_peaks, fs=125
    ... )
    >>> if is_valid:
    ...     print("Segment passed review")
    ... else:
    ...     print(f"Segment rejected: {reason}")
"""

import numpy as np
from itertools import groupby


__all__ = ['standardized_manual_review']


def standardized_manual_review(
    ecg_sig: np.ndarray,
    bcg_sig: np.ndarray,
    qrs_mask: np.ndarray,
    r_peaks: np.ndarray,
    j_peaks: np.ndarray,
    fs: int = 125  # Sampling rate in Hz
) -> tuple[bool, str]:
    """
    Standardized manual review process for ECG-BCG signal segment validation.
    
    Implements a 5-second paired signal manual review protocol in code form.
    
    Parameters
    ----------
    ecg_sig : np.ndarray
        Raw ECG signal
    bcg_sig : np.ndarray
        Raw BCG signal
    qrs_mask : np.ndarray
        U-Net annotated QRS region mask, where 1 indicates QRS complex, 0 otherwise
    r_peaks : np.ndarray
        ECG R-peak positions
    j_peaks : np.ndarray
        BCG J-peak positions
    fs : int, optional
        Signal sampling frequency in Hz, default 125 Hz
    
    Returns
    -------
    bool
        True if segment passes review, False if segment should be rejected
    str
        Review conclusion or rejection reason
    """
    # Criterion 1: Check for QRS truncation at segment boundaries
    if qrs_mask[0] == 1 or qrs_mask[-1] == 1:
        return False, "Segment-induced QRS Truncation"

    # Criterion 2: Validate QRS width within physiological range (50ms - 160ms)
    mask_lengths = [sum(1 for _ in group) for val, group in groupby(qrs_mask) if val == 1]
    min_qrs_len = int(0.05 * fs)
    max_qrs_len = int(0.16 * fs)

    for length in mask_lengths:
        if length < min_qrs_len or length > max_qrs_len:
            return False, "Artifact-induced False Positives"

    # Criterion 3: Detect BCG boundary noise and localized motion artifacts
    win_len = int(0.5 * fs)
    num_chunks = len(bcg_sig) // win_len

    if num_chunks == 0:
        return False, "BCG Signal Too Short"

    chunks = np.array_split(bcg_sig[:num_chunks * win_len], num_chunks)
    variances = [np.var(c) for c in chunks]
    baseline_var = np.median(variances)

    for i, var in enumerate(variances):
        if var > 3.0 * baseline_var:
            if i == 0 or i == num_chunks - 1:
                return False, "Boundary-induced Noise Intrusion"
            else:
                return False, "Localized Motion Artifacts"

    # Criterion 4: Verify cross-modal synchronization and waveform matching
    if len(r_peaks) != len(j_peaks):
        return False, "Unpaired Peaks (Severe Distortion)"

    if len(r_peaks) >= 2:
        rj_ms = (j_peaks - r_peaks) / fs * 1000
        drift = np.max(rj_ms) - np.min(rj_ms)
        if drift > 25:
            return False, "Excessive Cumulative Clock Drift"

    # All validation criteria passed
    return True, "Valid Enrolled Segment"


if __name__ == "__main__":
    # Demo: 5-second simulated data at 125Hz
    fs = 125
    test_ecg = np.random.randn(5 * fs)
    test_bcg = np.random.randn(5 * fs)
    test_qrs_mask = np.zeros(5 * fs, dtype=int)
    test_qrs_mask[50:65] = 1  # Normal QRS width (50-160ms)
    test_r = np.array([55, 180, 310])
    test_j = np.array([60, 185, 315])

    result, reason = standardized_manual_review(
        test_ecg, test_bcg, test_qrs_mask, test_r, test_j, fs
    )

    print(f"Review Result: {result}")
    print(f"Review Comment: {reason}")