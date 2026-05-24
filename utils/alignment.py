#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Kansas Dataset ECG-BCG Alignment Module
=======================================

Official implementation for ECG-BCG signal alignment following the manuscript description:

1. Sampling rate: 1000 Hz
2. Search BCG J-peak 150–350 ms AFTER the first ECG R-peak
3. Align by cropping BCG to match R-peak and J-peak positions
4. Unreliable segments are excluded by standardized manual review

Usage example:
    >>> from alignment import align_by_crop_bcg, detect_first_r_peak, detect_j_peak_in_150_350ms
    >>> ecg_aligned, bcg_aligned = align_by_crop_bcg(ecg_signal, bcg_signal, fs=1000)
    >>> if ecg_aligned is not None:
    ...     print("Alignment successful")
    ... else:
    ...     print("Alignment failed - sample requires manual review")
"""

import numpy as np
import scipy.signal as signal


# ======================
# Fixed Parameters (Per Manuscript Specification)
# ======================
FS = 1000  # Kansas dataset true sampling rate
J_SEARCH_WINDOW_MS = (150, 350)  # Search J-peak 150~350ms after R-peak


__all__ = [
    'FS',
    'J_SEARCH_WINDOW_MS',
    'detect_first_r_peak',
    'detect_j_peak_in_150_350ms',
    'align_by_crop_bcg'
]


def detect_first_r_peak(ecg, fs=FS):
    """
    Detect the first valid R-peak in ECG signal.
    
    Parameters:
        ecg: ECG signal array
        fs: Sampling rate in Hz (default: 1000)
    
    Returns:
        int: Index of the first R-peak, or None if no peak detected
    """
    distance = int(0.5 * fs)  # Minimum heartbeat interval 0.5s
    peaks, _ = signal.find_peaks(ecg, distance=distance)
    if len(peaks) == 0:
        return None
    return peaks[0]


def detect_j_peak_in_150_350ms(bcg, r_idx, fs=FS):
    """
    Find BCG J-peak within the window [150-350ms] AFTER the first R-peak.
    Strictly follows the manuscript specification.
    
    Parameters:
        bcg: BCG signal array
        r_idx: Index of the detected R-peak
        fs: Sampling rate in Hz (default: 1000)
    
    Returns:
        int: Index of the J-peak, or None if not found within window
    """
    start = r_idx + int(J_SEARCH_WINDOW_MS[0] / 1000 * fs)
    end = r_idx + int(J_SEARCH_WINDOW_MS[1] / 1000 * fs)

    if start >= len(bcg) or end >= len(bcg):
        return None

    window = bcg[start:end]
    j_local = np.argmax(window)
    j_idx = start + j_local
    return j_idx


def align_by_crop_bcg(ecg, bcg, fs=FS):
    """
    Core alignment function:
    1. Detect first R-peak in ECG
    2. Search for J-peak in the 150-350ms window after R-peak
    3. Crop BCG to align J-peak with R-peak position
    
    Parameters:
        ecg: ECG signal array
        bcg: BCG signal array
        fs: Sampling rate in Hz (default: 1000)
    
    Returns:
        tuple: (ecg_aligned, bcg_aligned)
               Returns (None, None) if alignment fails (requires manual review)
    """
    # Step 1: Detect first R-peak
    r_idx = detect_first_r_peak(ecg, fs)
    if r_idx is None:
        return None, None  # Manual review required

    # Step 2: Search J-peak in 150~350ms window after R-peak
    j_idx = detect_j_peak_in_150_350ms(bcg, r_idx, fs)
    if j_idx is None:
        return None, None  # Manual review required

    # Step 3: Calculate shift offset (J-peak position - R-peak position)
    shift = j_idx - r_idx

    # Step 4: Crop BCG to align J-peak with R-peak position
    bcg_aligned = bcg[shift:]
    ecg_aligned = ecg[:len(bcg_aligned)]

    return ecg_aligned, bcg_aligned


if __name__ == "__main__":
    fs = 1000
    ecg = np.random.randn(10 * fs)
    bcg = np.random.randn(10 * fs)

    ecg_ali, bcg_ali = align_by_crop_bcg(ecg, bcg, fs)

    if ecg_ali is not None:
        print("Alignment completed (R-peak <-> J-peak 150-350ms window)")
    else:
        print("Invalid sample, requires manual review")