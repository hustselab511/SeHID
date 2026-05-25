#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
J-peak Quality Assessment Module
===============================

Standardized manual review for BCG J-peak detection quality validation.

Key validation criteria:
1. Discernible local amplitude peak within 150-350ms search window
2. No ambiguous peaks (multiple peaks with similar amplitudes)

Usage example:
    >>> from manual_review import validate_j_peak_quality
    >>> is_valid, reason = validate_j_peak_quality(bcg_signal, r_peaks, j_peaks, fs=125)
"""

import numpy as np
import scipy.signal as signal


__all__ = ['validate_j_peak_quality']


def validate_j_peak_quality(
    bcg_sig: np.ndarray,
    r_peaks: np.ndarray,
    j_peaks: np.ndarray,
    fs: int = 125,
    j_search_window_ms: tuple = (150, 350),
    peak_ambiguity_threshold: float = 0.9,
    peak_to_avg_ratio_threshold: float = 0.5
) -> tuple[bool, str]:
    """
    Validate J-peak detection quality within the 150-350ms search window.
    
    Parameters
    ----------
    bcg_sig : np.ndarray
        BCG signal array
    r_peaks : np.ndarray
        ECG R-peak positions (indices)
    j_peaks : np.ndarray
        Detected BCG J-peak positions (indices)
    fs : int, optional
        Sampling rate in Hz, default 125
    j_search_window_ms : tuple, optional
        J-peak search window in ms after R-peak, default (150, 350)
    peak_ambiguity_threshold : float, optional
        Threshold for peak ambiguity detection, default 0.9 (90% - 10% difference)
    peak_to_avg_ratio_threshold : float, optional
        Threshold for peak-to-average amplitude ratio, default 0.5 (50%)
    
    Returns
    -------
    bool
        True if J-peaks are valid, False if review required
    str
        Validation result or rejection reason
    """
    # Check peak count consistency
    if len(r_peaks) != len(j_peaks):
        return False, "Unpaired R/J Peaks"
    
    # Validate each J-peak within its search window
    for r_peak, j_peak in zip(r_peaks, j_peaks):
        # Calculate search window boundaries
        window_start = r_peak + int(j_search_window_ms[0] / 1000 * fs)
        window_end = r_peak + int(j_search_window_ms[1] / 1000 * fs)
        
        # Check window bounds
        if window_start < 0 or window_end > len(bcg_sig):
            return False, "J-peak Search Window Out of Bounds"
        
        # Extract search window
        search_window = bcg_sig[window_start:window_end]
        
        # Find all local peaks in the search window
        peak_indices, _ = signal.find_peaks(search_window)
        
        # Criterion 1: Check for discernible local peak
        if len(peak_indices) == 0:
            return False, "No Discernible Local Peak in J-search Window"
        
        # Get peak amplitudes
        peak_amplitudes = search_window[peak_indices]
        
        # Criterion 1a: Check peak amplitude relative to window average
        max_amp = np.max(peak_amplitudes)
        window_avg = np.mean(np.abs(search_window))
        
        if max_amp < peak_to_avg_ratio_threshold * window_avg:
            return False, "Peak Amplitude Below 50% of Window Average"
        
        # Criterion 2: Check for ambiguous peaks (multiple similar amplitudes)
        if len(peak_amplitudes) >= 2:
            max_amp = np.max(peak_amplitudes)
            sorted_amps = np.sort(peak_amplitudes)
            second_max_amp = sorted_amps[-2]
            
            if second_max_amp > peak_ambiguity_threshold * max_amp:
                return False, "Ambiguous Peak Detection (Multiple Similar Amplitudes)"
    
    # All validations passed
    return True, "Valid J-peak Detection"


if __name__ == "__main__":
    fs = 125
    bcg = np.random.randn(5 * fs)
    
    # Simulate valid peaks
    r_peaks = np.array([150, 300, 450])
    j_peaks = r_peaks + int(250 / 1000 * fs)  # J-peak at 250ms after R-peak
    
    result, reason = validate_j_peak_quality(bcg, r_peaks, j_peaks, fs)
    print(f"Validation Result: {result}")
    print(f"Reason: {reason}")