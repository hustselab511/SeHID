#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Evaluation Metrics Module
=========================

This module contains evaluation metrics calculation functions for ECG/R-peak detection, including:
- Detection performance metrics: Recall, Precision, F1
- IBI error metrics: MAE, RMSE, MRE, Pr_N
- HRV metrics: SDNN, RMSSD, SDSD, pNN20, pNN50
- Poincaré metrics: SD1, SD2, SD1/SD2
- Frequency domain metrics: LF, HF, LF/HF
- BPM metrics: Heart rate calculation and error

File structure:
- train_test_util.py: Core evaluation metrics calculation
- Bpm.py: BPM metrics calculation and aggregation
- EvaluateLengthStudy.py: Evaluation for different length models
- EvaluateMultiModel_plus.py: Multi-model comparison evaluation

Usage example:
    >>> from evaluation_metrics import calculate_evaluation_metrics, calculate_hrv_metrics
    >>> metrics = calculate_evaluation_metrics(true_ecg, pred_qrs, fs=125)
    >>> hrv = calculate_hrv_metrics(ibi_array)
"""

import numpy as np
import pandas as pd
import os
from scipy.signal import welch


# ------------------- Detection Performance Metrics -------------------

def calculate_detection_metrics(true_r_peaks, pred_r_peaks, match_window_ms=75, fs=125):
    """
    Calculate R-peak detection performance metrics
    
    Parameters:
        true_r_peaks: Array of true R-peak positions
        pred_r_peaks: Array of predicted R-peak positions
        match_window_ms: Matching window size in milliseconds
        fs: Sampling rate in Hz
    
    Returns:
        dict: Contains Recall, Precision, F1 and other metrics
    """
    total_true = len(true_r_peaks)
    total_pred = len(pred_r_peaks)
    
    # R-peak matching
    matched_pairs, fn_count, fp_count = match_r_peaks(
        true_r_peaks, pred_r_peaks, fs, match_window_ms
    )
    tp_count = len(matched_pairs)
    
    # Calculate metrics
    recall = tp_count / total_true if total_true > 0 else np.nan
    precision = tp_count / total_pred if total_pred > 0 else np.nan
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else np.nan
    fnr = fn_count / total_true if total_true > 0 else np.nan
    fpr = fp_count / total_pred if total_pred > 0 else np.nan
    
    return {
        'Total_True_R_Peaks': total_true,
        'Total_Pred_R_Peaks': total_pred,
        'Matched_R_Pairs': tp_count,
        'False_Negative': fn_count,
        'False_Positive': fp_count,
        'FNR': fnr,
        'FPR': fpr,
        'Recall': recall,
        'Precision': precision,
        'F1': f1
    }


def match_r_peaks(true_r_peaks, pred_r_peaks, fs=125, match_window_ms=75):
    """
    R-peak matching algorithm
    
    Parameters:
        true_r_peaks: Array of true R-peak positions
        pred_r_peaks: Array of predicted R-peak positions
        fs: Sampling rate in Hz
        match_window_ms: Matching window size in milliseconds
    
    Returns:
        matched_pairs: List of matched R-peak pairs
        fn_count: Number of false negatives
        fp_count: Number of false positives
    """
    match_window_samples = int(match_window_ms * fs / 1000)
    matched_pairs = []
    used_pred = set()
    
    for true_peak in true_r_peaks:
        best_pred_idx = None
        min_distance = float('inf')
        
        for pred_idx, pred_peak in enumerate(pred_r_peaks):
            if pred_idx in used_pred:
                continue
            distance = abs(true_peak - pred_peak)
            if distance <= match_window_samples and distance < min_distance:
                min_distance = distance
                best_pred_idx = pred_idx
        
        if best_pred_idx is not None:
            matched_pairs.append((true_peak, pred_r_peaks[best_pred_idx]))
            used_pred.add(best_pred_idx)
    
    fn_count = len(true_r_peaks) - len(matched_pairs)
    fp_count = len(pred_r_peaks) - len(used_pred)
    
    return matched_pairs, fn_count, fp_count


# ------------------- IBI Error Metrics -------------------

def calculate_ibi_mae(true_ibi, pred_ibi):
    """Calculate Mean Absolute Error (MAE) of IBI"""
    if len(true_ibi) == 0 or len(pred_ibi) == 0:
        return np.nan
    min_len = min(len(true_ibi), len(pred_ibi))
    return np.mean(np.abs(true_ibi[:min_len] - pred_ibi[:min_len]))


def calculate_ibi_rmse(true_ibi, pred_ibi):
    """Calculate Root Mean Squared Error (RMSE) of IBI"""
    if len(true_ibi) == 0 or len(pred_ibi) == 0:
        return np.nan
    min_len = min(len(true_ibi), len(pred_ibi))
    return np.sqrt(np.mean((true_ibi[:min_len] - pred_ibi[:min_len]) ** 2))


def calculate_ibi_mre(true_ibi, pred_ibi):
    """Calculate Mean Relative Error (MRE) of IBI, returns percentage"""
    if len(true_ibi) == 0 or len(pred_ibi) == 0:
        return np.nan
    min_len = min(len(true_ibi), len(pred_ibi))
    true_aligned = true_ibi[:min_len]
    pred_aligned = pred_ibi[:min_len]
    true_aligned_safe = np.maximum(true_aligned, 1e-6)  # Avoid division by zero
    relative_errors = np.abs(true_aligned - pred_aligned) / true_aligned_safe
    return np.mean(relative_errors) * 100


def calculate_pr_metric(true_r_peaks, pred_r_peaks, fs=125, threshold_ms=10):
    """Calculate Pr_N metric (percentage of errors within N milliseconds)"""
    if len(true_r_peaks) == 0 or len(pred_r_peaks) == 0:
        return np.nan
    
    match_window_samples = int(threshold_ms * fs / 1000)
    matched_count = 0
    
    for true_peak in true_r_peaks:
        for pred_peak in pred_r_peaks:
            if abs(true_peak - pred_peak) <= match_window_samples:
                matched_count += 1
                break
    
    return matched_count / len(true_r_peaks)


# ------------------- HRV Time Domain Metrics -------------------

def calculate_hrv_metrics(ibi):
    """
    Calculate HRV time domain metrics
    
    Parameters:
        ibi: RR interval array in seconds
    
    Returns:
        dict: Contains SDNN, RMSSD, SDSD, pNN20, pNN50 and other metrics
    """
    if len(ibi) < 2:
        return {
            'NNI_counter': np.nan, 'NNI_max': np.nan, 'NNI_min': np.nan,
            'NNI_mean': np.nan, 'SDNN': np.nan,
            'SDSD': np.nan, 'RMSSD': np.nan, 'pNN20': np.nan, 'pNN50': np.nan
        }
    
    # Basic IBI metrics
    NNI_counter = len(ibi)
    NNI_max = np.max(ibi) * 1000  # Convert to milliseconds
    NNI_min = np.min(ibi) * 1000
    NNI_mean = np.mean(ibi) * 1000
    
    # Variability metrics
    SDNN = np.std(ibi) * 1000
    diff_ibi = np.diff(ibi) * 1000  # Differences between consecutive RR intervals (ms)
    SDSD = np.std(diff_ibi)
    RMSSD = np.sqrt(np.mean(diff_ibi ** 2))
    
    # Percentage metrics
    pNN20 = np.sum(np.abs(diff_ibi) > 20) / len(diff_ibi)
    pNN50 = np.sum(np.abs(diff_ibi) > 50) / len(diff_ibi)
    
    return {
        'NNI_counter': NNI_counter, 'NNI_max': NNI_max, 'NNI_min': NNI_min,
        'NNI_mean': NNI_mean, 'SDNN': SDNN,
        'SDSD': SDSD, 'RMSSD': RMSSD, 'pNN20': pNN20, 'pNN50': pNN50
    }


# ------------------- Poincaré Metrics (SD1, SD2) -------------------

def calculate_poincare_metrics(ibi):
    """
    Calculate Poincaré plot metrics
    
    Parameters:
        ibi: RR interval array in seconds
    
    Returns:
        dict: Contains SD1, SD2, SD1/SD2 ratio
    """
    if len(ibi) < 3:
        return {
            'SD1': np.nan,
            'SD2': np.nan,
            'SD1_SD2_ratio': np.nan
        }
    
    # Convert to milliseconds
    ibi_ms = ibi * 1000
    
    # Poincaré plot coordinates
    x = ibi_ms[:-1]
    y = ibi_ms[1:]
    
    # Calculate SD1 and SD2
    diff = x - y
    SD1 = np.sqrt(np.mean(diff ** 2) / 2)
    
    sum_vals = x + y
    SD2 = np.sqrt(np.mean(sum_vals ** 2) / 2)
    
    # Calculate SD1/SD2 ratio
    SD1_SD2_ratio = SD1 / SD2 if SD2 > 0 else np.nan
    
    return {
        'SD1': SD1,
        'SD2': SD2,
        'SD1_SD2_ratio': SD1_SD2_ratio
    }


# ------------------- Frequency Domain Metrics (LF, HF) -------------------

def calculate_frequency_domain_metrics(ibi, fs=4):
    """
    Calculate frequency domain metrics (LF, HF, LF/HF)
    
    Parameters:
        ibi: RR interval array in seconds
        fs: Resampling frequency (default 4Hz)
    
    Returns:
        dict: Contains LF, HF, LF/HF ratio
    """
    # Frequency domain analysis requires at least 30 IBI samples for reliable results
    if len(ibi) < 30:
        return {
            'LF': np.nan,
            'HF': np.nan,
            'LF_HF_ratio': np.nan
        }
    
    avg_ibi = np.mean(ibi)
    if avg_ibi <= 0:
        return {
            'LF': np.nan,
            'HF': np.nan,
            'LF_HF_ratio': np.nan
        }
    
    try:
        # Use appropriate window size (at least 64 points)
        nperseg = max(64, len(ibi) // 2)
        freqs, psd = welch(ibi, fs=fs, nperseg=nperseg)
        
        # LF: 0.04-0.15 Hz
        lf_mask = (freqs >= 0.04) & (freqs <= 0.15)
        if len(psd[lf_mask]) > 0:
            freq_resolution = np.mean(np.diff(freqs)) if len(freqs) > 1 else 0.01
            LF = np.sum(psd[lf_mask] * freq_resolution)
        else:
            LF = np.nan
        
        # HF: 0.15-0.4 Hz
        hf_mask = (freqs >= 0.15) & (freqs <= 0.4)
        if len(psd[hf_mask]) > 0:
            freq_resolution = np.mean(np.diff(freqs)) if len(freqs) > 1 else 0.01
            HF = np.sum(psd[hf_mask] * freq_resolution)
        else:
            HF = np.nan
        
        # LF/HF ratio
        LF_HF_ratio = LF / HF if HF > 0 else np.nan
        
    except Exception as e:
        print(f"Frequency domain calculation error: {e}")
        return {
            'LF': np.nan,
            'HF': np.nan,
            'LF_HF_ratio': np.nan
        }
    
    return {
        'LF': LF,
        'HF': HF,
        'LF_HF_ratio': LF_HF_ratio
    }


# ------------------- BPM Metrics -------------------

def calculate_bpm_from_ibi(ibi):
    """
    Calculate BPM from IBI
    
    Parameters:
        ibi: RR interval array in seconds
    
    Returns:
        float: BPM value
    """
    if len(ibi) < 2:
        return np.nan
    avg_ibi = np.mean(ibi)
    return 60 / avg_ibi if avg_ibi > 0 else np.nan


def calculate_bpm_metrics(true_ibi, pred_ibi):
    """
    Calculate BPM-related metrics
    
    Parameters:
        true_ibi: True RR interval array in seconds
        pred_ibi: Predicted RR interval array in seconds
    
    Returns:
        dict: Contains true_bpm, pred_bpm, mae, rmse
    """
    true_bpm = calculate_bpm_from_ibi(true_ibi)
    pred_bpm = calculate_bpm_from_ibi(pred_ibi)
    
    mae = np.nan
    rmse = np.nan
    if not (np.isnan(true_bpm) or np.isnan(pred_bpm)):
        bpm_diff = true_bpm - pred_bpm
        mae = abs(bpm_diff)
        rmse = np.sqrt(bpm_diff ** 2)
    
    return {
        'true_bpm': true_bpm,
        'pred_bpm': pred_bpm,
        'mae': mae,
        'rmse': rmse
    }


# ------------------- Comprehensive Evaluation Function -------------------

def calculate_evaluation_metrics(true_r_peaks, pred_r_peaks, fs=125, match_window_ms=75):
    """
    Comprehensive evaluation function: calculate all metrics
    
    Parameters:
        true_r_peaks: Array of true R-peak positions
        pred_r_peaks: Array of predicted R-peak positions
        fs: Sampling rate in Hz
        match_window_ms: Matching window size in milliseconds
    
    Returns:
        dict: Contains all evaluation metrics
    """
    # Detection performance metrics
    detection_metrics = calculate_detection_metrics(
        true_r_peaks, pred_r_peaks, match_window_ms, fs
    )
    
    # Calculate IBI
    true_ibi = np.diff(true_r_peaks) / fs if len(true_r_peaks) >= 2 else np.array([])
    pred_ibi = np.diff(pred_r_peaks) / fs if len(pred_r_peaks) >= 2 else np.array([])
    
    # IBI error metrics
    ibi_metrics = {
        'MAE': calculate_ibi_mae(true_ibi, pred_ibi),
        'RMSE': calculate_ibi_rmse(true_ibi, pred_ibi),
        'MRE': calculate_ibi_mre(true_ibi, pred_ibi),
        'Pr_10': calculate_pr_metric(true_r_peaks, pred_r_peaks, fs, 10),
        'Pr_20': calculate_pr_metric(true_r_peaks, pred_r_peaks, fs, 20),
        'Pr_30': calculate_pr_metric(true_r_peaks, pred_r_peaks, fs, 30),
        'Pr_50': calculate_pr_metric(true_r_peaks, pred_r_peaks, fs, 50)
    }
    
    # HRV time domain metrics error
    true_hrv = calculate_hrv_metrics(true_ibi)
    pred_hrv = calculate_hrv_metrics(pred_ibi)
    hrv_errors = {
        'NNI_max_error': np.abs(pred_hrv['NNI_max'] - true_hrv['NNI_max']),
        'NNI_min_error': np.abs(pred_hrv['NNI_min'] - true_hrv['NNI_min']),
        'NNI_mean_error': np.abs(pred_hrv['NNI_mean'] - true_hrv['NNI_mean']),
        'SDNN_error': np.abs(pred_hrv['SDNN'] - true_hrv['SDNN']),
        'SDSD_error': np.abs(pred_hrv['SDSD'] - true_hrv['SDSD']),
        'RMSSD_error': np.abs(pred_hrv['RMSSD'] - true_hrv['RMSSD']),
    }
    
    # Poincaré metrics error
    true_poincare = calculate_poincare_metrics(true_ibi)
    pred_poincare = calculate_poincare_metrics(pred_ibi)
    poincare_errors = {
        'SD1_error': np.abs(pred_poincare['SD1'] - true_poincare['SD1']),
        'SD2_error': np.abs(pred_poincare['SD2'] - true_poincare['SD2']),
        'SD1_SD2_ratio_error': np.abs(pred_poincare['SD1_SD2_ratio'] - true_poincare['SD1_SD2_ratio']),
    }
    
    # Frequency domain metrics error
    true_freq = calculate_frequency_domain_metrics(true_ibi)
    pred_freq = calculate_frequency_domain_metrics(pred_ibi)
    freq_errors = {
        'LF_error': np.abs(pred_freq['LF'] - true_freq['LF']),
        'HF_error': np.abs(pred_freq['HF'] - true_freq['HF']),
        'LF_HF_ratio_error': np.abs(pred_freq['LF_HF_ratio'] - true_freq['LF_HF_ratio']),
    }
    
    # BPM metrics
    bpm_metrics = calculate_bpm_metrics(true_ibi, pred_ibi)
    
    # Heart rate metrics (based on R-peak count)
    hr_metrics = {
        'True_HR': len(true_r_peaks),
        'Pred_HR': len(pred_r_peaks),
        'HR_error': abs(len(pred_r_peaks) - len(true_r_peaks))
    }
    
    # Combine all metrics
    all_metrics = {
        **detection_metrics, 
        **ibi_metrics, 
        **hrv_errors, 
        **poincare_errors,
        **freq_errors,
        **bpm_metrics, 
        **hr_metrics
    }
    
    return all_metrics


# ------------------- Metrics Aggregation Functions -------------------

def aggregate_metrics(metrics_list):
    """
    Aggregate metrics from multiple samples
    
    Parameters:
        metrics_list: List of metrics dictionaries
    
    Returns:
        dict: Aggregated metrics (mean values)
    """
    if not metrics_list:
        return {}
    
    # Extract all metric keys
    all_keys = set(key for metrics in metrics_list for key in metrics.keys())
    
    aggregated = {}
    for key in all_keys:
        values = [m[key] for m in metrics_list if key in m and not np.isnan(m[key])]
        if values:
            aggregated[key] = np.mean(values)
        else:
            aggregated[key] = np.nan
    
    return aggregated


def save_metrics_to_csv(metrics, file_path):
    """
    Save metrics to CSV file
    
    Parameters:
        metrics: Metrics dictionary or list of dictionaries
        file_path: Output file path
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    if isinstance(metrics, list):
        df = pd.DataFrame(metrics)
    else:
        df = pd.DataFrame([metrics])
    
    # Add mean row
    mean_row = df.mean(numeric_only=True).to_dict()
    mean_row['sample_idx'] = 'mean'
    df.loc[len(df)] = mean_row
    
    df.to_csv(file_path, index=False)


# ------------------- Test Code -------------------

def test_evaluation_metrics():
    """Test evaluation metrics calculation"""
    print("=== Testing Evaluation Metrics ===")
    
    # Generate more R-peaks for frequency domain testing (need at least 30 IBIs)
    fs = 125
    duration = 30  # 30 seconds
    num_beats = 35  # 35 R-peaks produce 34 IBIs
    true_r_peaks = np.linspace(125, duration * fs, num_beats).astype(int)
    pred_r_peaks = true_r_peaks + np.random.randint(-3, 4, len(true_r_peaks))  # Small deviations
    
    print(f"Number of true R-peaks: {len(true_r_peaks)}")
    print(f"Number of predicted R-peaks: {len(pred_r_peaks)}")
    print(f"Number of IBIs: {len(np.diff(true_r_peaks))}")
    
    # Calculate metrics
    metrics = calculate_evaluation_metrics(true_r_peaks, pred_r_peaks, fs=fs)
    
    print("\nDetection Performance Metrics:")
    for key in ['Recall', 'Precision', 'F1']:
        print(f"  {key}: {metrics.get(key, np.nan):.4f}")
    
    print("\nIBI Error Metrics:")
    for key in ['MAE', 'RMSE', 'MRE']:
        print(f"  {key}: {metrics.get(key, np.nan):.4f}")
    
    print("\nHRV Error Metrics:")
    for key in ['SDNN_error', 'RMSSD_error', 'SDSD_error']:
        print(f"  {key}: {metrics.get(key, np.nan):.4f}")
    
    print("\nPoincaré Error Metrics:")
    for key in ['SD1_error', 'SD2_error', 'SD1_SD2_ratio_error']:
        print(f"  {key}: {metrics.get(key, np.nan):.4f}")
    
    print("\nFrequency Domain Error Metrics:")
    for key in ['LF_error', 'HF_error', 'LF_HF_ratio_error']:
        value = metrics.get(key, np.nan)
        if np.isnan(value):
            print(f"  {key}: NaN (requires at least 30 IBIs)")
        else:
            print(f"  {key}: {value:.6f}")
    
    print("\nBPM Metrics:")
    for key in ['true_bpm', 'pred_bpm', 'mae', 'rmse']:
        print(f"  {key}: {metrics.get(key, np.nan):.4f}")
    
    print("\n=== Testing Complete ===")


if __name__ == '__main__':
    test_evaluation_metrics()
