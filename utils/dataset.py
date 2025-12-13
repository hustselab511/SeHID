import torch
import numpy as np
from torch.utils.data import Dataset
from typing import Tuple, Optional, Dict, Any, List

__all__ = ['SETUP_SEED', 'DataAugmentor', 'BCGAugmentDataset']

def SETUP_SEED(seed = 114514):
    import numpy as np
    import random
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


import glob
import os
import shutil
import numpy as np
from sklearn.model_selection import train_test_split


def load_data_or_generate_dummy(data_root, num_dummy=100, seq_len=625):
    """
    Load data from the specified directory.
    If the directory does not exist or is empty, generate dummy data for demonstration.

    Args:
        data_root (str): Path to the dataset directory.
        num_dummy (int): Number of dummy samples to generate if data is missing.
        seq_len (int): Sequence length for dummy data.

    Returns:
        tuple: (train_signals, train_labels, val_signals, val_labels) - Lists of file paths.
    """

    # 1. Try to load real data
    # Assumption: The dataset folder contains 'signals' and 'labels' subfolders
    # with matching .npy filenames.
    if os.path.exists(data_root):
        print(f"[*] Scanning data from: {data_root}")
        # Search for .npy files
        # Adjust the glob pattern if your file structure is different (e.g., recursive search)
        signals = sorted(glob.glob(os.path.join(data_root, "signals", "*.npy")))
        labels = sorted(glob.glob(os.path.join(data_root, "labels", "*.npy")))

        if len(signals) > 0 and len(signals) == len(labels):
            print(f"[*] Found {len(signals)} samples. Splitting into Train/Val...")
            return train_test_split(signals, labels, test_size=0.2, random_state=42)
        else:
            print(
                f"[!] Warning: Data root exists but is empty or unmatched. ({len(signals)} signals, {len(labels)} labels)")

    # 2. Fallback: Generate Dummy Data (For GitHub Demo purposes)
    print("=" * 50)
    print("[!] No real data found. Switching to DEMO MODE.")
    print("[!] Generating dummy data to verify the training pipeline...")
    print("=" * 50)

    dummy_dir = "./dummy_data_cache"
    if os.path.exists(dummy_dir):
        shutil.rmtree(dummy_dir)  # Clean up previous run

    os.makedirs(os.path.join(dummy_dir, "signals"), exist_ok=True)
    os.makedirs(os.path.join(dummy_dir, "labels"), exist_ok=True)

    dummy_signals = []
    dummy_labels = []

    for i in range(num_dummy):
        # Create random signal: [Length, 2] (ECG, BCG)
        # Note: This is random noise! The model will NOT learn anything meaningful.
        sig = np.random.randn(seq_len, 2).astype(np.float32)
        lbl = np.random.randint(0, 2, (seq_len,)).astype(np.float32)

        s_path = os.path.join(dummy_dir, "signals", f"sample_{i}.npy")
        l_path = os.path.join(dummy_dir, "labels", f"sample_{i}.npy")

        np.save(s_path, sig)
        np.save(l_path, lbl)

        dummy_signals.append(s_path)
        dummy_labels.append(l_path)

    print(f"[*] Generated {num_dummy} dummy samples in '{dummy_dir}'.")

    # Split the dummy data
    return train_test_split(dummy_signals, dummy_labels, test_size=0.2, random_state=42)


class DataAugmentor:
    """
    Data augmentation pipeline for 1D biomedical signals (ECG/BCG).
    Supports time shifting, noise injection, amplitude scaling,
    random masking, and frequency domain perturbation.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Args:
            config: Dictionary containing augmentation hyperparameters.
        """
        # Default configuration
        self.cfg = {
            # Time Shift
            'use_time_shift': True,
            'shift_range': (-3, 4),
            'shift_prob': 0.5,

            # Noise Injection
            'use_noise': True,
            'noise_std': 0.05,
            'noise_prob': 0.5,

            # Amplitude Scaling
            'use_scale': True,
            'scale_range': (0.8, 1.2),
            'scale_prob': 0.5,

            # Random Masking
            'use_mask': True,
            'mask_prob': 0.3,
            'mask_ratio': 0.1,

            # Frequency Perturbation
            'use_freq_aug': True,
            'freq_prob': 0.3,
            'freq_ratio': 0.1
        }

        # Update defaults with user config
        if config:
            self.cfg.update(config)

    def time_shift(self, signal: np.ndarray, label: Optional[np.ndarray], shift: int) -> Tuple[
        np.ndarray, Optional[np.ndarray]]:
        """Applies circular time shift to signal and label."""
        if shift == 0:
            return signal, label

        shifted_signal = np.roll(signal, shift, axis=-1)
        shifted_label = np.roll(label, shift, axis=-1) if label is not None else None
        return shifted_signal, shifted_label

    def add_noise(self, signal: np.ndarray) -> np.ndarray:
        """Adds Gaussian noise to the signal."""
        noise = np.random.normal(0, self.cfg['noise_std'], size=signal.shape).astype(np.float32)
        return signal + noise

    def scale_amplitude(self, signal: np.ndarray) -> np.ndarray:
        """Scales the signal amplitude by a random factor."""
        scale = np.random.uniform(*self.cfg['scale_range'])
        return signal * scale

    def random_mask(self, signal: np.ndarray) -> np.ndarray:
        """Randomly zeros out a segment of the signal."""
        # Avoid modifying the original array in place if it's shared
        signal = signal.copy()
        T = signal.shape[1]
        mask_len = int(T * self.cfg['mask_ratio'])

        if mask_len > 0:
            start = np.random.randint(0, T - mask_len)
            signal[:, start:start + mask_len] = 0

        return signal

    def freq_augment(self, signal: np.ndarray) -> np.ndarray:
        """
        Perturbs the signal in the frequency domain.
        Multiplies a random frequency band by a random factor.
        """
        # FFT
        fft_sig = np.fft.fft(signal)
        N = len(fft_sig)

        # Select random frequency band
        freq_len = int(N * self.cfg['freq_ratio'])
        if freq_len > 0:
            start = np.random.randint(0, N - freq_len)
            factor = np.random.uniform(0.5, 1.5)

            # Apply scaling to the band
            fft_sig[:, start:start + freq_len] *= factor

        # IFFT (Take real part)
        return np.fft.ifft(fft_sig).real.astype(np.float32)

    def __call__(self, ecg: np.ndarray, bcg: np.ndarray, qrs_label: np.ndarray) -> Tuple[
        np.ndarray, np.ndarray, np.ndarray]:
        """
        Apply augmentation pipeline to the data samples.
        """
        # 1. Time Shift (Synchronized)
        if self.cfg['use_time_shift'] and np.random.rand() < self.cfg['shift_prob']:
            shift = np.random.randint(*self.cfg['shift_range'])
            ecg, qrs_label = self.time_shift(ecg, qrs_label, shift)
            bcg, _ = self.time_shift(bcg, None, shift)

        # 2. Add Noise (BCG only)
        if self.cfg['use_noise'] and np.random.rand() < self.cfg['noise_prob']:
            bcg = self.add_noise(bcg)

        # 3. Amplitude Scaling (BCG only)
        if self.cfg['use_scale'] and np.random.rand() < self.cfg['scale_prob']:
            bcg = self.scale_amplitude(bcg)

        # 4. Random Masking (Independent)
        if self.cfg['use_mask'] and np.random.rand() < self.cfg['mask_prob']:
            bcg = self.random_mask(bcg)
            ecg = self.random_mask(ecg)
            # Masking the label ensures we don't punish the model for missing masked QRS complexes
            qrs_label = self.random_mask(qrs_label)

        # 5. Frequency Perturbation (Independent)
        if self.cfg['use_freq_aug'] and np.random.rand() < self.cfg['freq_prob']:
            bcg = self.freq_augment(bcg)
            ecg = self.freq_augment(ecg)

        return ecg, bcg, qrs_label


class BCGAugmentDataset(Dataset):
    """
    PyTorch Dataset for loading and augmenting BCG/ECG data pairs.
    """

    def __init__(self, data_paths: List[str], label_paths: List[str],
                 is_train: bool = True, augment_config: Optional[Dict[str, Any]] = None):
        """
        Args:
            data_paths: List of file paths for signal data (.npy containing [ECG, BCG]).
            label_paths: List of file paths for label data (.npy).
            is_train: Boolean flag. If True, applies data augmentation.
            augment_config: Dictionary to override default augmentation parameters.
        """
        self.data_paths = data_paths
        self.label_paths = label_paths
        self.is_train = is_train

        # Initialize augmentor with merged config
        self.augmentor = DataAugmentor(augment_config)

    def __len__(self) -> int:
        return len(self.data_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Load data
        # Assumption: signal.npy shape is (Length, Channels) -> [ECG, BCG]
        try:
            signal = np.load(self.data_paths[idx])
            label_data = np.load(self.label_paths[idx])
        except Exception as e:
            print(f"Error loading index {idx}: {e}")
            # Return zeros or handle error appropriately (here re-raising for visibility)
            raise e

        # Preprocess / Reshape to (Channels, Length)
        # Assuming signal[:, 0] is ECG and signal[:, 1] is BCG
        ecg = signal[:, 0].astype(np.float32).reshape(1, -1)
        bcg = signal[:, 1].astype(np.float32).reshape(1, -1)
        qrs_label = label_data.astype(np.float32).reshape(1, -1)

        # Apply Augmentation (only during training)
        if self.is_train:
            ecg, bcg, qrs_label = self.augmentor(ecg, bcg, qrs_label)

        # Convert to PyTorch Tensors
        return (
            torch.from_numpy(ecg),
            torch.from_numpy(qrs_label),
            torch.from_numpy(bcg)
        )