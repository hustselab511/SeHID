import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from scipy.signal import resample_poly
from scipy.ndimage import gaussian_filter1d
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import wfdb

# Import the refactored model
# Ensure QRSModel.py is in the same directory
from QRSUnet import QRSUNet


def set_seed(seed=42):
    """Set random seed for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class LUDBDataset(Dataset):
    """
    Dataset class for LUDB (Lobachevsky University Electrocardiography Database).
    Handles loading, preprocessing, and label generation for QRS segmentation.
    """

    def __init__(self, db_dir, record_ids=None, fs=500, target_fs=500):
        self.db_dir = db_dir
        self.fs = fs
        self.target_fs = target_fs
        self.downsample_factor = fs // target_fs
        self.scaler = StandardScaler()

        # LUDB typically has records 1-200
        if record_ids is None:
            self.record_ids = list(range(1, 201))
        else:
            self.record_ids = [i for i in record_ids if 1 <= i <= 200]

    def __len__(self):
        return len(self.record_ids)

    def preprocess_signal(self, ecg_signal):
        """
        Applies baseline removal, denoising, resampling, and normalization.
        """
        # 1. Baseline removal using Moving Average
        window_size = int(0.2 * self.fs)
        baseline = np.convolve(ecg_signal, np.ones(window_size) / window_size, mode='same')
        ecg_filtered = ecg_signal - baseline

        # 2. Gaussian filtering (Simulating Non-Local Means denoising)
        ecg_filtered = gaussian_filter1d(ecg_filtered, sigma=2)

        # 3. Resample to target frequency
        if self.fs != self.target_fs:
            ecg_filtered = resample_poly(ecg_filtered, self.target_fs, self.fs)

        # 4. Z-score Standardization
        # Reshape for scaler, then flatten back
        ecg_filtered = self.scaler.fit_transform(ecg_filtered.reshape(-1, 1)).flatten()

        return ecg_filtered

    def add_noise(self, ecg_signal):
        """
        Data Augmentation: Adds synthetic noise to the signal.
        """
        # 1. Random Periodic Spike Noise
        if np.random.rand() > 0.5:
            spike_pos = np.random.randint(0, len(ecg_signal))
            spike_width = np.random.randint(5, 20)
            spike_height = np.random.uniform(0.1, 0.5) * np.max(ecg_signal)
            # Ensure indices remain within bounds
            end_pos = min(spike_pos + spike_width, len(ecg_signal))
            ecg_signal[spike_pos:end_pos] += spike_height

        # 2. Gaussian White Noise
        noise_level = np.random.uniform(0.01, 0.05)
        ecg_signal += noise_level * np.random.randn(len(ecg_signal))

        # 3. Random Baseline Flip
        if np.random.rand() > 0.5:
            ecg_signal = -ecg_signal

        return ecg_signal

    def create_labels(self, ecg_length, annotations):
        """
        Generates binary masks from WFDB annotations.
        Label 1: QRS Complex (from onset '(' to offset ')')
        Label 0: Background
        """
        sample_conversion = self.target_fs / self.fs
        label_mask = np.zeros(ecg_length, dtype=int)

        i = 0
        while i < len(annotations.symbol):
            # LUDB Format: '(' marks onset, 'N'/'V' marks peak, ')' marks offset
            if annotations.symbol[i] in ['N', 'V']:
                # Check for surrounding boundaries
                if (i > 0 and annotations.symbol[i - 1] == '(') and \
                        (i < len(annotations.symbol) - 1 and annotations.symbol[i + 1] == ')'):
                    start = int(round(annotations.sample[i - 1] * sample_conversion))
                    end = int(round(annotations.sample[i + 1] * sample_conversion))

                    # Safety clipping
                    start = max(0, min(start, ecg_length - 1))
                    end = max(start, min(end, ecg_length))

                    label_mask[start:end] = 1
                    i += 2  # Skip processed symbols
            i += 1

        return label_mask

    def __getitem__(self, idx):
        record_id = self.record_ids[idx]
        record_path = os.path.join(self.db_dir, str(record_id))

        # Read ECG Record (using lead II, typically index 1 in LUDB)
        record = wfdb.rdrecord(record_path)
        ecg_signal = record.p_signal[:, 1]

        # Read Annotations (extension 'atr_ii' for lead II)
        annotations = wfdb.rdann(record_path, 'atr_ii')

        # Preprocessing
        ecg_filtered = self.preprocess_signal(ecg_signal)

        # Augmentation (Training only - usually handled by flags, here random for simplicity)
        if np.random.rand() > 0.5:
            ecg_filtered = self.add_noise(ecg_filtered)

        # Generate Label Mask
        label = self.create_labels(len(ecg_filtered), annotations)

        # To Tensor: Input [1, Length], Label [Length]
        segment = torch.FloatTensor(ecg_filtered).unsqueeze(0)
        label = torch.LongTensor(label)

        return segment, label


def collate_fn(batch):
    """Filter out None values if any."""
    batch = [b for b in batch if b is not None]
    return torch.utils.data.dataloader.default_collate(batch)


def train_epoch(model, loader, criterion, optimizer, device, epoch, num_epochs):
    model.train()
    running_loss = 0.0

    # Progress bar for the batch
    batch_bar = tqdm(loader, desc=f"Epoch {epoch}/{num_epochs} [Train]", leave=False)

    for inputs, labels in batch_bar:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)  # Output: [Batch, Classes, Length]

        # CrossEntropyLoss expects:
        # Input: [Batch, Classes, Length]
        # Target: [Batch, Length]
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        batch_bar.set_postfix({"Loss": f"{loss.item():.4f}"})

    return running_loss / len(loader.dataset)


def validate_epoch(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0

    batch_bar = tqdm(loader, desc="Validating", leave=False)

    with torch.no_grad():
        for inputs, labels in batch_bar:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            batch_bar.set_postfix({"Val Loss": f"{loss.item():.4f}"})

    return running_loss / len(loader.dataset)


def train_qrs_model(model, train_loader, val_loader, criterion, optimizer, scheduler,
                    num_epochs, patience, device, save_path):
    best_val_loss = float('inf')
    epochs_no_improve = 0

    print(f"Starting training on device: {device}")

    for epoch in range(1, num_epochs + 1):
        # 1. Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, epoch, num_epochs)

        # 2. Validate
        val_loss = validate_epoch(model, val_loader, criterion, device)

        # 3. Logging
        current_lr = optimizer.param_groups[0]['lr']
        print(
            f"Epoch {epoch}/{num_epochs} - Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | LR: {current_lr:.6f}")

        # 4. Checkpointing & Early Stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), save_path)
            print(f"[*] Best model saved to {save_path}")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"[!] Early stopping triggered after {epoch} epochs.")
                break

        # 5. LR Scheduling
        if scheduler:
            scheduler.step(val_loss)


def main():
    # --- Configuration ---
    # TODO: Update this path to your local LUDB directory
    DB_DIR = '/path/to/your/ludb/database/1.0.0'
    SAVE_PATH = './checkpoints/qrs_unet_best.pth'
    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)

    BATCH_SIZE = 32
    EPOCHS = 100
    PATIENCE = 25
    LR = 0.01

    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Dataset Setup ---
    print("Preparing Datasets...")
    try:
        full_dataset = LUDBDataset(db_dir=DB_DIR)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Please check if DB_DIR points to the correct WFDB directory.")
        return

    # Split Train/Val (80/20)
    dataset_size = len(full_dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(0.8 * dataset_size))

    np.random.shuffle(indices)
    train_indices, val_indices = indices[:split], indices[split:]

    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(full_dataset, val_indices)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, collate_fn=collate_fn)

    # --- Model Setup ---
    print("Initializing QRSUNet...")
    model = QRSUNet(in_channels=1, num_classes=2).to(device)

    # Optimizer & Criterion
    # CrossEntropyLoss automatically handles class imbalance if weight argument is provided (optional)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=LR, momentum=0.9)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=10, verbose=True)

    # --- Training Loop ---
    train_qrs_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=EPOCHS,
        patience=PATIENCE,
        device=device,
        save_path=SAVE_PATH
    )


if __name__ == "__main__":
    main()