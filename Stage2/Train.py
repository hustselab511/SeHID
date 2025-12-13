import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

# Import your model
from SemiHID.Stage1.CmSAModel import CmSAModel
# Import your custom dataset modules
# Ensure these match your actual file structure
from SemiHID.utils.dataset import load_data_or_generate_dummy, SETUP_SEED, BCGAugmentDataset


def train_cmsa_model(model, train_loader, val_loader, optimizer,
                     epochs, device, scheduler=None,
                     save_path="./checkpoints/best_cmsa_model.pth"):
    """
    Stage 1 Training: Train the Attention Model (CmSAModel).
    Uses a weighted MSE loss to focus on QRS regions.
    """
    history = {
        "train_loss": [],
        "val_loss": []
    }

    best_val_loss = float('inf')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Loss Weights Configuration
    QRS_WEIGHT = 3.0
    NON_QRS_WEIGHT = 1.0

    for epoch in range(1, epochs + 1):
        # --- Training Phase ---
        model.train()
        running_train_loss = 0.0

        train_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} [Train]", unit="batch")
        for batch_data in train_bar:
            # Unpack batch data (adjust strictly to your dataset __getitem__ return)
            # Assuming: signals, labels, aux_signals
            ecg_signals, ecg_qrs_labels, bcg_signals = batch_data

            bcg = bcg_signals.to(device)
            # ecg_target = ecg_signals.to(device) # Not used in pure attention training loss logic provided previously
            qrs_labels = ecg_qrs_labels.to(device)

            # Forward pass: CmSAModel returns (features, prediction)
            _, prediction = model(bcg)

            # Weighted MSE Loss Calculation
            # Focus on reconstructing the QRS location/mask
            weight_matrix = qrs_labels * (QRS_WEIGHT - NON_QRS_WEIGHT) + NON_QRS_WEIGHT

            # Error calculation
            squared_error = (prediction - qrs_labels) ** 2
            weighted_mse = (squared_error * weight_matrix).mean()

            loss = weighted_mse

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item()
            train_bar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_train_loss = running_train_loss / len(train_loader)
        history["train_loss"].append(avg_train_loss)

        # --- Validation Phase ---
        model.eval()
        running_val_loss = 0.0

        with torch.no_grad():
            val_bar = tqdm(val_loader, desc=f"Epoch {epoch}/{epochs} [Val]", unit="batch")
            for batch_data in val_bar:
                ecg_signals, ecg_qrs_labels, bcg_signals = batch_data

                bcg = bcg_signals.to(device)
                qrs_labels = ecg_qrs_labels.to(device)

                _, prediction = model(bcg)

                # Validation Loss
                weight_matrix = qrs_labels * (QRS_WEIGHT - NON_QRS_WEIGHT) + NON_QRS_WEIGHT
                squared_error = (prediction - qrs_labels) ** 2
                weighted_mse = (squared_error * weight_matrix).mean()

                running_val_loss += weighted_mse.item()
                val_bar.set_postfix({"val_loss": f"{weighted_mse.item():.4f}"})

            # Scheduler Step
            if scheduler:
                scheduler.step(running_val_loss)

        avg_val_loss = running_val_loss / len(val_loader)
        history["val_loss"].append(avg_val_loss)

        print(f"\n[Epoch {epoch}] Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        # Save Best Model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), save_path)
            print(f"[*] Best model saved to {save_path}")

    return history


def main():
    # --- Configuration ---
    # TODO: Modify these paths for your local environment
    DATA_ROOTS = ["/path/to/your/dataset/directory"]
    SAVE_DIR = "./experiments/stage1_results"
    MODEL_SAVE_PATH = os.path.join(SAVE_DIR, "cmsa_model_best.pth")
    SPLIT_INFO_PATH = os.path.join(SAVE_DIR, "data_split_info.json")

    # Hyperparameters
    BATCH_SIZE = 64
    EPOCHS = 40
    LR = 3e-4
    WEIGHT_DECAY = 1e-4
    SEED = 42

    # Setup
    SETUP_SEED(seed=SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(SAVE_DIR, exist_ok=True)

    # Data Loading
    print(f"Loading datasets from: {DATA_ROOTS}")

    # Using your custom data splitting function
    DATA_ROOT = "dataset/Kansas"  # Users will change this to their real path
    train_signals, val_signals, train_labels, val_labels = load_data_or_generate_dummy(DATA_ROOT)

    # Augmentation Configuration
    augment_config = {
        'use_time_shift': True, 'shift_range': (-2, 2), 'shift_prob': 0.5,
        'use_noise': True, 'noise_std': 0.05, 'noise_prob': 0.7,
        'use_scale': True, 'scale_range': (0.8, 1.2), 'scale_prob': 0.7,
        'use_mask': True, 'mask_prob': 0.2, 'mask_ratio': 0.1,
        'use_freq_aug': True, 'freq_prob': 0.3, 'freq_ratio': 0.1
    }

    # Initialize Datasets
    train_dataset = BCGAugmentDataset(train_signals, train_labels, is_train=True, augment_config=augment_config)
    val_dataset = BCGAugmentDataset(val_signals, val_labels, is_train=False, augment_config=augment_config)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    # Initialize Model
    print("\nInitializing CmSAModel...")
    model = CmSAModel(max_len=625, ffn_dim=128).to(device)

    # Optimizer & Scheduler
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True, min_lr=1e-6
    )

    # Run Training
    train_cmsa_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        epochs=EPOCHS,
        device=device,
        scheduler=scheduler,
        save_path=MODEL_SAVE_PATH
    )


if __name__ == "__main__":
    main()