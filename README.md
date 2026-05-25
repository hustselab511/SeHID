# SeHID: Semantics-enhanced Heartbeat Identification for Robust BCG Monitoring in Arrhythmia Patients

## 📖 Abstract

Ballistocardiography (BCG) signals enable continuous cardiovascular health monitoring but exhibit varying morphologies, especially in patients with arrhythmia. Existing methods often rely on morphological consistency, which fails under arrhythmic conditions.

We propose **SeHID** (Semantics-indicated Heartbeat Identification), a two-stage deep learning framework that reframes heartbeat detection as a semantic feature conversion task. 
1.  **Stage I (CmSA):** A Transformer encoder captures cross-modal semantic alignment to anchor potential heartbeat locations.
2.  **Stage II (FGWR):** A U-Net backbone performs multi-scale fusion of waveforms and semantic features to reconstruct fine-grained QRS-like indicators.

## 🛠️ Environment Setup

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/hustselab511/SeHID.git](https://github.com/hustselab511/SeHID.git)
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    
## 💾 Data Preparation

This study utilizes four public datasets. Please download them from the official sources below:

| Dataset | Description | Link |
| :--- | :--- | :--- |
| **LUDB** | Pre-processing (QRS Label Generation) | [PhysioNet / Lobachevsky Univ DB v1.0.1](https://physionet.org/content/ludb/1.0.1/) |
| **Kansas** | Training | [IEEE DataPort](https://ieee-dataport.org/open-access/bed-based-ballistocardiography-dataset) |
| **SRRSH** | Testing | [Figshare (DOI: 10.6084/m9.figshare.28643153)](https://doi.org/10.6084/m9.figshare.28643153) |
| **WHU** | Testing | [Figshare (DOI: 10.6084/m9.figshare.28416896)](https://doi.org/10.6084/m9.figshare.28416896) |

*Note: Please update the `DB_DIR` or `DATA_ROOTS` paths in the python scripts to point to your local data directory.*

### ECG-BCG Temporal Alignment

ECG and BCG training samples are temporally aligned through a two-step process:

1. **Coarse Synchronization**: Utilize timestamps to eliminate significant time offsets between the two signals.
2. **J-peak Refinement**: 
   - Identify the first R-wave peak in each ECG segment
   - Search for the local amplitude maximum of the BCG J-wave within a physiological delay window of **150-350 ms** after each R-peak
   - Apply a global offset to the entire signal segment based on the detected R-J time difference

**Quality Control**: All auto-aligned segments undergo manual review with strict criteria:
- Segments with peak amplitude below **50%** of the window average are excluded
- Segments with ambiguous peaks (multiple peaks with relative amplitude difference < **10%**) are excluded
- Approximately **4.2%** of samples from the Kansas training set were excluded

## 🚀 Usage

The training pipeline consists of three steps. 

### 1. Pre-processing: QRS Label Generation
Train a U-Net on the LUDB dataset to generate ground-truth QRS masks for BCG data supervision.
```bash
python -m Pre_processing.Train --db_dir /path/to/LUDB --save_path checkpoints/qrs_unet.pth
```

### 2. Stage I: Cross-modal Semantic Anchoring
Train the CsWAModel to learn coarse-grained semantic features from BCG signals.
```bash
python -m Stage1.Train --data_root /path/to/Kansas 
```

### 3. Stage II: Fine-Grained Waveform Reconstruction
Train the FGWRModel (ResUNet) using features extracted from the frozen Stage I model.
```bash
python -m Stage2.Train \
  --data_root /path/to/Kansas \
  --stage1_checkpoint experiments/stage1_results/cswa_model_best.pth \
```

## 📊 Hyperparameters

### Signal Preprocessing & Data Segmentation

| Parameter | Description | Value | Notes |
| :--- | :--- | :--- | :--- |
| Resampling Rate | Signal resampling rate | **125 Hz** | Uniform time resolution for ECG and BCG |
| Band-pass Filter (ECG) | ECG filter cutoff frequency | **0.5–40 Hz** | Remove baseline drift and EMG noise |
| Band-pass Filter (BCG) | BCG filter cutoff frequency | **1–12 Hz** | Preserve I, J, K complex components |
| Window Size | Signal segmentation window | **5 seconds** | Optimal balance after 2.5–20s sensitivity analysis |
| Sliding Window Step Size | Training data augmentation step | **1 second** | Sliding window augmentation |
| Physiological Alignment Window | Electro-mechanical coupling window | **150–350 ms** | Reasonable time span for BCG J-wave after ECG R-wave |
| Manual Review Exclude Ratio | Samples excluded after manual review | **~4.2%** | Kansas training set, J-wave unclear samples |
| Time Tolerance Window | Beat detection tolerance | **±75 ms** | AAMI standard compliant |

### Model Architecture (Stage 1 & Stage 2)

| Module / Stage | Parameter | Description | Value |
| :--- | :--- | :--- | :--- |
| Stage 1 (CmSA) | Embedding Dimension ($C$) | Conv embedding feature dimension | **32** |
| Stage 1 (CmSA) | Convolution Kernel Sizes | Multi-scale convolution kernels | **1, 3, 5** |
| Stage 1 (CmSA) | Stacked Blocks ($L$) | Transformer encoder blocks | **2** |
| Stage 1 (CmSA) | Attention Heads ($H$) | Multi-head self-attention heads | **4** |
| Stage 2 (FGWR) | Kernel Size in Conv Block | Conv block kernel size | **3** |
| Stage 2 (FGWR) | Cross-Attention Heads | Multi-head cross-attention heads | **4** |
| Stage 2 (FGWR) | Bi-LSTM Input Dimension | Bi-LSTM input feature dimension | **128** |
| Stage 2 (FGWR) | Bi-LSTM Hidden Units | Bi-LSTM hidden layer units | **128** |
| Stage 2 (FGWR) | Bi-LSTM Output Dimension | Bi-LSTM final output dimension | **256** |

### Loss Function Weights

| Loss Term | Parameter | Description | Value | Notes |
| :--- | :--- | :--- | :--- | :--- |
| $\mathcal{L}_1$ / $\mathcal{L}_2$ | QRS Region Weight ($w^{(t)}$) | QRS region loss weight | **3** | Alleviate sample imbalance due to sparse waveforms |
| $\mathcal{L}_1$ / $\mathcal{L}_2$ | Non-QRS Region Weight | Non-QRS region loss weight | **1** | Standard weight for background regions |

### Training & Optimizer Configuration

The framework employs a staged progressive training strategy with **Adam optimizer** across all three steps:

| Training Step / Stage | Epochs | Learning Rate | Batch Size | Loss Function |
| :--- | :--- | :--- | :--- | :--- |
| Step 0: ECG QRS Segmentation Pre-training (LUDB) | **100** | $1\times 10^{-3}$ | **128** | Binary Cross-Entropy Loss |
| Step 1: CmSA Module Training (Kansas) | **50** | $1\times 10^{-4}$ | **64** | Weighted MSE Loss ($\mathcal{L}_1$) |
| Step 2: FGWR End-to-End Training (Kansas, CmSA frozen) | **50** | $1\times 10^{-4}$ | **64** | Weighted MSE Loss ($\mathcal{L}_2$) |

