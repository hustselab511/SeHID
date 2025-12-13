# SemiHID: Enhancing BCG Heartbeat Detection for Arrhythmic Patients

## 📖 Abstract

Ballistocardiography (BCG) signals enable continuous cardiovascular health monitoring but exhibit varying morphologies, especially in patients with arrhythmia. Existing methods often rely on morphological consistency, which fails under arrhythmic conditions.

We propose **SemiHID** (Semantics-indicated Heartbeat Identification), a two-stage deep learning framework that reframes heartbeat detection as a semantic feature conversion task. 
1.  **Stage I (CmSA):** A Transformer encoder captures cross-modal semantic alignment to anchor potential heartbeat locations.
2.  **Stage II (FGWR):** A U-Net backbone performs multi-scale fusion of waveforms and semantic features to reconstruct fine-grained QRS-like indicators.

## 🛠️ Environment Setup

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/hustselab511/SemiHID.git](https://github.com/hustselab511/SemiHID.git)
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

