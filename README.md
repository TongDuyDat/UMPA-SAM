# UMPA-SAM

**UMPA-SAM: Unified Multi-Prompt Adaptation for Robust Polyp Segmentation**

---

## Overview

UMPA-SAM is a research framework that adapts the [Segment Anything Model (SAM)](https://github.com/facebookresearch/segment-anything) for **robust polyp segmentation** in colonoscopy images. The framework is designed to handle imprecise, inconsistent, or incomplete prompt inputs that commonly occur in clinical settings.

The model accepts multiple prompt modalities — **bounding box, point, mask, and text** — and produces stable, high-quality segmentation masks even under prompt variability.

> **Note:** Model architecture details are currently withheld and will be released upon publication.

## Model Architecture

![UMPA-SAM Architecture](assets/model_architecture.png)

---

## Key Features

- **Multi-modal prompt support**: bounding box, point, mask, and text prompts  
- **Robust to prompt noise**: handles clinical annotation imperfections  
- **Built on SAM3**: leverages Segment Anything Model v3 as backbone  
- **Three-phase training strategy**: phased optimization schedule for stable convergence  
- **Ablation study framework**: 10 pre-defined scenarios for systematic component analysis  
- **Multi-dataset support**: evaluated on 5 standard polyp benchmarks  

---

## Project Structure

```
sam3/                          # Project root
├── main.py                    # Quick-start training script (single dataset)
├── train_ablation.py          # Ablation study training CLI
├── evaluate_ablation.py       # Ablation study evaluation CLI
├── run_all_datasets.py        # One-click runner: all datasets × all scenarios
├── requirements.txt           # Python dependencies
├── py310_export.yml           # Conda environment export (Python 3.10)
│
├── umpt_sam/                  # Core framework
│   ├── config/                # Configuration dataclasses
│   │   ├── model_config.py    # Model hyperparameters
│   │   ├── train_config.py    # 3-phase training schedule
│   │   └── experiment_config.py  # Ablation scenario definitions
│   ├── data/                  # Dataset loading & transforms
│   │   ├── dataset_registry.py   # 5-dataset registry
│   │   ├── polyp_dataset.py      # PolypDataset class
│   │   ├── polyp_transforms.py   # Albumentations pipelines
│   │   └── make_split.py         # Split generation utility
│   ├── losses/                # Loss functions
│   ├── modules/               # Model components (withheld)
│   ├── training/              # Trainer, evaluator, schedulers
│   └── umpa_model.py          # Model entry point (withheld)
│
├── sam3/                      # SAM3 backbone (third-party)
├── scripts/                   # Helper scripts
│   ├── download_datasets.sh   # Dataset download from Google Drive
│   └── eval/                  # Evaluation utilities
├── data/                      # Additional data processing
├── model_trained/             # Pre-trained checkpoints directory
├── tests/                     # Unit tests
└── docs/                      # Documentation & diagrams
```

---

## Requirements

| Component | Minimum Version |
|-----------|----------------|
| Python    | ≥ 3.10         |
| PyTorch   | ≥ 2.0          |
| CUDA      | ≥ 12.4 (recommended) |
| GPU VRAM  | ≥ 16 GB (recommended) |

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/TongDuyDat/UMPA-SAM.git
cd UMPA-SAM
```

### 2. Create conda environment

**Option A — From exported environment (recommended):**

```bash
conda env create -f py310_export.yml
conda activate <env_name>
```

**Option B — From requirements.txt:**

```bash
conda create -n umpa_sam python=3.10 -y
conda activate umpa_sam
pip install -r requirements.txt
```

### 3. Download SAM3 checkpoint

Place the SAM3 pre-trained checkpoint at:

```
model_trained/sam3.pt
```

---

## Dataset Preparation

### Supported Datasets

| Name              | Registry Key     | Description                      |
|-------------------|------------------|----------------------------------|
| Kvasir-SEG        | `kvasir_seg`     | 1000 polyp images                |
| CVC-ClinicDB      | `cvc_clinicdb`   | 612 polyp images                 |
| CVC-ColonDB       | `cvc_colondb`    | 380 polyp images                 |
| CVC-300           | `cvc_300`        | 300 polyp images                 |
| ETIS-LaribPolypDB | `etis_larib`     | 196 polyp images                 |

### Download datasets

**Automatic download** (from Google Drive):

```bash
bash scripts/download_datasets.sh
```

**Manual download**: Place each dataset under `umpt_sam/data/` with this structure:

```
umpt_sam/data/
├── Kvasir_SEG/
│   ├── images/          # .jpg / .png images
│   └── masks/           # Corresponding binary masks
├── CVC-ClinicDB/
│   ├── images/
│   └── masks/
├── CVC-ColonDB/
│   ├── images/
│   └── masks/
├── CVC_300/
│   ├── images/
│   └── masks/
└── ETIS-Larib/
    ├── images/
    └── masks/
```

### Train/Val/Test splits

Splits are **auto-generated** the first time you run training or evaluation. The system creates `split/{train,val,test}.txt` inside each dataset folder with an 80/10/10 ratio (seed=42).

To manually generate splits for all datasets:

```python
from umpt_sam.data.dataset_registry import ensure_all_splits
ensure_all_splits()
```

---

## Training

### Quick Start — Single Dataset

```bash
python main.py
```

This trains on Kvasir-SEG with default hyperparameters using the 3-phase training schedule:

| Phase      | Epochs | Learning Rate | λ_con | Frozen Components                      |
|------------|--------|---------------|-------|----------------------------------------|
| Warmup     | 5      | 1e-4          | 0.0   | Image Encoder, Prompt Encoder, Mask Decoder |
| Adaptation | 5      | 5e-5          | 0.0   | Image Encoder, Mask Decoder            |
| Consistency| 10     | 1e-5          | 0.5   | Image Encoder                          |

### Ablation Study Training

Train a **single ablation scenario**:

```bash
python train_ablation.py --scenario full_model --dataset kvasir_seg
```

Train **all 10 scenarios** for a dataset:

```bash
python train_ablation.py --all --dataset kvasir_seg
```

**Dry-run** (1 epoch, 4 samples — verify setup):

```bash
python train_ablation.py --scenario full_model --dry-run
```

List all available scenarios:

```bash
python train_ablation.py --list
```

#### Available Ablation Scenarios

| Group | Scenario | Active Prompts | Active Components |
|-------|----------|----------------|-------------------|
| **A — Prompt** | `only_box` | box | MPPG, UPFE, MPCL |
| | `only_point` | point | MPPG, UPFE, MPCL |
| | `only_mask` | mask | MPPG, UPFE, MPCL |
| | `only_text` | text | MPPG, UPFE, MPCL |
| | `box_point` | box, point | MPPG, UPFE, MPCL |
| | `box_point_mask` | box, point, mask | MPPG, UPFE, MPCL |
| **B — Component** | `wo_mppg` | all | UPFE, MPCL |
| | `wo_upfe` | all | MPPG, MPCL |
| | `wo_mpcl` | all | MPPG, UPFE |
| **C — Baseline** | `full_model` | all | MPPG, UPFE, MPCL |

### One-Click: All Datasets × All Scenarios

Train **all 50 experiments** (5 datasets × 10 scenarios):

```bash
python run_all_datasets.py
```

Selective runs:

```bash
# Single dataset, all scenarios
python run_all_datasets.py --dataset kvasir_seg

# Single scenario, all datasets
python run_all_datasets.py --scenario full_model

# Custom device and checkpoint
python run_all_datasets.py --device cuda:1 --sam-checkpoint /path/to/sam3.pt

# Dry-run to verify setup
python run_all_datasets.py --dry-run
```

### Training CLI Options

| Argument            | Default                   | Description                       |
|---------------------|---------------------------|-----------------------------------|
| `--scenario`        | —                         | Ablation scenario name            |
| `--all`             | —                         | Train all scenarios               |
| `--dataset`         | `kvasir_seg`              | Dataset name                      |
| `--save-dir`        | `checkpoints/ablation`    | Output directory                  |
| `--device`          | `cuda`                    | Compute device                    |
| `--sam-checkpoint`  | `model_trained/sam3.pt`   | SAM3 weights path                 |
| `--dry-run`         | `false`                   | Quick test (1 epoch, 4 samples)   |

### Training Outputs

```
checkpoints/ablation/<dataset>/<scenario>/run_YYYYMMDD_HHMMSS/
├── best_model.pth          # Best validation checkpoint
├── latest_model.pth        # Latest checkpoint
├── training_log.txt        # Detailed training log
└── experiment_config.json  # Scenario configuration snapshot
```

---

## Evaluation

### Evaluate a single scenario

```bash
python evaluate_ablation.py \
    --scenario full_model \
    --checkpoint checkpoints/ablation/kvasir_seg/full_model/run_.../best_model.pth \
    --dataset kvasir_seg
```

### Evaluate all trained scenarios

```bash
python evaluate_ablation.py --all --base-dir checkpoints/ablation --dataset kvasir_seg
```

### Generate summary table only

```bash
python evaluate_ablation.py --summary --base-dir checkpoints/ablation --dataset kvasir_seg
```

### Evaluation Metrics

| Metric       | Description                                 |
|--------------|---------------------------------------------|
| **Dice**     | Dice similarity coefficient                 |
| **mIoU**     | Mean Intersection over Union                |
| **Precision**| Pixel-level precision (full_metrics mode)    |
| **Recall**   | Pixel-level recall (full_metrics mode)       |
| **F2-Score** | F2 measure emphasizing recall               |
| **Mask AP**  | AP at IoU thresholds 0.50:0.05:0.95         |

### Evaluation CLI Options

| Argument            | Default                   | Description                       |
|---------------------|---------------------------|-----------------------------------|
| `--scenario`        | —                         | Single scenario to evaluate       |
| `--all`             | —                         | Evaluate all scenarios            |
| `--summary`         | —                         | Generate summary table only       |
| `--checkpoint`      | —                         | Checkpoint path (with --scenario) |
| `--base-dir`        | `checkpoints/ablation`    | Base results directory            |
| `--dataset`         | `kvasir_seg`              | Dataset name                      |
| `--device`          | `cuda`                    | Compute device                    |
| `--batch-size`      | `4`                       | Evaluation batch size             |
| `--sam-checkpoint`  | `model_trained/sam3.pt`   | SAM3 weights path                 |

---

## Experiment Report

When using `run_all_datasets.py`, a JSON report is automatically generated:

```
checkpoints/ablation/experiment_report.json
```

The report includes: run status (OK/FAILED/SKIPPED), elapsed time, error messages, and tracebacks for failed experiments. The runner is **fault-tolerant** — if a scenario crashes, the error is logged and the next experiment continues.

---

## Testing

Run unit tests with:

```bash
pytest tests/ -v
```

---

## Results

*To be released upon publication.*

---

## Citation

*To be released upon publication.*

---

## License

This project is licensed under the terms of the [LICENSE](LICENSE) file.
