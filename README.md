# UMPA-SAM

**Unified Multi-Prompt Adaptation for the Segment Anything Model**
---

## Overview

UMPA-SAM is a research framework that adapts the [Segment Anything Model (SAM)](https://github.com/facebookresearch/segment-anything) for robust polyp segmentation in colonoscopy images. The framework is designed to handle imprecise, inconsistent, or incomplete prompt inputs that commonly occur in clinical settings.

The model accepts multiple prompt modalities — **bounding box, point, mask, and text** — and produces stable, high-quality segmentation masks even under prompt variability.

---

## Key Features

- Multi-modal prompt support (box, point, mask, text)
- Robust to prompt noise and clinical annotation imperfections
- Built on top of SAM3 (Segment Anything Model v3)
- Three-phase training strategy for stable convergence
- Evaluated on standard polyp benchmarks (Kvasir-SEG, CVC-ClinicDB)

---

## Project Structure

```
umpt_sam/          # Core model and training code (details withheld)
├── config/        # Hyperparameter configurations
├── data/          # Dataset loading and augmentation
├── losses/        # Loss functions
├── modules/       # Model components
sam3/              # SAM3 backbone (third-party)
scripts/           # Evaluation scripts
tests/             # Unit tests
data/              # Dataset directory (not included)
```

---

## Requirements

- Python ≥ 3.9
- PyTorch ≥ 2.0
- CUDA-capable GPU (recommended: ≥ 16 GB VRAM)

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Getting Started

### 1. Prepare dataset

Download [Kvasir-SEG](https://datasets.simula.no/kvasir-seg/) or [CVC-ClinicDB](https://polyp.grand-challenge.org/CVCClinicDB/) and place them under `data/` following this structure:

```
data/
├── images/
├── masks/
└── split/
    ├── train.txt
    └── val.txt
```

### 2. Download SAM3 checkpoint

Place the SAM3 model checkpoint at the project root:

```
sam3.pt
```

### 3. Training

```bash
python -m umpt_sam.training.trainer \
    --data_root data/ \
    --sam_checkpoint sam3.pt
```

### 4. Evaluation

```bash
python -m umpt_sam.evaluate \
    --data_root data/ \
    --checkpoint checkpoints/best_model.pth
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
