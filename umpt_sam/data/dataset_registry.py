"""Unified dataset registry for 5 polyp benchmark datasets.

Provides:
- ``POLYP_DATASETS``: registry dict with root paths and text labels
- ``get_dataset_config(name)``: returns a ``DatasetConfig`` ready for ``PolypDataset``
- ``ensure_splits(root)``: auto-generates train/val/test split files if missing
- ``list_datasets()``: returns available dataset names

All paths are resolved relative to the project root (``sam3/``).
"""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Dict, List

from .polyp_dataset import DatasetConfig

# ---------------------------------------------------------------------------
# Project root = directory containing this file's grandparent (umpt_sam/)
# i.e. sam3/
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parents[2]

# ---------------------------------------------------------------------------
# Dataset registry
# ---------------------------------------------------------------------------

POLYP_DATASETS: Dict[str, dict] = {
    "kvasir_seg": {
        "root": "umpt_sam/data/Kvasir_SEG",
        "text_label": "polyp in colonoscopy image",
    },
    "cvc_clinicdb": {
        "root": "umpt_sam/data/CVC-ClinicDB",
        "text_label": "polyp in colonoscopy image",
    },
    "cvc_colondb": {
        "root": "umpt_sam/data/CVC-ColonDB",
        "text_label": "polyp in colonoscopy image",
    },
    "cvc_300": {
        "root": "umpt_sam/data/CVC_300",
        "text_label": "polyp in colonoscopy image",
    },
    "etis_larib": {
        "root": "umpt_sam/data/ETIS-Larib",
        "text_label": "polyp in colonoscopy image",
    },
}

_IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif"}


def list_datasets() -> list[str]:
    """Return sorted list of available dataset names."""
    return sorted(POLYP_DATASETS.keys())


def _resolve_root(relative_root: str) -> str:
    """Resolve a relative dataset path against project root."""
    full = _PROJECT_ROOT / relative_root
    return str(full)


# ---------------------------------------------------------------------------
# Split generation
# ---------------------------------------------------------------------------

def ensure_splits(
    root: str,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> None:
    """Generate train/val/test split files if they don't already exist.

    Creates ``root/split/{train,val,test}.txt`` containing file stems
    (no extensions). Idempotent: skips if split dir already exists with
    all 3 files.

    Parameters
    ----------
    root : str
        Absolute path to dataset root (must contain ``images/``).
    train_ratio : float
        Fraction for training set.
    val_ratio : float
        Fraction for validation set. Test = 1 - train - val.
    seed : int
        Random seed for reproducibility.
    """
    split_dir = os.path.join(root, "split")
    expected_files = ["train.txt", "val.txt", "test.txt"]

    # Skip if all splits already exist
    if os.path.isdir(split_dir) and all(
        os.path.isfile(os.path.join(split_dir, f)) for f in expected_files
    ):
        return

    image_dir = os.path.join(root, "images")
    if not os.path.isdir(image_dir):
        raise FileNotFoundError(f"Image directory not found: {image_dir}")

    # Collect file stems
    all_stems: List[str] = []
    for fname in sorted(os.listdir(image_dir)):
        stem, ext = os.path.splitext(fname)
        if ext.lower() in _IMG_EXTS:
            all_stems.append(stem)

    if not all_stems:
        raise ValueError(f"No images found in {image_dir}")

    # Shuffle and split
    rng = random.Random(seed)
    rng.shuffle(all_stems)

    total = len(all_stems)
    train_end = int(total * train_ratio)
    val_end = int(total * (train_ratio + val_ratio))

    splits = {
        "train.txt": all_stems[:train_end],
        "val.txt": all_stems[train_end:val_end],
        "test.txt": all_stems[val_end:],
    }

    os.makedirs(split_dir, exist_ok=True)
    for filename, stems in splits.items():
        filepath = os.path.join(split_dir, filename)
        with open(filepath, "w") as f:
            f.write("\n".join(stems))

    print(
        f"📂 Splits created for {os.path.basename(root)}: "
        f"train={len(splits['train.txt'])}, "
        f"val={len(splits['val.txt'])}, "
        f"test={len(splits['test.txt'])} "
        f"(total={total})"
    )


def ensure_all_splits() -> None:
    """Ensure split files exist for all registered datasets."""
    for name in list_datasets():
        info = POLYP_DATASETS[name]
        root = _resolve_root(info["root"])
        if not os.path.isdir(root):
            print(f"⚠️  {name}: dataset directory not found at {root}")
            continue
        ensure_splits(root)
        print(f"✅ {name}: splits OK")


# ---------------------------------------------------------------------------
# DatasetConfig factory
# ---------------------------------------------------------------------------

def get_dataset_config(name: str) -> DatasetConfig:
    """Build a ``DatasetConfig`` for the named dataset.

    Automatically resolves paths and ensures splits exist.

    Parameters
    ----------
    name : str
        One of :func:`list_datasets`.

    Returns
    -------
    DatasetConfig
        Ready to pass to ``PolypDataset(cfg=..., phase=...)``.

    Raises
    ------
    KeyError
        If dataset name is unknown.
    """
    if name not in POLYP_DATASETS:
        available = ", ".join(list_datasets())
        raise KeyError(f"Unknown dataset '{name}'. Available: {available}")

    info = POLYP_DATASETS[name]
    root = _resolve_root(info["root"])

    if not os.path.isdir(root):
        raise FileNotFoundError(
            f"Dataset '{name}' not found at: {root}"
        )

    # Auto-generate splits if needed
    ensure_splits(root)

    return DatasetConfig(
        root=root,
        text_label=info["text_label"],
    )
