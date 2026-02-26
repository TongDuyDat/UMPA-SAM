"""
umpt_sam/data/polyp_dataset.py

PolypDataset — Dataset chính cho training UMPA-SAM (Ngày 2).

Format thư mục dataset (theo PLAN.md Task 2.1):
    dataset_root/
    ├── images/          ← ảnh .jpg hoặc .png
    ├── masks/           ← binary mask .png, cùng tên với ảnh
    └── split/
        ├── train.txt    ← danh sách tên file (không có extension)
        └── val.txt

Output mỗi __getitem__:
    {
        "image":        Tensor (3, H, W),   float32, normalized [-1, 1]
        "mask":         Tensor (1, H, W),   float32, binary [0, 1]
        "bbox":         Tensor (4,),        float32, pixel xyxy
        "points":       Tensor (N, 2),      float32, pixel xy
        "point_labels": Tensor (N,),        int64,   1=positive, 0=negative
        "coarse_mask":  Tensor (1, H, W),   float32, = mask (perturbation áp sau)
        "text":         str,                "polyp in colonoscopy image"
    }

Normalization: (pixel / 127.5) - 1.0 → [-1, 1], khớp Sam3Processor.
"""

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .transforms import TransformConfig, build_transforms

_DEFAULT_TEXT = "polyp in colonoscopy image"

# ---------------------------------------------------------------------------
# Dataset config dataclass (thay thế file .py exec() cũ)
# ---------------------------------------------------------------------------

@dataclass
class DatasetConfig:
    """Cấu hình một dataset polyp.

    Thêm dataset mới → subclass hoặc khởi tạo DatasetConfig trực tiếp.
    Không cần sửa PolypDataset.
    """
    root: str                                       # đường dẫn tuyệt đối đến dataset
    text_label: str = _DEFAULT_TEXT
    n_pos_points: int = 3
    n_neg_points: int = 1
    transform_cfg: TransformConfig = field(default_factory=TransformConfig)

    # Các dataset được định nghĩa sẵn
    @classmethod
    def kvasir_seg(cls, root: str, **kwargs) -> "DatasetConfig":
        return cls(root=root, **kwargs)

    @classmethod
    def cvc_clinicdb(cls, root: str, **kwargs) -> "DatasetConfig":
        return cls(root=root, **kwargs)

    @classmethod
    def cvc_colondb(cls, root: str, **kwargs) -> "DatasetConfig":
        return cls(root=root, **kwargs)

    @classmethod
    def cvc_300(cls, root: str, **kwargs) -> "DatasetConfig":
        return cls(root=root, **kwargs)

    @classmethod
    def etis_larib(cls, root: str, **kwargs) -> "DatasetConfig":
        return cls(root=root, **kwargs)

    @classmethod
    def kvasir_sessile(cls, root: str, **kwargs) -> "DatasetConfig":
        return cls(root=root, text_label="sessile polyp in colonoscopy image", **kwargs)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def _load_split(split_file: str) -> List[str]:
    """Đọc file split/*.txt, trả về list tên file (không extension)."""
    with open(split_file, "r") as f:
        return [line.strip() for line in f if line.strip()]


def _discover_files(image_dir: str, mask_dir: str) -> List[Tuple[str, str]]:
    """Auto-discover cặp (image_path, mask_path) khi không có split file."""
    pairs = []
    for fname in sorted(os.listdir(image_dir)):
        name, ext = os.path.splitext(fname)
        if ext.lower() not in _IMG_EXTS:
            continue
        img_path = os.path.join(image_dir, fname)
        # mask có thể là .png bất kể ảnh là .jpg
        for mask_ext in [ext, ".png", ".jpg"]:
            mask_path = os.path.join(mask_dir, name + mask_ext)
            if os.path.exists(mask_path):
                pairs.append((img_path, mask_path))
                break
    return pairs


def _extract_bbox(mask_bin: np.ndarray) -> np.ndarray:
    """Binary mask (H,W) uint8 → xyxy pixel float32. Fallback: full image."""
    H, W = mask_bin.shape
    ys, xs = np.where(mask_bin > 0)
    if len(xs) == 0:
        return np.array([0.0, 0.0, float(W - 1), float(H - 1)], dtype=np.float32)
    return np.array(
        [float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())],
        dtype=np.float32,
    )


def _sample_points(
    mask_bin: np.ndarray, n_pos: int, n_neg: int, rng: np.random.Generator
) -> Tuple[np.ndarray, np.ndarray]:
    """Sample positive + negative points từ mask. Trả về (points, labels)."""
    H, W = mask_bin.shape
    pos_ys, pos_xs = np.where(mask_bin > 0)
    neg_ys, neg_xs = np.where(mask_bin == 0)

    def _pick(xs, ys, n):
        if len(xs) == 0 or n == 0:
            return np.empty((0, 2), dtype=np.float32)
        idx = rng.choice(len(xs), size=min(n, len(xs)), replace=False)
        return np.stack([xs[idx], ys[idx]], axis=-1).astype(np.float32)

    pos_pts = _pick(pos_xs, pos_ys, n_pos)
    neg_pts = _pick(neg_xs, neg_ys, n_neg)

    if len(pos_pts) == 0:
        # Fallback: centre of image
        pos_pts = np.array([[W / 2.0, H / 2.0]], dtype=np.float32)

    pts = np.concatenate([pos_pts, neg_pts], axis=0)
    lbls = np.array([1] * len(pos_pts) + [0] * len(neg_pts), dtype=np.int64)
    return pts, lbls


# ---------------------------------------------------------------------------
# PolypDataset
# ---------------------------------------------------------------------------

class PolypDataset(Dataset):
    """Dataset polyp segmentation cho UMPA-SAM.

    Hỗ trợ 2 chế độ tổ chức thư mục:
    1. Có split file:
           root/images/, root/masks/, root/split/train.txt
    2. Không có split file (tự khám phá):
           root/{phase}/images/, root/{phase}/masks/

    Args:
        cfg   : DatasetConfig — toàn bộ tham số dataset
        phase : "train" hoặc "val"
    """

    def __init__(self, cfg: DatasetConfig, phase: str = "train"):
        self.cfg = cfg
        self.phase = phase
        self.transform = build_transforms(cfg.transform_cfg, phase)
        self._rng = np.random.default_rng()

        self.pairs: List[Tuple[str, str]] = []
        self._load_pairs()

    # ------------------------------------------------------------------
    def _load_pairs(self):
        root = self.cfg.root

        # --- Chế độ 1: có split file ---
        split_file = os.path.join(root, "split", f"{self.phase}.txt")
        image_dir_flat = os.path.join(root, "images")
        mask_dir_flat = os.path.join(root, "masks")

        if os.path.exists(split_file) and os.path.exists(image_dir_flat):
            names = _load_split(split_file)
            for name in names:
                img_path = mask_path = None
                for ext in _IMG_EXTS:
                    p = os.path.join(image_dir_flat, name + ext)
                    if os.path.exists(p):
                        img_path = p
                        break
                for ext in [".png", ".jpg", ".jpeg"]:
                    p = os.path.join(mask_dir_flat, name + ext)
                    if os.path.exists(p):
                        mask_path = p
                        break
                if img_path and mask_path:
                    self.pairs.append((img_path, mask_path))
            return

        # --- Chế độ 2: {root}/{phase}/images + masks ---
        image_dir = os.path.join(root, self.phase, "images")
        mask_dir = os.path.join(root, self.phase, "masks")

        if os.path.exists(image_dir) and os.path.exists(mask_dir):
            self.pairs = _discover_files(image_dir, mask_dir)
            return

        # --- Chế độ 3: flat images/ masks/ (không có split, không có phase subdir) ---
        if os.path.exists(image_dir_flat) and os.path.exists(mask_dir_flat):
            self.pairs = _discover_files(image_dir_flat, mask_dir_flat)
            return

        raise ValueError(
            f"Không tìm thấy data tại '{root}'. "
            "Cần có split/train.txt + images/ + masks/, "
            "hoặc {phase}/images/ + {phase}/masks/."
        )

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Dict:
        img_path, mask_path = self.pairs[idx]

        # --- Đọc ảnh + mask ---
        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"Không đọc được ảnh: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Không đọc được mask: {mask_path}")

        # --- Augmentation (Resize + geometric + color) ---
        transformed = self.transform(image=image, mask=mask)
        image = transformed["image"]   # (H, W, 3) uint8
        mask = transformed["mask"]     # (H, W)    uint8

        # --- Normalize ảnh về [-1, 1] như Sam3Processor ---
        image_f = (image.astype(np.float32) / 127.5) - 1.0  # (H, W, 3)

        # --- Binary float mask ---
        mask_f = (mask.astype(np.float32) / 255.0)           # (H, W) [0,1]
        mask_bin = (mask_f > 0.5).astype(np.uint8)

        # --- Trích xuất raw prompts từ GT mask ---
        bbox = _extract_bbox(mask_bin)                        # (4,) xyxy pixel
        pts, lbls = _sample_points(
            mask_bin, self.cfg.n_pos_points, self.cfg.n_neg_points, self._rng
        )                                                      # (N,2), (N,)

        # --- Chuyển sang tensor ---
        image_t = torch.from_numpy(image_f.transpose(2, 0, 1))      # (3,H,W)
        mask_t = torch.from_numpy(mask_f).unsqueeze(0)              # (1,H,W)
        coarse_mask_t = mask_t.clone()                              # (1,H,W)

        return {
            "image":        image_t,                                 # (3, H, W) float32
            "mask":         mask_t,                                  # (1, H, W) float32 [0,1]
            "bbox":         torch.from_numpy(bbox),                  # (4,)      float32 pixel xyxy
            "points":       torch.from_numpy(pts),                   # (N, 2)    float32 pixel xy
            "point_labels": torch.from_numpy(lbls),                  # (N,)      int64
            "coarse_mask":  coarse_mask_t,                           # (1, H, W) float32
            "text":         self.cfg.text_label,                     # str
        }


# ---------------------------------------------------------------------------
# collate_fn — xử lý padding khi N points khác nhau giữa các sample
# ---------------------------------------------------------------------------

def collate_fn(batch: List[Dict]) -> Dict:
    """
    Custom collate cho PolypDataset.

    Vấn đề: số points (N) có thể khác nhau giữa các sample trong batch
    do mask có thể rỗng hoặc nhỏ. Cần pad về N_max.

    Returns dict với các key:
        "image"        : (B, 3, H, W)
        "mask"         : (B, 1, H, W)
        "bbox"         : (B, 4)
        "points"       : (B, N_max, 2)    — padded với 0
        "point_labels" : (B, N_max)       — padded với -1 (ignore label)
        "points_mask"  : (B, N_max)       — bool, True = valid point
        "coarse_mask"  : (B, 1, H, W)
        "text"         : List[str]
    """
    images = torch.stack([s["image"] for s in batch])
    masks = torch.stack([s["mask"] for s in batch])
    bboxes = torch.stack([s["bbox"] for s in batch])
    coarse_masks = torch.stack([s["coarse_mask"] for s in batch])
    texts = [s["text"] for s in batch]

    # Padding points
    n_max = max(s["points"].shape[0] for s in batch)
    B = len(batch)
    pts_padded = torch.zeros(B, n_max, 2, dtype=torch.float32)
    lbl_padded = torch.full((B, n_max), fill_value=-1, dtype=torch.int64)
    pts_mask = torch.zeros(B, n_max, dtype=torch.bool)

    for i, s in enumerate(batch):
        n = s["points"].shape[0]
        pts_padded[i, :n] = s["points"]
        lbl_padded[i, :n] = s["point_labels"]
        pts_mask[i, :n] = True

    return {
        "image":        images,       # (B, 3, H, W)
        "mask":         masks,        # (B, 1, H, W)
        "bbox":         bboxes,       # (B, 4)
        "points":       pts_padded,   # (B, N_max, 2)
        "point_labels": lbl_padded,   # (B, N_max)   -1 = padding
        "points_mask":  pts_mask,     # (B, N_max)   bool
        "coarse_mask":  coarse_masks, # (B, 1, H, W)
        "text":         texts,        # List[str]
    }
