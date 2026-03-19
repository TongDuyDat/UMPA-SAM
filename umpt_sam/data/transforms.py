"""
umpt_sam/data/transforms.py

TransformConfig + build_transforms cho PolypDataset.

Mỗi transform nhận TransformConfig — thêm augmentation mới
không cần sửa dataset hay model.

Lưu ý:
- Resize về 1024 là bắt buộc vì SAM image encoder cần đúng kích thước này.
- ColorJitter chỉ áp lên ảnh, không áp lên mask.
- HorizontalFlip / VerticalFlip áp đồng thời lên ảnh + mask + bbox + points
  (albumentations xử lý tự động qua additional_targets).
"""

from dataclasses import dataclass, field
from typing import Tuple

import albumentations as A


@dataclass
class TransformConfig:
    image_size: int = 1008                          # SAM standard

    # Geometric
    hflip_p: float = 0.5
    vflip_p: float = 0.5
    rotate90_p: float = 0.5

    # Elastic / distortion (OneOf group)
    elastic_p: float = 0.3                          # prob toàn nhóm
    elastic_alpha: float = 1.0
    elastic_sigma: float = 50.0
    grid_distort_p: float = 1.0
    optical_distort_p: float = 1.0
    optical_distort_limit: float = 0.2

    # Color (chỉ ảnh, mask không đổi)
    color_jitter_p: float = 0.3
    brightness: float = 0.2
    contrast: float = 0.2
    saturation: float = 0.2
    hue: float = 0.1


def build_transforms(cfg: TransformConfig, phase: str) -> A.Compose:
    """
    Trả về A.Compose tương thích với albumentations.
    Khai báo additional_targets để flip/rotate áp lên mask + bbox + keypoints.
    """
    if phase == "train":
        pipeline = A.Compose(
            [
                A.Resize(height=cfg.image_size, width=cfg.image_size),
                A.HorizontalFlip(p=cfg.hflip_p),
                A.VerticalFlip(p=cfg.vflip_p),
                A.RandomRotate90(p=cfg.rotate90_p),
                A.OneOf(
                    [
                        A.ElasticTransform(
                            alpha=cfg.elastic_alpha,
                            sigma=cfg.elastic_sigma,
                            p=1.0,
                        ),
                        A.GridDistortion(p=cfg.grid_distort_p),
                        A.OpticalDistortion(
                            distort_limit=cfg.optical_distort_limit,
                            mode="camera",
                            p=cfg.optical_distort_p,
                        ),
                    ],
                    p=cfg.elastic_p,
                ),
                A.ColorJitter(
                    brightness=cfg.brightness,
                    contrast=cfg.contrast,
                    saturation=cfg.saturation,
                    hue=cfg.hue,
                    p=cfg.color_jitter_p,
                ),
            ],
            is_check_shapes=False,
        )
    else:  # val / test
        pipeline = A.Compose(
            [A.Resize(height=cfg.image_size, width=cfg.image_size)],
            is_check_shapes=False,
        )

    return pipeline
