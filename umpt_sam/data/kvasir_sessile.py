# config.py
import os
from pathlib import Path
import albumentations as A

# Resolve dataset path relative to this file's location (umpt_sam/data/)
_DATA_DIR = Path(__file__).resolve().parent
DATASET_SOURCE = str(_DATA_DIR / "kvasir-sessile" / "sessile-main-Kvasir-SEG")
IMAGE_SIZE = (1008, 1008)
NORMALIZE = True

TRANSFORM_PIPELINE = {
    "train": A.Compose(
        [
            A.Resize(height=IMAGE_SIZE[0], width=IMAGE_SIZE[1]),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.OneOf(
                [
                    A.ElasticTransform(alpha=1.0, sigma=50, p=1.0),
                    A.GridDistortion(p=1.0),
                    A.OpticalDistortion(distort_limit=0.2, mode="camera", p=1.0),
                ],
                p=0.3,
            ),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2, p=0.3),
        ],
        is_check_shapes=False,
    ),
    "val": A.Compose(
        [A.Resize(height=IMAGE_SIZE[0], width=IMAGE_SIZE[1])],
        is_check_shapes=False,
    ),
}
