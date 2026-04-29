"""Training configuration for UMPAv2.

Defines phase-based config (Method 1) and continuous config (Method 2a/2b).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict


# ═══════════════════════════════════════════════════════════════════════
# Method 1 — Phase-based training config
# ═══════════════════════════════════════════════════════════════════════


@dataclass(slots=True)
class PhaseV2Config:
    """Single phase config for UMPAv2 Pipeline A."""

    name: str = ""
    epochs: int = 5
    lr: float = 1e-4
    lambda_con: float = 0.0

    # Freeze flags for Pipeline A components
    freeze_image_encoder: bool = True
    freeze_geometry_encoder: bool = True
    freeze_transformer_encoder: bool = True
    freeze_transformer_decoder: bool = True
    freeze_segmentation_head: bool = True
    freeze_dot_prod_scoring: bool = True

    def __post_init__(self) -> None:
        if self.epochs <= 0:
            raise ValueError("epochs must be > 0")
        if self.lr <= 0:
            raise ValueError("lr must be > 0")
        if self.lambda_con < 0:
            raise ValueError("lambda_con must be >= 0")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class TrainV2Config:
    """3-phase training config for UMPAv2 (Method 1)."""

    batch_size: int = 4
    K: int = 3

    phase1: PhaseV2Config = field(
        default_factory=lambda: PhaseV2Config(
            name="warmup",
            epochs=5,
            lr=1e-4,
            lambda_con=0.0,
            freeze_image_encoder=True,
            freeze_geometry_encoder=True,
            freeze_transformer_encoder=True,
            freeze_transformer_decoder=True,
            freeze_segmentation_head=True,
            freeze_dot_prod_scoring=True,
        )
    )
    phase2: PhaseV2Config = field(
        default_factory=lambda: PhaseV2Config(
            name="adaptation",
            epochs=5,
            lr=5e-5,
            lambda_con=0.0,
            freeze_image_encoder=True,
            freeze_geometry_encoder=True,
            freeze_transformer_encoder=False,
            freeze_transformer_decoder=False,
            freeze_segmentation_head=False,
            freeze_dot_prod_scoring=False,
        )
    )
    phase3: PhaseV2Config = field(
        default_factory=lambda: PhaseV2Config(
            name="consistency",
            epochs=10,
            lr=1e-5,
            lambda_con=0.5,
            freeze_image_encoder=True,
            freeze_geometry_encoder=True,
            freeze_transformer_encoder=False,
            freeze_transformer_decoder=False,
            freeze_segmentation_head=False,
            freeze_dot_prod_scoring=False,
        )
    )

    # Loss weights
    dice_weight: float = 1.0
    bce_weight: float = 1.0
    consistency_weight: float = 0.5
    importance_reg_weight: float = 0.1

    @property
    def total_epochs(self) -> int:
        return self.phase1.epochs + self.phase2.epochs + self.phase3.epochs

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ═══════════════════════════════════════════════════════════════════════
# Method 2 — Continuous SAM3-style training config
# ═══════════════════════════════════════════════════════════════════════


@dataclass(slots=True)
class TrainV2SAM3Config:
    """Continuous training config for UMPAv2 (Method 2a/2b)."""

    # Schedule
    total_epochs: int = 20
    warmup_epochs: int = 2
    lr: float = 1e-4
    min_lr: float = 1e-6
    lr_schedule: str = "cosine"  # "cosine" | "step" | "linear"

    # Optimizer
    weight_decay: float = 0.05
    transformer_lr_mult: float = 0.1

    # Loss weights — DETR-style
    focal_alpha: float = 0.25
    focal_gamma: float = 2.0
    loss_ce_weight: float = 2.0
    loss_bbox_weight: float = 5.0
    loss_giou_weight: float = 2.0
    loss_mask_weight: float = 5.0
    loss_dice_weight: float = 5.0

    # UMPA-specific loss weights
    consistency_weight: float = 0.5
    importance_reg_weight: float = 0.1

    # Perturbation
    K: int = 3
    enable_permutation_aug: bool = True

    # Runtime
    batch_size: int = 4
    gradient_accumulation_steps: int = 1
    amp_enabled: bool = True
    amp_dtype: str = "float16"
    grad_clip_norm: float = 0.1

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class TrainV2AllPermConfig(TrainV2SAM3Config):
    """All-permutation training config (Method 2b)."""

    n_permutations: int = 6
    lambda_perm: float = 0.1
    perm_temperature: float = 1.0
    perm_grad_mode: str = "canonical_only"  # "canonical_only" | "all"
    enable_permutation_aug: bool = False  # disabled — uses all-perm instead
