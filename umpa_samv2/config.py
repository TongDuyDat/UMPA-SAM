"""Configuration dataclasses for UMPAv2.

Separate from v1 configs (``umpt_sam.config``) — no cross-dependency.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

# Import perturbation config from v1 (reuse as-is)
from umpt_sam.config.model_config import MPPGConfig


@dataclass(slots=True)
class PIMv2Config:
    """Hyperparameters for the Prompt Importance Module v2."""

    embed_dim: int = 256
    num_heads: int = 8
    gate_hidden_dim: int = 128
    dropout: float = 0.1

    def __post_init__(self) -> None:
        if self.embed_dim <= 0:
            raise ValueError("embed_dim must be > 0")
        if self.embed_dim % self.num_heads != 0:
            raise ValueError(
                f"embed_dim ({self.embed_dim}) must be divisible by "
                f"num_heads ({self.num_heads})"
            )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class LossConfig:
    """Loss weights for UMPAv2 training."""

    dice_weight: float = 1.0
    bce_weight: float = 1.0
    consistency_weight: float = 0.5
    importance_reg_weight: float = 0.1
    smooth: float = 1e-6

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class UMPAv2ModelConfig:
    """Top-level model configuration for UMPAv2 — Pipeline A (DETR Grounding).

    Components:
        - SAM3VLBackbone (frozen)
        - SAM3 GeometryEncoder / PromptEncoder (frozen)
        - TextProjection (new, trainable)
        - PIM v2 (new, trainable)
        - TransformerEncoder 6L (pretrained, fine-tune)
        - TransformerDecoder 6L (pretrained, fine-tune)
        - DotProductScoring (pretrained, fine-tune)
        - SegmentationHead (pretrained, fine-tune)
    """

    # SAM3 checkpoint (for loading pretrained backbone/encoder/decoder)
    sam_checkpoint: Optional[str] = None

    # Trained UMPAv2 checkpoint (for inference / resume)
    checkpoint_path: Optional[str] = None

    # Architecture
    embed_dim: int = 256
    text_embed_dim: int = 512
    image_size: int = 1008

    # Freeze control — Pipeline A components
    freeze_image_encoder: bool = True
    freeze_prompt_encoder: bool = True     # SAM3 geometry encoding (frozen)
    freeze_transformer_encoder: bool = False  # 6L encoder (fine-tune)
    freeze_transformer_decoder: bool = False  # 6L DETR decoder (fine-tune)
    freeze_segmentation_head: bool = False    # MaskFormer-style SegHead (fine-tune)
    freeze_dot_prod_scoring: bool = False     # Scoring head (fine-tune)

    # Sub-module configs
    pim: PIMv2Config = field(default_factory=PIMv2Config)
    mppg: MPPGConfig = field(default_factory=MPPGConfig)
    loss: LossConfig = field(default_factory=LossConfig)

    def __post_init__(self) -> None:
        if self.embed_dim <= 0:
            raise ValueError("embed_dim must be > 0")
        if self.text_embed_dim <= 0:
            raise ValueError("text_embed_dim must be > 0")
        if not self.checkpoint_path and not self.sam_checkpoint:
            raise ValueError(
                "Either 'checkpoint_path' (trained UMPAv2 weights) or "
                "'sam_checkpoint' (SAM3 pre-trained weights) must be provided."
            )

    @property
    def sam_checkpoint_path(self) -> Optional[Path]:
        return Path(self.sam_checkpoint) if self.sam_checkpoint else None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def perturbation_kwargs(self) -> dict[str, Any]:
        return self.mppg.to_dict()
