"""Model configuration dataclasses for UMPA-SAM.

This module defines configuration objects only. They are not integrated into
the existing runtime code yet.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class MPPGConfig:
    """Hyperparameters for the Multi-Prompt Perturbation Generator."""

    sigma_b: float = 5.0
    gamma_range: tuple[float, float] = (-2.0, 2.0)
    rotation_range: float = 3.0
    sigma_p: float = 3.0
    q_flip: float = 0.05
    dilate_radius: int = 1
    erode_radius: int = 1
    warp_strength: float = 0.02
    sigma_t: float = 0.1

    def __post_init__(self) -> None:
        if self.sigma_b < 0:
            raise ValueError("sigma_b must be >= 0")
        if self.rotation_range < 0:
            raise ValueError("rotation_range must be >= 0")
        if self.sigma_p < 0:
            raise ValueError("sigma_p must be >= 0")
        if not 0.0 <= self.q_flip <= 1.0:
            raise ValueError("q_flip must be in [0, 1]")
        if self.dilate_radius < 0:
            raise ValueError("dilate_radius must be >= 0")
        if self.erode_radius < 0:
            raise ValueError("erode_radius must be >= 0")
        if self.warp_strength < 0:
            raise ValueError("warp_strength must be >= 0")
        if self.sigma_t < 0:
            raise ValueError("sigma_t must be >= 0")
        if len(self.gamma_range) != 2:
            raise ValueError("gamma_range must contain exactly 2 values")
        if self.gamma_range[0] > self.gamma_range[1]:
            raise ValueError("gamma_range must be ordered as (min, max)")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class UPFEConfig:
    """Hyperparameters for the Unified Prompt Fusion Encoder."""

    embed_dim: int = 256
    scoring_hidden_dim: int = 256

    def __post_init__(self) -> None:
        if self.embed_dim <= 0:
            raise ValueError("embed_dim must be > 0")
        if self.scoring_hidden_dim <= 0:
            raise ValueError("scoring_hidden_dim must be > 0")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class UMPAModelConfig:
    """Top-level model configuration for UMPA-SAM."""

    sam_checkpoint: str
    embed_dim: int = 256
    text_embed_dim: int = 512
    image_size: int = 1024
    freeze_image_encoder: bool = True
    mppg: MPPGConfig = field(default_factory=MPPGConfig)
    upfe: UPFEConfig = field(default_factory=UPFEConfig)

    def __post_init__(self) -> None:
        if self.embed_dim <= 0:
            raise ValueError("embed_dim must be > 0")
        if self.text_embed_dim <= 0:
            raise ValueError("text_embed_dim must be > 0")
        if self.image_size <= 0:
            raise ValueError("image_size must be > 0")
        if not self.sam_checkpoint:
            raise ValueError("sam_checkpoint must be a non-empty path")

    @property
    def sam_checkpoint_path(self) -> Path:
        return Path(self.sam_checkpoint)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def perturbation_kwargs(self) -> dict[str, Any]:
        """Compatibility helper for future PromptPerturbation wiring."""
        return self.mppg.to_dict()

    def upfe_kwargs(self) -> dict[str, Any]:
        """Compatibility helper for future UPFE wiring."""
        return {
            "embed_dim": self.upfe.embed_dim,
            "scoting_network_hidden_dim": self.upfe.scoring_hidden_dim,
        }
