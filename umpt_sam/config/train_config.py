"""Training configuration dataclasses for UMPA-SAM.

This module defines the three-phase optimization schedule described in PLAN.md.
It is intentionally standalone and not connected to the current trainer code.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class PhaseConfig:
    """Configuration for a single training phase."""

    epochs: int
    lambda_con: float
    freeze_image_encoder: bool
    freeze_prompt_encoder: bool
    freeze_mask_decoder: bool
    lr: float
    name: str = ""

    def __post_init__(self) -> None:
        if self.epochs <= 0:
            raise ValueError("epochs must be > 0")
        if self.lambda_con < 0:
            raise ValueError("lambda_con must be >= 0")
        if self.lr <= 0:
            raise ValueError("lr must be > 0")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class TrainConfig:
    """Top-level training config with the default 3-phase schedule."""

    batch_size: int = 2
    K: int = 3
    lambda_reg: float = 0.01
    phase1: PhaseConfig = field(
        default_factory=lambda: PhaseConfig(
            name="warmup",
            epochs=5,
            lambda_con=0.0,
            freeze_image_encoder=True,
            freeze_prompt_encoder=True,
            freeze_mask_decoder=True,
            lr=1e-4,
        )
    )
    phase2: PhaseConfig = field(
        default_factory=lambda: PhaseConfig(
            name="adaptation",
            epochs=5,
            lambda_con=0.0,
            freeze_image_encoder=True,
            freeze_prompt_encoder=False,
            freeze_mask_decoder=True,
            lr=5e-5,
        )
    )
    phase3: PhaseConfig = field(
        default_factory=lambda: PhaseConfig(
            name="consistency",
            epochs=10,
            lambda_con=0.5,
            freeze_image_encoder=True,
            freeze_prompt_encoder=False,
            freeze_mask_decoder=False,
            lr=1e-5,
        )
    )

    def __post_init__(self) -> None:
        if self.batch_size <= 0:
            raise ValueError("batch_size must be > 0")
        if self.K <= 0:
            raise ValueError("K must be > 0")
        if self.lambda_reg < 0:
            raise ValueError("lambda_reg must be >= 0")

    @property
    def total_epochs(self) -> int:
        return self.phase1.epochs + self.phase2.epochs + self.phase3.epochs

    def phases(self) -> tuple[PhaseConfig, PhaseConfig, PhaseConfig]:
        return (self.phase1, self.phase2, self.phase3)

    def get_phase(self, epoch: int) -> PhaseConfig:
        """Return the phase config for a zero-based global epoch index."""
        if epoch < 0:
            raise ValueError("epoch must be >= 0")

        cursor = 0
        for phase in self.phases():
            if cursor <= epoch < cursor + phase.epochs:
                return phase
            cursor += phase.epochs
        raise IndexError(
            f"epoch={epoch} is outside configured schedule with total_epochs={self.total_epochs}"
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
