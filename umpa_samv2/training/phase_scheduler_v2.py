"""Phase scheduler for UMPAv2 — 3-phase freeze/unfreeze (Method 1).

Adapts v1 ``PhaseScheduler`` for Pipeline A components.
"""

from __future__ import annotations

import logging

import torch.nn as nn

from .config_v2 import PhaseV2Config, TrainV2Config

logger = logging.getLogger(__name__)


def _set_requires_grad(module: nn.Module, requires_grad: bool) -> None:
    """Set ``requires_grad`` on all parameters of *module*."""
    if module is None:
        return
    for p in module.parameters():
        p.requires_grad = requires_grad


class PhaseSchedulerV2:
    """3-phase scheduler for UMPAv2 Pipeline A components.

    Phase boundaries::

        Phase 1:  epoch 1 .. phase1.epochs
        Phase 2:  phase1.epochs+1 .. phase1.epochs + phase2.epochs
        Phase 3:  phase2_end+1 .. total_epochs
    """

    def __init__(self, train_config: TrainV2Config) -> None:
        self.config = train_config
        self.phase1_end = train_config.phase1.epochs
        self.phase2_end = self.phase1_end + train_config.phase2.epochs
        self.total_epochs = train_config.total_epochs

    def get_current_phase(self, epoch: int) -> PhaseV2Config:
        """Return the phase config for 1-indexed *epoch*."""
        if epoch <= self.phase1_end:
            return self.config.phase1
        if epoch <= self.phase2_end:
            return self.config.phase2
        return self.config.phase3

    def apply_phase(
        self,
        model: nn.Module,
        epoch: int,
        optimizer=None,
    ) -> PhaseV2Config:
        """Freeze/unfreeze components and update LR for *epoch*."""
        phase = self.get_current_phase(epoch)

        # Always frozen
        _set_requires_grad(model.image_encoder, False)
        _set_requires_grad(model.geometry_encoder, False)

        # Always trainable (UMPAv2-specific)
        _set_requires_grad(model.text_projection, True)
        _set_requires_grad(model.pim, True)
        # Perturbation has no trainable params but be explicit
        _set_requires_grad(model.perturbation, True)

        # Per-phase freeze for Pipeline A components
        _set_requires_grad(
            model.transformer.encoder,
            not phase.freeze_transformer_encoder,
        )
        _set_requires_grad(
            model.transformer.decoder,
            not phase.freeze_transformer_decoder,
        )
        _set_requires_grad(
            model.segmentation_head,
            not phase.freeze_segmentation_head,
        )
        _set_requires_grad(
            model.dot_prod_scoring,
            not phase.freeze_dot_prod_scoring,
        )

        # Update LR
        if optimizer is not None:
            for pg in optimizer.param_groups:
                pg["lr"] = phase.lr

        trainable = sum(
            p.numel() for p in model.parameters() if p.requires_grad
        )
        logger.info(
            "Phase=%s epoch=%d lr=%.2e trainable=%.1fM",
            phase.name, epoch, phase.lr, trainable / 1e6,
        )

        return phase
