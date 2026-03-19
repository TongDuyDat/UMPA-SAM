# umpt_sam/training/phase_scheduler.py
"""Phase-aware scheduler implementing Differential Learning Rate.

Instead of binary freeze/unfreeze, each module receives a phase-specific
LR multiplier.  This ensures gradient always flows through the network
while protecting pre-trained weights from catastrophic forgetting.

LR_effective(group, phase) = phase.lr * phase.lr_multipliers[group_name]
"""

import logging
import torch.nn as nn

logger = logging.getLogger(__name__)


class PhaseScheduler:
    """Three-phase LR scheduler with per-module multipliers.

    Expects the optimizer to have named param groups identified by a
    ``"name"`` key (e.g., ``"upfe"``, ``"pe"``, ``"dec"``, ``"proj"``).
    """

    def __init__(self, train_config):
        self.config = train_config

        self.phase1_end = self.config.phase1.epochs
        self.phase2_end = self.phase1_end + self.config.phase2.epochs
        self.total_epochs = self.phase2_end + self.config.phase3.epochs

    def get_current_phase(self, epoch: int):
        """Return PhaseConfig for the given 1-based epoch."""
        if epoch <= self.phase1_end:
            return self.config.phase1
        elif epoch <= self.phase2_end:
            return self.config.phase2
        else:
            return self.config.phase3

    def get_lambda_con(self, epoch: int) -> float:
        return self.get_current_phase(epoch).lambda_con

    def apply_phase(self, model: nn.Module, epoch: int, optimizer=None):
        """Apply per-group LR multipliers for the current phase.

        Image encoder is ALWAYS frozen (requires_grad=False).
        All other modules remain unfrozen so that gradient flows through
        the entire network; their effective LR is controlled by the
        per-group multiplier.

        A multiplier of 0.0 will set both LR=0 and requires_grad=False
        for that group (true freeze, backward-compatible).
        """
        phase = self.get_current_phase(epoch)

        # Image encoder: always frozen, never in optimizer
        for param in model.image_encoder.parameters():
            param.requires_grad = False

        # Ensure all other trainable modules have requires_grad=True
        # (unless multiplier is exactly 0.0)
        _module_map = {
            "upfe": getattr(model, "upfe_encoder", None),
            "proj": getattr(model, "text_projection", None),
            "pe":   getattr(model, "prompt_encoder", None),
            "dec":  getattr(model, "sam_mask_decoder", None),
        }
        for group_name, module in _module_map.items():
            if module is None or isinstance(module, nn.Identity):
                continue
            mult = phase.lr_multipliers.get(group_name, 1.0)
            requires_grad = mult > 0.0
            for param in module.parameters():
                param.requires_grad = requires_grad

        # Update optimizer param group LRs
        if optimizer is not None:
            lr_log_parts = []
            for pg in optimizer.param_groups:
                group_name = pg.get("name", "unknown")
                multiplier = phase.lr_multipliers.get(group_name, 1.0)
                effective_lr = phase.lr * multiplier
                pg["lr"] = effective_lr
                lr_log_parts.append(f"{group_name}={effective_lr:.1e}")

            logger.info(
                "Phase '%s' | Base LR: %.1e | Effective: %s",
                phase.name, phase.lr, ", ".join(lr_log_parts),
            )

        return phase