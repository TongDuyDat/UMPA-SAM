"""Loss functions for UMPAv2.

- ``ImportanceRegularizationLoss``: Binary entropy pushing PIM gates to 0 or 1.
- ``ComposerV2Loss``: Full training objective combining Dice + BCE + Consistency + ImportanceReg.

Reuses ``DiceLoss`` and ``MultiPromptConsistencyLoss`` from v1.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from umpt_sam.losses.dice_loss import DiceLoss
from umpt_sam.losses.consistency_loss import MultiPromptConsistencyLoss


# ═════════════════════════════════════════════════════════════════════
# Importance Regularization — Binary Entropy
# ═════════════════════════════════════════════════════════════════════

class ImportanceRegularizationLoss(nn.Module):
    """Binary entropy regularization for PIM gate weights.

    Pushes each gate weight toward 0 or 1 (decisive gating):

    $$\\mathcal{H}(w) = -[w \\log(w + \\epsilon) + (1-w) \\log(1-w + \\epsilon)]$$

    $$\\mathcal{L}_{\\text{reg}} = \\sum_{i \\in \\{t, b, m\\}} \\mathcal{H}(w_i)$$

    This is maximized at w=0.5 (uncertain), minimized at w∈{0,1} (decisive).
    The loss *rewards* decisive gating by penalizing uncertainty.
    """

    def __init__(self, epsilon: float = 1e-7) -> None:
        super().__init__()
        self.epsilon = epsilon

    def forward(self, gate_weights: Tensor) -> Tensor:
        """Compute binary entropy loss.

        Args:
            gate_weights: ``[B, 4]`` — PIM gate weights ∈ (0, 1).
                          Columns: ``(w_t, w_b, w_p, w_m)``.

        Returns:
            Scalar loss — mean binary entropy over batch and gates.
        """
        eps = self.epsilon
        # H(w) = -[w·log(w+ε) + (1-w)·log(1-w+ε)]
        entropy = -(
            gate_weights * torch.log(gate_weights + eps)
            + (1 - gate_weights) * torch.log(1 - gate_weights + eps)
        )
        # Sum over 4 gates, mean over batch
        return entropy.sum(dim=-1).mean()


# ═════════════════════════════════════════════════════════════════════
# Composer V2 Loss — Full Training Objective
# ═════════════════════════════════════════════════════════════════════

class ComposerV2Loss(nn.Module):
    """Full UMPAv2 training loss.

    $$\\mathcal{L} = \\mathcal{L}_{\\text{Dice}} + \\lambda_{\\text{bce}} \\mathcal{L}_{\\text{BCE}}
    + \\lambda_{\\text{con}} \\mathcal{L}_{\\text{con}}
    + \\lambda_{\\text{reg}} \\sum_i \\mathcal{H}(w_i)$$
    """

    def __init__(
        self,
        dice_weight: float = 1.0,
        bce_weight: float = 1.0,
        consistency_weight: float = 0.5,
        importance_reg_weight: float = 0.1,
        smooth: float = 1e-6,
    ) -> None:
        super().__init__()
        self.dice_loss = DiceLoss(smooth=smooth)
        self.consistency_loss = MultiPromptConsistencyLoss()
        self.importance_reg = ImportanceRegularizationLoss()

        self.bce_weight = bce_weight
        self.consistency_weight = consistency_weight
        self.importance_reg_weight = importance_reg_weight

    @classmethod
    def from_config(cls, loss_config) -> "ComposerV2Loss":
        """Build from ``LossConfig`` dataclass."""
        return cls(
            dice_weight=loss_config.dice_weight,
            bce_weight=loss_config.bce_weight,
            consistency_weight=loss_config.consistency_weight,
            importance_reg_weight=loss_config.importance_reg_weight,
            smooth=loss_config.smooth,
        )

    def forward(
        self,
        pred_masks: Tensor,
        gt_masks: Tensor,
        perturbed_masks: Optional[Tensor] = None,
        gate_weights: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        """Compute full training loss.

        Args:
            pred_masks:      ``[B, 1, H, W]`` — predicted masks (logits or probs).
            gt_masks:        ``[B, 1, H, W]`` — ground truth masks.
            perturbed_masks: ``[B, K, 1, H, W]`` — K perturbed predictions (optional).
            gate_weights:    ``[B, 3]`` — PIM gate weights (optional).

        Returns:
            Dict with keys: ``total_loss``, ``seg_loss``, ``bce_loss``,
            ``consistency_loss``, ``importance_reg_loss``.
        """
        # ── Segmentation loss ────────────────────────────────────────
        pred_probs = torch.sigmoid(pred_masks)
        dice = self.dice_loss(pred_probs, gt_masks)
        bce = F.binary_cross_entropy_with_logits(pred_masks, gt_masks)
        seg_loss = dice + self.bce_weight * bce

        # ── Consistency loss ─────────────────────────────────────────
        if perturbed_masks is not None and self.consistency_weight > 0:
            con_loss = self.consistency_loss(perturbed_masks)
        else:
            con_loss = torch.tensor(0.0, device=pred_masks.device)

        # ── Importance regularization ────────────────────────────────
        if gate_weights is not None and self.importance_reg_weight > 0:
            reg_loss = self.importance_reg(gate_weights)
        else:
            reg_loss = torch.tensor(0.0, device=pred_masks.device)

        # ── Total ────────────────────────────────────────────────────
        total = (
            seg_loss
            + self.consistency_weight * con_loss
            + self.importance_reg_weight * reg_loss
        )

        return {
            "total_loss": total,
            "seg_loss": seg_loss,
            "bce_loss": bce,
            "consistency_loss": con_loss,
            "importance_reg_loss": reg_loss,
        }
