"""Hungarian-matched DETR loss for UMPAv2 (Method 2a/2b).

Wraps SAM3 loss functions (dice, focal) and adds Hungarian matching
for multi-query training of all 200 DETR queries.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torch import Tensor, nn

# Reuse SAM3 loss primitives
from sam3.train.loss.loss_fns import (
    dice_loss as _sam3_dice_loss,
    sigmoid_focal_loss as _sam3_focal_loss,
)

from umpt_sam.losses.consistency_loss import MultiPromptConsistencyLoss


# ═══════════════════════════════════════════════════════════════════════
# Utilities
# ═══════════════════════════════════════════════════════════════════════

@torch.no_grad()
def masks_to_boxes(masks: Tensor) -> Tensor:
    """Derive xyxy bounding boxes from binary masks.

    Args:
        masks: ``[B, 1, H, W]`` binary masks.

    Returns:
        boxes: ``[B, 1, 4]`` xyxy normalised to [0, 1].
    """
    B, _, H, W = masks.shape
    device = masks.device
    boxes = torch.zeros(B, 1, 4, device=device)

    for i in range(B):
        m = masks[i, 0]
        if m.sum() == 0:
            continue
        rows = torch.any(m, dim=1)
        cols = torch.any(m, dim=0)
        rmin = torch.where(rows)[0].min().float()
        rmax = torch.where(rows)[0].max().float()
        cmin = torch.where(cols)[0].min().float()
        cmax = torch.where(cols)[0].max().float()
        boxes[i, 0] = torch.tensor(
            [cmin / W, rmin / H, cmax / W, rmax / H],
            device=device,
        )
    return boxes


# ═══════════════════════════════════════════════════════════════════════
# Hungarian Matcher
# ═══════════════════════════════════════════════════════════════════════

class HungarianMatcher(nn.Module):
    """Simple Hungarian matcher (no grad) for DETR-style training.

    Uses classification cost + L1 box cost + dice mask cost.
    """

    def __init__(
        self,
        cost_class: float = 2.0,
        cost_bbox: float = 5.0,
        cost_mask: float = 5.0,
    ) -> None:
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_mask = cost_mask

    @torch.no_grad()
    def forward(
        self,
        pred_logits: Tensor,
        pred_boxes: Tensor,
        pred_masks: Tensor,
        gt_masks: Tensor,
        gt_boxes: Tensor,
    ) -> List[Tuple[Tensor, Tensor]]:
        """Compute optimal assignment.

        Args:
            pred_logits: ``[B, N_q, 1]``
            pred_boxes:  ``[B, N_q, 4]`` xyxy
            pred_masks:  ``[B, N_q, H, W]`` logits
            gt_masks:    ``[B, 1, H, W]``
            gt_boxes:    ``[B, 1, 4]`` xyxy

        Returns:
            List of (pred_indices, gt_indices) per batch element.
        """
        B, N_q = pred_logits.shape[:2]
        indices = []

        for b in range(B):
            # Cast to float32 to avoid AMP float16 NaN issues
            # Classification cost: BCE-like
            out_prob = pred_logits[b].squeeze(-1).float().sigmoid()  # [N_q]
            cost_class = -out_prob.unsqueeze(1)  # lower = better match

            # Box cost: L1
            if pred_boxes is not None and gt_boxes is not None:
                cost_bbox = torch.cdist(
                    pred_boxes[b].float(), gt_boxes[b].float(), p=1,
                )  # [N_q, 1]
            else:
                cost_bbox = torch.zeros(N_q, 1, device=pred_logits.device)

            # Mask cost: Dice
            pred_m = pred_masks[b].flatten(1).float().sigmoid()  # [N_q, H*W]
            gt_m = gt_masks[b].flatten(1).float()  # [1, H*W]
            numerator = 2 * torch.mm(pred_m, gt_m.T)  # [N_q, 1]
            denominator = pred_m.sum(-1, keepdim=True) + gt_m.sum(-1)
            cost_dice = 1 - (numerator + 1) / (denominator + 1)

            # Combined cost
            C = (
                self.cost_class * cost_class
                + self.cost_bbox * cost_bbox
                + self.cost_mask * cost_dice
            )
            C = C.cpu().numpy()

            src_idx, tgt_idx = linear_sum_assignment(C)
            indices.append(
                (
                    torch.as_tensor(src_idx, dtype=torch.long, device=pred_logits.device),
                    torch.as_tensor(tgt_idx, dtype=torch.long, device=pred_logits.device),
                )
            )
        return indices


# ═══════════════════════════════════════════════════════════════════════
# Importance Regularisation (reuse from v2 losses)
# ═══════════════════════════════════════════════════════════════════════

class ImportanceRegLoss(nn.Module):
    """Entropy regularisation on PIM gate weights."""

    def __init__(self, weight: float = 0.1) -> None:
        super().__init__()
        self.weight = weight

    def forward(self, gate_weights: Optional[Tensor]) -> Tensor:
        if gate_weights is None or self.weight == 0:
            return torch.tensor(0.0, device=gate_weights.device if gate_weights is not None else "cpu")
        # gate_weights: [B, 4] ∈ (0, 1)
        eps = 1e-7
        w = gate_weights.clamp(eps, 1 - eps)
        entropy = -(w * w.log() + (1 - w) * (1 - w).log())  # [B, 4]
        return self.weight * entropy.mean()


# ═══════════════════════════════════════════════════════════════════════
# Main loss
# ═══════════════════════════════════════════════════════════════════════

class UMPAv2MatchedLoss(nn.Module):
    """Hungarian-matched DETR loss for UMPAv2.

    Components:
        1. Focal loss on classification logits (matched)
        2. Dice loss on mask predictions (matched)
        3. L1 + GIoU on box predictions (matched)
        4. Consistency loss (multi-prompt, K perturbations)
        5. Importance gate regularisation
    """

    def __init__(
        self,
        *,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        loss_ce_weight: float = 2.0,
        loss_bbox_weight: float = 5.0,
        loss_giou_weight: float = 2.0,
        loss_mask_weight: float = 5.0,
        loss_dice_weight: float = 5.0,
        consistency_weight: float = 0.5,
        importance_reg_weight: float = 0.1,
    ) -> None:
        super().__init__()
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.loss_ce_weight = loss_ce_weight
        self.loss_bbox_weight = loss_bbox_weight
        self.loss_giou_weight = loss_giou_weight
        self.loss_mask_weight = loss_mask_weight
        self.loss_dice_weight = loss_dice_weight
        self.consistency_weight = consistency_weight

        self.matcher = HungarianMatcher(
            cost_class=loss_ce_weight,
            cost_bbox=loss_bbox_weight,
            cost_mask=loss_dice_weight,
        )
        self.consistency_loss_fn = MultiPromptConsistencyLoss()
        self.importance_reg = ImportanceRegLoss(weight=importance_reg_weight)

    def forward(
        self,
        outputs: Dict[str, Tensor],
        gt_masks: Tensor,
        gt_boxes: Optional[Tensor] = None,
        perturbed_masks: Optional[Tensor] = None,
        gate_weights: Optional[Tensor] = None,
    ) -> Dict[str, Any]:
        """Compute matched loss.

        Args:
            outputs: Model output dict with keys:
                - ``all_pred_masks``: ``[B, N_q, H, W]``
                - ``pred_logits``: ``[B, N_q, 1]``
                - ``pred_boxes``: ``[B, N_q, 4]``
                - ``pred_masks``: ``[B, 1, H, W]`` (selected best)
            gt_masks: ``[B, 1, H, W]``
            gt_boxes: ``[B, 1, 4]`` (optional; derived from masks if None)
            perturbed_masks: ``[B, K, 1, H, W]`` for consistency loss
            gate_weights: ``[B, 4]`` for importance reg
        """
        device = gt_masks.device
        pred_masks_all = outputs.get("all_pred_masks")  # [B, N_q, H, W]
        pred_logits = outputs.get("pred_logits")  # [B, N_q, 1]
        pred_boxes = outputs.get("pred_boxes")  # [B, N_q, 4]

        # Derive GT boxes from masks if not provided
        if gt_boxes is None:
            gt_boxes = masks_to_boxes(gt_masks)

        # ── 1. Hungarian matching ─────────────────────────────────────
        if pred_masks_all is not None and pred_logits is not None:
            # Downsample masks for matching cost computation
            pred_masks_down = pred_masks_all
            if pred_masks_all.shape[-2:] != gt_masks.shape[-2:]:
                pred_masks_down = F.interpolate(
                    pred_masks_all,
                    size=gt_masks.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )

            indices = self.matcher(
                pred_logits=pred_logits,
                pred_boxes=pred_boxes,
                pred_masks=pred_masks_down,
                gt_masks=gt_masks,
                gt_boxes=gt_boxes,
            )

            # ── 2. Classification loss (focal) ────────────────────────
            src_logits = pred_logits.squeeze(-1)  # [B, N_q]
            target_classes = torch.zeros_like(src_logits)
            batch_idx = torch.cat(
                [torch.full_like(s, i) for i, (s, _) in enumerate(indices)]
            )
            src_idx = torch.cat([s for (s, _) in indices])
            target_classes[batch_idx, src_idx] = 1.0

            num_boxes = max(len(src_idx), 1)
            loss_ce = _sam3_focal_loss(
                src_logits.unsqueeze(-1),
                target_classes.unsqueeze(-1),
                num_boxes=num_boxes,
                alpha=self.focal_alpha,
                gamma=self.focal_gamma,
                triton=False,
            )

            # ── 3. Mask losses (dice + focal on matched pairs) ────────
            matched_pred_masks = pred_masks_down[batch_idx, src_idx]  # [M, H, W]
            gt_idx = torch.cat([t for (_, t) in indices])
            matched_gt_masks = gt_masks.squeeze(1)[batch_idx]  # [M, H, W]
            matched_gt_masks = matched_gt_masks.float()

            if matched_pred_masks.numel() > 0:
                loss_dice = _sam3_dice_loss(
                    matched_pred_masks.flatten(1),
                    matched_gt_masks.flatten(1),
                    num_boxes=num_boxes,
                )
                loss_mask = _sam3_focal_loss(
                    matched_pred_masks.flatten(1),
                    matched_gt_masks.flatten(1),
                    num_boxes=num_boxes,
                    alpha=self.focal_alpha,
                    gamma=self.focal_gamma,
                    triton=False,
                )
            else:
                loss_dice = torch.tensor(0.0, device=device)
                loss_mask = torch.tensor(0.0, device=device)

            # ── 4. Box losses ─────────────────────────────────────────
            if pred_boxes is not None:
                matched_pred_boxes = pred_boxes[batch_idx, src_idx]  # [M, 4]
                matched_gt_boxes = gt_boxes.squeeze(1)[batch_idx]  # [M, 4]
                if matched_pred_boxes.numel() > 0:
                    loss_bbox = F.l1_loss(
                        matched_pred_boxes, matched_gt_boxes, reduction="sum",
                    ) / num_boxes
                else:
                    loss_bbox = torch.tensor(0.0, device=device)
            else:
                loss_bbox = torch.tensor(0.0, device=device)

        else:
            # Fallback: no multi-query, use pred_masks directly (like v1)
            pred_masks_sel = outputs["pred_masks"]
            if pred_masks_sel.shape[-2:] != gt_masks.shape[-2:]:
                pred_masks_sel = F.interpolate(
                    pred_masks_sel,
                    size=gt_masks.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )

            num_boxes = gt_masks.shape[0]
            loss_dice = _sam3_dice_loss(
                pred_masks_sel.flatten(1),
                gt_masks.float().flatten(1),
                num_boxes=num_boxes,
            )
            loss_mask = _sam3_focal_loss(
                pred_masks_sel.flatten(1),
                gt_masks.float().flatten(1),
                num_boxes=num_boxes,
                alpha=self.focal_alpha,
                gamma=self.focal_gamma,
                triton=False,
            )
            loss_ce = torch.tensor(0.0, device=device)
            loss_bbox = torch.tensor(0.0, device=device)

        # ── 5. Consistency loss ───────────────────────────────────────
        if perturbed_masks is not None and self.consistency_weight > 0:
            loss_con = self.consistency_loss_fn(perturbed_masks)
        else:
            loss_con = torch.tensor(0.0, device=device)

        # ── 6. Importance regularisation ──────────────────────────────
        loss_reg = self.importance_reg(gate_weights)

        # ── Aggregate ─────────────────────────────────────────────────
        total = (
            self.loss_ce_weight * loss_ce
            + self.loss_dice_weight * loss_dice
            + self.loss_mask_weight * loss_mask
            + self.loss_bbox_weight * loss_bbox
            + self.consistency_weight * loss_con
            + loss_reg
        )

        return {
            "total_loss": total,
            "loss_ce": loss_ce,
            "loss_dice": loss_dice,
            "loss_mask": loss_mask,
            "loss_bbox": loss_bbox,
            "consistency_loss": loss_con,
            "importance_reg_loss": loss_reg,
        }
