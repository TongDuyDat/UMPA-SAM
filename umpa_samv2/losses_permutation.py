"""Permutation consistency loss for UMPAv2 (Method 2b).

KL-divergence between canonical and permuted mask predictions,
enforcing output invariance regardless of modality concatenation order.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class PermutationConsistencyLoss(nn.Module):
    """Symmetric KL-divergence consistency between canonical and permuted outputs.

    $$L_{perm} = \\frac{1}{|P|} \\sum_{\\pi \\in P}
        \\frac{1}{2} \\left[
            KL(\\sigma(m_{canon}) \\| \\sigma(m_\\pi))
          + KL(\\sigma(m_\\pi) \\| \\sigma(m_{canon}))
        \\right]$$
    """

    def __init__(
        self,
        temperature: float = 1.0,
        symmetric: bool = True,
    ) -> None:
        super().__init__()
        self.temperature = temperature
        self.symmetric = symmetric

    def forward(
        self,
        canonical_mask: Tensor,
        permuted_masks: List[Tensor],
    ) -> Tensor:
        """Compute permutation consistency loss.

        Args:
            canonical_mask: ``[B, 1, H, W]`` logits from canonical order.
            permuted_masks: List of ``[B, 1, H, W]`` logits from permuted orders.

        Returns:
            Scalar loss.
        """
        if not permuted_masks:
            return torch.tensor(0.0, device=canonical_mask.device)

        p_canon = torch.sigmoid(canonical_mask / self.temperature)
        eps = 1e-8

        kl_total = torch.tensor(0.0, device=canonical_mask.device)
        for p_mask in permuted_masks:
            p_perm = torch.sigmoid(p_mask / self.temperature)

            # Forward KL: KL(canon || perm)
            log_canon = torch.log(p_canon + eps)
            log_perm = torch.log(p_perm + eps)

            kl_fwd = F.kl_div(
                log_perm, p_canon, reduction="batchmean", log_target=False,
            )

            if self.symmetric:
                kl_bwd = F.kl_div(
                    log_canon, p_perm, reduction="batchmean", log_target=False,
                )
                kl_total = kl_total + (kl_fwd + kl_bwd) / 2
            else:
                kl_total = kl_total + kl_fwd

        return kl_total / len(permuted_masks)


class AllPermutationLoss(nn.Module):
    """Wraps matched loss + permutation consistency for Method 2b.

    Computes:
    1. Canonical forward (with gradient) → matched loss
    2. N sampled permutation forwards (no gradient) → averaged matched loss
    3. KL consistency between canonical and permuted predictions
    """

    def __init__(
        self,
        matched_loss: nn.Module,
        n_permutations: int = 6,
        lambda_perm: float = 0.1,
        perm_temperature: float = 1.0,
        perm_grad_mode: str = "canonical_only",
    ) -> None:
        super().__init__()
        self.matched_loss = matched_loss
        self.perm_consistency = PermutationConsistencyLoss(
            temperature=perm_temperature, symmetric=True,
        )
        self.n_permutations = n_permutations
        self.lambda_perm = lambda_perm
        self.perm_grad_mode = perm_grad_mode

    def _sample_permutations(self) -> List[List[int]]:
        """Sample N-1 random permutations (canonical is always index 0)."""
        perms = []
        seen = {(0, 1, 2, 3)}  # canonical already handled
        while len(perms) < self.n_permutations - 1:
            p = torch.randperm(4).tolist()
            key = tuple(p)
            if key not in seen:
                seen.add(key)
                perms.append(p)
        return perms

    def forward(
        self,
        model: nn.Module,
        image: Tensor,
        gt_masks: Tensor,
        gt_boxes: Optional[Tensor],
        *,
        boxes: Optional[Tensor] = None,
        points: Optional[Tensor] = None,
        point_labels: Optional[Tensor] = None,
        masks: Optional[Tensor] = None,
        captions=None,
    ) -> Dict[str, Tensor]:
        """Compute all-permutation loss.

        Uses model's forward with explicit ``permute_order`` to avoid
        redundant backbone computation (backbone caching is implicit
        via model's internal ``torch.no_grad()`` on frozen backbone).
        """
        device = image.device

        # ── 1. Canonical forward (with gradient) ─────────────────────
        canonical_outputs = model(
            image=image,
            boxes=boxes,
            points=points,
            point_labels=point_labels,
            masks=masks,
            captions=captions,
            return_gate_weights=True,
            return_all_queries=True,
            enable_permutation=False,
            permute_order=None,  # canonical order
        )

        canonical_loss_dict = self.matched_loss(
            outputs=canonical_outputs,
            gt_masks=gt_masks,
            gt_boxes=gt_boxes,
            gate_weights=canonical_outputs.get("gate_weights"),
        )

        canonical_mask = canonical_outputs["pred_masks"]  # [B, 1, H, W]

        # ── 2. Permuted forwards ─────────────────────────────────────
        perms = self._sample_permutations()
        perm_masks: List[Tensor] = []
        perm_losses: List[float] = []

        ctx = torch.no_grad if self.perm_grad_mode == "canonical_only" else torch.enable_grad

        with ctx():
            for perm in perms:
                perm_outputs = model(
                    image=image,
                    boxes=boxes,
                    points=points,
                    point_labels=point_labels,
                    masks=masks,
                    captions=captions,
                    return_gate_weights=False,
                    return_all_queries=True,
                    enable_permutation=False,
                    permute_order=perm,
                )
                perm_masks.append(perm_outputs["pred_masks"].detach())

                perm_loss_dict = self.matched_loss(
                    outputs=perm_outputs,
                    gt_masks=gt_masks,
                    gt_boxes=gt_boxes,
                )
                perm_losses.append(perm_loss_dict["total_loss"].item())

        # ── 3. Averaged matched loss ─────────────────────────────────
        avg_perm_loss = sum(perm_losses) / len(perm_losses) if perm_losses else 0.0
        combined_matched = (
            canonical_loss_dict["total_loss"] + avg_perm_loss
        ) / (1 + len(perm_losses))

        # ── 4. Permutation consistency loss ──────────────────────────
        loss_perm = self.perm_consistency(canonical_mask, perm_masks)

        # ── 5. Total ─────────────────────────────────────────────────
        total = canonical_loss_dict["total_loss"] + self.lambda_perm * loss_perm

        return {
            "total_loss": total,
            "matched_loss_canonical": canonical_loss_dict["total_loss"],
            "matched_loss_avg": combined_matched,
            "perm_consistency_loss": loss_perm,
            **{k: v for k, v in canonical_loss_dict.items() if k != "total_loss"},
        }
