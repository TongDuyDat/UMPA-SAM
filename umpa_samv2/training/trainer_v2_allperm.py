"""UMPAv2 Trainer — Method 2b (All-Permutation Average).

Extends SAM3-style trainer with AllPermutationLoss that computes
multiple permuted forwards + KL consistency.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any, Dict, List

import torch
import torch.nn.functional as F
from tqdm import tqdm

from .config_v2 import TrainV2AllPermConfig
from .trainer_v2_sam3 import UMPAv2SAM3Trainer


class UMPAv2AllPermTrainer(UMPAv2SAM3Trainer):
    """Trainer using AllPermutationLoss with backbone caching.

    Inherits optimizer, LR schedule, logging from ``UMPAv2SAM3Trainer``.
    Overrides ``train_one_epoch`` to use ``AllPermutationLoss``.
    """

    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        test_loader,
        config: TrainV2AllPermConfig,
        all_perm_loss,
        evaluate_fn,
        save_dir: str = "checkpoints",
        device: str = "cuda",
    ) -> None:
        # AllPermutationLoss contains matched_loss internally
        super().__init__(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            config=config,
            matched_loss=all_perm_loss.matched_loss,
            evaluate_fn=evaluate_fn,
            save_dir=save_dir,
            device=device,
        )
        self.all_perm_loss = all_perm_loss

    def _init_log_file(self) -> None:
        cfg = self.config
        log = (
            f"\n{'='*60}\n"
            f"UMPAv2 Method 2b — All-Permutation Average\n"
            f"Save: {self.save_dir}\n"
            f"Epochs: {cfg.total_epochs}, Warmup: {cfg.warmup_epochs}\n"
            f"LR: {cfg.lr} → {cfg.min_lr} ({cfg.lr_schedule})\n"
            f"N_perms: {cfg.n_permutations}, λ_perm: {cfg.lambda_perm}\n"
            f"Grad mode: {cfg.perm_grad_mode}\n"
            f"K: {cfg.K}, λ_con: {cfg.consistency_weight}\n"
            f"{'='*60}\n"
        )
        with open(self.log_file_path, "w", encoding="utf-8") as f:
            f.write(log)
        print(log)

    def train_one_epoch(self, epoch: int) -> Dict[str, float]:
        """Override: uses AllPermutationLoss instead of matched_loss."""
        self.model.train()
        cfg = self.config

        epoch_losses: Dict[str, float] = {}
        pbar = tqdm(
            self.train_loader,
            desc=f"Train E{epoch} (AllPerm)",
            leave=False,
        )

        for batch in pbar:
            lr = self._update_lr(self.global_step)

            images = batch["image"].to(self.device, non_blocking=True)
            gt_masks = batch["mask"].to(self.device, non_blocking=True)

            boxes = batch.get("bbox")
            points = batch.get("points")
            point_labels = batch.get("point_labels")
            captions = batch.get("text")
            masks_input = batch.get("coarse_mask")

            if boxes is not None:
                boxes = boxes.to(self.device, non_blocking=True)
            if points is not None:
                points = points.to(self.device, non_blocking=True)
            if point_labels is not None:
                point_labels = point_labels.to(self.device, non_blocking=True)
            if masks_input is not None:
                masks_input = masks_input.to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)

            amp_dtype = (
                torch.bfloat16 if cfg.amp_dtype == "bfloat16" else torch.float16
            )
            with torch.amp.autocast(
                self.device, enabled=cfg.amp_enabled, dtype=amp_dtype,
            ):
                # AllPermutationLoss handles all forwards internally
                loss_dict = self.all_perm_loss(
                    model=self.model,
                    image=images,
                    gt_masks=gt_masks,
                    gt_boxes=None,
                    boxes=boxes,
                    points=points,
                    point_labels=point_labels,
                    masks=masks_input,
                    captions=captions,
                )
                total_loss = loss_dict["total_loss"]

            scaled_loss = total_loss / cfg.gradient_accumulation_steps
            self.scaler.scale(scaled_loss).backward()

            if (self.global_step + 1) % cfg.gradient_accumulation_steps == 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), cfg.grad_clip_norm,
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()

            self.global_step += 1

            for k, v in loss_dict.items():
                val = v.item() if isinstance(v, torch.Tensor) else float(v)
                epoch_losses[k] = epoch_losses.get(k, 0.0) + val

            pbar.set_postfix(
                L=f"{total_loss.item():.4f}",
                perm=f"{loss_dict.get('perm_consistency_loss', torch.tensor(0)).item():.4f}",
                lr=f"{lr:.2e}",
            )

        n = len(self.train_loader)
        return {k: v / n for k, v in epoch_losses.items()}
