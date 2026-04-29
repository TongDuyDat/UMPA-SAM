"""UMPAv2 Trainer — Method 2a (SAM3-style + Random Permutation Augmentation).

Single continuous training with cosine LR warmup, Hungarian-matched loss,
and random modality permutation augmentation in PIM.
"""

from __future__ import annotations

import json
import math
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F
from torch import optim
from tqdm import tqdm

from .config_v2 import TrainV2SAM3Config


class UMPAv2SAM3Trainer:
    """SAM3-style trainer with continuous schedule + permutation aug."""

    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        test_loader,
        config: TrainV2SAM3Config,
        matched_loss,
        evaluate_fn,
        save_dir: str = "checkpoints",
        device: str = "cuda",
    ) -> None:
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.config = config
        self.matched_loss = matched_loss
        self.evaluate_fn = evaluate_fn
        self.device = device
        self.best_val_dice = 0.0

        # Freeze backbone + geometry encoder (always frozen in all methods)
        self._freeze_backbone()

        # Build optimizer with param groups (only trainable params)
        self.optimizer = self._build_optimizer()
        self.scaler = torch.amp.GradScaler(
            device, enabled=config.amp_enabled,
        )

        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.save_dir = os.path.join(save_dir, f"run_{run_id}")
        os.makedirs(self.save_dir, exist_ok=True)
        self.log_file_path = os.path.join(self.save_dir, "training_log.txt")

        self.global_step = 0
        self.steps_per_epoch = len(train_loader)
        self.total_steps = config.total_epochs * self.steps_per_epoch

        self._init_log_file()

    # ── Freeze frozen components ────────────────────────────────────

    def _freeze_backbone(self) -> None:
        """Freeze pretrained SAM3 components.

        Frozen modules:
          - ``image_encoder`` (SAM3VLBackbone): contains both the
            **image backbone** (Hiera) and the **CLIP text encoder**.
            Freezing this single module freezes both image and text encoding.
          - ``geometry_encoder`` (SAMGeometryEncoder): wraps SAM3
            PromptEncoder for box/point/mask positional embeddings.

        These use pretrained SAM3/CLIP weights and must NOT be updated.
        """
        frozen_count = 0
        frozen_modules = {
            "image_encoder (image backbone + CLIP text)": self.model.image_encoder,
            "geometry_encoder (PromptEncoder)": self.model.geometry_encoder,
        }
        for name, module in frozen_modules.items():
            if module is None:
                continue
            n = 0
            for p in module.parameters():
                p.requires_grad = False
                n += p.numel()
            frozen_count += n

        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        import logging
        logging.info(
            "Frozen: %.1fM params (image backbone + CLIP text + geometry) "
            "| Trainable: %.1fM params",
            frozen_count / 1e6, trainable / 1e6,
        )

    # ── Optimizer ────────────────────────────────────────────────────

    def _build_optimizer(self) -> optim.AdamW:
        """Param groups with different LR multipliers."""
        cfg = self.config
        params: List[Dict[str, Any]] = []

        # PIM + TextProjection: full LR
        params.append({
            "params": list(self.model.pim.parameters())
                    + list(self.model.text_projection.parameters()),
            "lr": cfg.lr,
            "name": "pim_text",
        })

        # Perturbation (if has trainable params)
        pert_params = list(self.model.perturbation.parameters())
        if pert_params:
            params.append({
                "params": pert_params,
                "lr": cfg.lr,
                "name": "perturbation",
            })

        # Transformer + heads: reduced LR
        transformer_params = (
            list(self.model.transformer.parameters())
            + list(self.model.dot_prod_scoring.parameters())
            + list(self.model.segmentation_head.parameters())
        )
        if transformer_params:
            params.append({
                "params": transformer_params,
                "lr": cfg.lr * cfg.transformer_lr_mult,
                "name": "transformer_heads",
            })

        return optim.AdamW(params, weight_decay=cfg.weight_decay)

    # ── LR Schedule ──────────────────────────────────────────────────

    def _get_lr(self, step: int) -> float:
        """Cosine warmup decay LR."""
        cfg = self.config
        warmup_steps = cfg.warmup_epochs * self.steps_per_epoch

        if step < warmup_steps:
            return cfg.lr * step / max(warmup_steps, 1)

        progress = (step - warmup_steps) / max(self.total_steps - warmup_steps, 1)
        progress = min(progress, 1.0)

        if cfg.lr_schedule == "cosine":
            return cfg.min_lr + 0.5 * (cfg.lr - cfg.min_lr) * (
                1 + math.cos(math.pi * progress)
            )
        elif cfg.lr_schedule == "linear":
            return cfg.lr - (cfg.lr - cfg.min_lr) * progress
        else:
            return cfg.lr  # constant

    def _update_lr(self, step: int) -> float:
        base_lr = self._get_lr(step)
        cfg = self.config
        for pg in self.optimizer.param_groups:
            if pg.get("name") == "transformer_heads":
                pg["lr"] = base_lr * cfg.transformer_lr_mult
            else:
                pg["lr"] = base_lr
        return base_lr

    # ── Train one epoch ──────────────────────────────────────────────

    def train_one_epoch(self, epoch: int) -> Dict[str, float]:
        self.model.train()
        cfg = self.config

        epoch_losses: Dict[str, float] = {}
        pbar = tqdm(
            self.train_loader,
            desc=f"Train E{epoch}",
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
            # Forward pass in mixed precision
            with torch.amp.autocast(
                self.device, enabled=cfg.amp_enabled, dtype=amp_dtype,
            ):
                outputs = self.model(
                    image=images,
                    boxes=boxes,
                    points=points,
                    point_labels=point_labels,
                    masks=masks_input,
                    captions=captions,
                    return_gate_weights=True,
                    return_all_queries=True,
                    enable_permutation=cfg.enable_permutation_aug,
                )

                # K-perturbation for consistency
                perturbed_masks = None
                if cfg.K > 0 and cfg.consistency_weight > 0:
                    k_outputs = self.model.forward_k_perturbations(
                        image=images, K=cfg.K,
                        boxes=boxes, points=points,
                        point_labels=point_labels,
                        masks=masks_input, captions=captions,
                    )
                    p_list = []
                    for out in k_outputs:
                        pm = out["pred_masks"]
                        if pm.shape[-2:] != gt_masks.shape[-2:]:
                            pm = F.interpolate(
                                pm, size=gt_masks.shape[-2:],
                                mode="bilinear", align_corners=False,
                            )
                        p_list.append(pm)
                    perturbed_masks = torch.stack(p_list, dim=1)

            # Loss in float32 (outside autocast) — prevents
            # float16 overflow in sigmoid/matmul during Hungarian matching
            loss_dict = self.matched_loss(
                outputs=outputs,
                gt_masks=gt_masks,
                perturbed_masks=perturbed_masks,
                gate_weights=outputs.get("gate_weights"),
            )
            total_loss = loss_dict["total_loss"]

            # Gradient accumulation
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

            # Accumulate losses
            for k, v in loss_dict.items():
                val = v.item() if isinstance(v, torch.Tensor) else float(v)
                epoch_losses[k] = epoch_losses.get(k, 0.0) + val

            pbar.set_postfix(
                L=f"{total_loss.item():.4f}",
                lr=f"{lr:.2e}",
            )

        n = len(self.train_loader)
        return {k: v / n for k, v in epoch_losses.items()}

    # ── Full training ────────────────────────────────────────────────

    def run(self) -> Dict[str, float]:
        """Run full continuous training."""
        self._write_log(f"Starting training: {self.config.total_epochs} epochs")

        history: List[Dict] = []

        for epoch in range(1, self.config.total_epochs + 1):
            train_info = self.train_one_epoch(epoch)

            val_metrics = self.evaluate_fn(
                self.model, self.val_loader, device=self.device,
            )
            val_dice = val_metrics.get("dice", 0.0)
            val_miou = val_metrics.get("miou", 0.0)

            record = {
                "epoch": epoch,
                **train_info,
                "val_dice": val_dice,
                "val_miou": val_miou,
            }
            history.append(record)
            self._save_history(history)

            lr_now = self._get_lr(self.global_step)
            log_str = (
                f"[E{epoch:03d}] lr={lr_now:.2e} "
                f"| total={train_info.get('total_loss', 0):.4f} "
                f"| dice={val_dice:.4f} mIoU={val_miou:.4f}"
            )
            self._write_log(log_str)

            ckpt = {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "global_step": self.global_step,
                "val_dice": val_dice,
                "val_miou": val_miou,
            }
            torch.save(ckpt, os.path.join(self.save_dir, "latest_model.pth"))

            if val_dice > self.best_val_dice:
                self.best_val_dice = val_dice
                torch.save(ckpt, os.path.join(self.save_dir, "best_model.pth"))
                self._write_log(f"[NEW BEST] dice={val_dice:.4f}")

        self._write_log("\nTraining complete. Final evaluation...")
        final = self._final_evaluation()
        return final

    # ── Helpers ───────────────────────────────────────────────────────

    def _init_log_file(self) -> None:
        cfg = self.config
        log = (
            f"\n{'='*60}\n"
            f"UMPAv2 Method 2a — SAM3-style + Perm Aug\n"
            f"Save: {self.save_dir}\n"
            f"Epochs: {cfg.total_epochs}, Warmup: {cfg.warmup_epochs}\n"
            f"LR: {cfg.lr} → {cfg.min_lr} ({cfg.lr_schedule})\n"
            f"Perm aug: {cfg.enable_permutation_aug}\n"
            f"K: {cfg.K}, λ_con: {cfg.consistency_weight}\n"
            f"{'='*60}\n"
        )
        with open(self.log_file_path, "w", encoding="utf-8") as f:
            f.write(log)
        print(log)

    def _write_log(self, msg: str) -> None:
        print(msg)
        with open(self.log_file_path, "a", encoding="utf-8") as f:
            f.write(msg + "\n")

    def _save_history(self, history: List[Dict]) -> None:
        path = os.path.join(self.save_dir, "training_history.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2, default=str)

    def _final_evaluation(self) -> Dict[str, float]:
        best_path = os.path.join(self.save_dir, "best_model.pth")
        if not os.path.exists(best_path):
            self._write_log("No best_model.pth found")
            return {}

        ckpt = torch.load(best_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model_state_dict"])

        loader = self.test_loader or self.val_loader
        split = "test" if self.test_loader is not None else "val"
        self._write_log(f"Evaluating on {split}...")

        metrics = self.evaluate_fn(
            self.model, loader, device=self.device, full_metrics=True,
        )
        self._write_log(f"[FINAL {split.upper()}]")
        for k, v in metrics.items():
            self._write_log(f"  {k}: {v:.6f}" if isinstance(v, float) else f"  {k}: {v}")

        path = os.path.join(self.save_dir, "test_results.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, default=str)
        return metrics
