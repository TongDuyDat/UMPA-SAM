"""UMPAv2 Trainer — Method 1 (V1-style 3-phase).

Adapted from ``umpt_sam.training.trainer.UMPATrainer``.
Key changes vs v1:
  - Forward call matches ``UMPAv2Model.forward()`` (Pipeline A)
  - Tracks ``importance_reg_loss`` and ``gate_weights``
  - Uses ``ComposerV2Loss`` instead of ``ComposerLoss``
"""

from __future__ import annotations

import os
from datetime import datetime
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
from tqdm import tqdm

from .config_v2 import PhaseV2Config, TrainV2Config


class UMPAv2Trainer:
    """V1-style 3-phase trainer for UMPAv2."""

    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        test_loader,
        optimizer,
        scheduler,
        train_config: TrainV2Config,
        composer_loss,
        evaluate_fn,
        save_dir: str = "checkpoints",
        device: str = "cuda",
    ) -> None:
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config = train_config
        self.composer_loss = composer_loss
        self.evaluate_fn = evaluate_fn
        self.device = device

        self.best_val_dice = 0.0
        self.total_epochs = scheduler.total_epochs
        self.scaler = torch.amp.GradScaler(device, enabled=True)

        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.save_dir = os.path.join(save_dir, f"run_{run_id}")
        os.makedirs(self.save_dir, exist_ok=True)
        self.log_file_path = os.path.join(self.save_dir, "training_log.txt")
        self._init_log_file()

    # ── Logging ──────────────────────────────────────────────────────

    def _init_log_file(self) -> None:
        start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log = (
            f"\n{'='*60}\n"
            f"UMPAv2 Method 1 Training — {start_time}\n"
            f"Save dir: {self.save_dir}\n"
            f"Total epochs: {self.total_epochs}\n"
            f"Batch size: {self.config.batch_size}\n"
            f"K: {self.config.K}\n"
            f"Phase 1: {self.config.phase1.epochs}ep lr={self.config.phase1.lr}\n"
            f"Phase 2: {self.config.phase2.epochs}ep lr={self.config.phase2.lr}\n"
            f"Phase 3: {self.config.phase3.epochs}ep lr={self.config.phase3.lr} "
            f"λ_con={self.config.phase3.lambda_con}\n"
            f"{'='*60}\n"
        )
        with open(self.log_file_path, "a", encoding="utf-8") as f:
            f.write(log)
        print(log)

    def _write_log(self, message: str) -> None:
        print(message)
        with open(self.log_file_path, "a", encoding="utf-8") as f:
            f.write(message + "\n")

    # ── Train one epoch ──────────────────────────────────────────────

    def train_one_epoch(
        self, epoch: int, phase: PhaseV2Config,
    ) -> Dict[str, float]:
        """Run one training epoch."""
        self.model.train()

        epoch_total = 0.0
        epoch_seg = 0.0
        epoch_bce = 0.0
        epoch_con = 0.0
        epoch_reg = 0.0

        phase_name = phase.name or f"Phase {epoch}"
        pbar = tqdm(
            self.train_loader,
            desc=f"Train E{epoch} [{phase_name}]",
            leave=False,
        )

        for batch in pbar:
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

            self.optimizer.zero_grad()

            with torch.amp.autocast(self.device, enabled=True):
                outputs = self.model(
                    image=images,
                    boxes=boxes,
                    points=points,
                    point_labels=point_labels,
                    masks=masks_input,
                    captions=captions,
                    return_gate_weights=True,
                )
                pred_masks = outputs["pred_masks"]
                gate_weights = outputs.get("gate_weights")

                # Resize to GT size
                if pred_masks.shape[-2:] != gt_masks.shape[-2:]:
                    pred_masks = F.interpolate(
                        pred_masks,
                        size=gt_masks.shape[-2:],
                        mode="bilinear",
                        align_corners=False,
                    )

                # K-perturbation for consistency loss
                perturbed_masks = None
                if self.config.K > 0 and phase.lambda_con > 0.0:
                    k_outputs = self.model.forward_k_perturbations(
                        image=images,
                        K=self.config.K,
                        boxes=boxes,
                        points=points,
                        point_labels=point_labels,
                        masks=masks_input,
                        captions=captions,
                    )
                    p_list = []
                    for out in k_outputs:
                        pm = out["pred_masks"]
                        if pm.shape[-2:] != gt_masks.shape[-2:]:
                            pm = F.interpolate(
                                pm,
                                size=gt_masks.shape[-2:],
                                mode="bilinear",
                                align_corners=False,
                            )
                        p_list.append(pm)
                    perturbed_masks = torch.stack(p_list, dim=1)

                loss_dict = self.composer_loss(
                    pred_masks=pred_masks,
                    gt_masks=gt_masks,
                    perturbed_masks=perturbed_masks,
                    gate_weights=gate_weights,
                )
                total_loss = loss_dict["total_loss"]

            self.scaler.scale(total_loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            epoch_total += total_loss.item()
            epoch_seg += loss_dict["seg_loss"].item()
            epoch_bce += loss_dict["bce_loss"].item()
            con_val = (
                loss_dict["consistency_loss"].item()
                if isinstance(loss_dict["consistency_loss"], torch.Tensor)
                else 0.0
            )
            epoch_con += con_val
            reg_val = (
                loss_dict["importance_reg_loss"].item()
                if isinstance(loss_dict["importance_reg_loss"], torch.Tensor)
                else 0.0
            )
            epoch_reg += reg_val

            pbar.set_postfix(
                L=f"{total_loss.item():.4f}",
                seg=f"{loss_dict['seg_loss'].item():.4f}",
                con=f"{con_val:.4f}",
            )

        n = len(self.train_loader)
        return {
            "total_loss": epoch_total / n,
            "seg_loss": epoch_seg / n,
            "bce_loss": epoch_bce / n,
            "consistency_loss": epoch_con / n,
            "importance_reg_loss": epoch_reg / n,
        }

    # ── Full training run ────────────────────────────────────────────

    def run(self) -> Dict[str, float]:
        """Run full 3-phase training."""
        self._write_log(
            f"Starting training: {self.total_epochs} epochs"
        )

        training_history: List[Dict] = []

        for epoch in range(1, self.total_epochs + 1):
            phase = self.scheduler.get_current_phase(epoch)
            self.scheduler.apply_phase(
                model=self.model, epoch=epoch, optimizer=self.optimizer,
            )
            phase_name = phase.name or "Phase"

            self._write_log(
                f"\n=== Epoch {epoch}/{self.total_epochs} | {phase_name} "
                f"| LR={phase.lr:.2e} | λ_con={phase.lambda_con} ==="
            )

            train_info = self.train_one_epoch(epoch, phase)

            # Validation
            val_metrics = self.evaluate_fn(
                self.model, self.val_loader, device=self.device,
            )
            val_dice = val_metrics.get("dice", 0.0)
            val_miou = val_metrics.get("miou", 0.0)

            # Record
            record = {
                "epoch": epoch,
                "phase": phase_name,
                "lr": phase.lr,
                **train_info,
                "val_dice": val_dice,
                "val_miou": val_miou,
            }
            training_history.append(record)
            self._save_history(training_history)

            log_str = (
                f"[E{epoch:03d}] {phase_name:10s} "
                f"| seg={train_info['seg_loss']:.4f} "
                f"| con={train_info['consistency_loss']:.4f} "
                f"| reg={train_info['importance_reg_loss']:.4f} "
                f"| total={train_info['total_loss']:.4f} "
                f"| dice={val_dice:.4f} mIoU={val_miou:.4f}"
            )
            self._write_log(log_str)

            # Checkpoints
            ckpt = {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "val_dice": val_dice,
                "val_miou": val_miou,
            }
            torch.save(ckpt, os.path.join(self.save_dir, "latest_model.pth"))

            if val_dice > self.best_val_dice:
                self.best_val_dice = val_dice
                torch.save(ckpt, os.path.join(self.save_dir, "best_model.pth"))
                self._write_log(f"[NEW BEST] dice={val_dice:.4f}")

        # Final evaluation
        self._write_log("\n" + "=" * 60)
        self._write_log("Training complete. Running final evaluation...")
        final = self._final_evaluation()
        self._write_log("=" * 60)

        return final

    # ── Helpers ───────────────────────────────────────────────────────

    def _save_history(self, history: List[Dict]) -> None:
        import json

        path = os.path.join(self.save_dir, "training_history.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2, ensure_ascii=False)

    def _final_evaluation(self) -> Dict[str, float]:
        best_path = os.path.join(self.save_dir, "best_model.pth")
        if not os.path.exists(best_path):
            self._write_log("⚠️ No best_model.pth found")
            return {}

        ckpt = torch.load(best_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self._write_log(f"Loaded best weights from epoch {ckpt.get('epoch', '?')}")

        loader = self.test_loader or self.val_loader
        split = "test" if self.test_loader is not None else "val"
        self._write_log(f"Evaluating on {split} set...")

        metrics = self.evaluate_fn(
            self.model, loader, device=self.device, full_metrics=True,
        )

        self._write_log(f"[FINAL {split.upper()}]")
        for k, v in metrics.items():
            self._write_log(f"  {k}: {v:.6f}" if isinstance(v, float) else f"  {k}: {v}")

        import json

        path = os.path.join(self.save_dir, "test_results.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, default=str)

        return metrics
