"""Ablation trainer for UMPT-SAM.

Extends ``UMPATrainer`` with ablation-specific controls:
- Prompt filtering (via AblationModelWrapper)
- MPCL control (K=0, lambda_con=0 when disabled)
- Mask prompt forwarding from batch
- Training history tracking (epoch-by-epoch JSON)
- Auto-evaluation on test set after training

This file is NEW and does NOT modify the existing trainer.py.
"""

from __future__ import annotations

import os
from copy import copy
from datetime import datetime
from typing import Optional

import torch
import torch.nn.functional as F
from tqdm import tqdm

from ..config.experiment_config import ExperimentConfig
from .ablation_results import AblationResultsManager, EpochRecord
from .trainer import UMPATrainer


class AblationTrainer(UMPATrainer):
    """Trainer subclass for ablation study experiments.

    Inherits all functionality from ``UMPATrainer`` and adds:
    1. Prompt filtering + mask forwarding in ``train_one_epoch``
    2. MPCL override (disable consistency loss)
    3. Epoch-by-epoch ``training_history.json``
    4. Final test evaluation with ``test_results.json``
    5. Experiment config snapshot saved to ``experiment_config.json``
    """

    def __init__(
        self,
        *args,
        experiment_config: ExperimentConfig,
        results_manager: Optional[AblationResultsManager] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.exp_cfg = experiment_config
        self.results_manager = results_manager or AblationResultsManager(
            base_dir=os.path.dirname(self.save_dir),
        )
        self.training_history: list[EpochRecord] = []

    # ------------------------------------------------------------------
    # Override training loop
    # ------------------------------------------------------------------

    def train_one_epoch(self, epoch, phase):
        """Override training loop with ablation controls.

        Changes vs parent:
        1. Load ``coarse_mask`` from batch for scenarios using mask prompt
        2. MPCL control: skip K-perturbation when disabled
        3. Prompt filtering is handled by AblationModelWrapper.forward()
        """
        self.model.train()

        epoch_total_loss = 0.0
        epoch_seg_loss = 0.0
        epoch_con_loss = 0.0

        # MPCL override: force lambda_con = 0 if disabled
        if not self.exp_cfg.enable_mpcl:
            self.composer_loss.config_loss['consistency_loss_weight'] = 0.0
        else:
            self.composer_loss.config_loss['consistency_loss_weight'] = phase.lambda_con

        phase_name = getattr(phase, 'name', f'Phase {epoch}')
        scenario = self.exp_cfg.name
        pbar = tqdm(
            self.train_loader,
            desc=f"Train Epoch {epoch} [{phase_name}] ({scenario})",
            leave=False,
        )

        for batch in pbar:
            images = batch['image'].to(self.device, non_blocking=True)
            gt_masks = batch['mask'].to(self.device, non_blocking=True)

            boxes = batch.get('bbox', None)
            points = batch.get('points', None)
            point_labels = batch.get('point_labels', None)
            captions = batch.get('text', None)
            # Load coarse_mask from batch (trainer gốc không làm điều này)
            masks_input = batch.get('coarse_mask', None)

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
                # AblationModelWrapper.forward() handles prompt filtering
                outputs = self.model(
                    image=images,
                    boxes=boxes,
                    points=points,
                    point_labels=point_labels,
                    masks=masks_input,
                    captions=captions,
                )
                pred_masks = outputs['pred_masks']

                if pred_masks.shape[-2:] != gt_masks.shape[-2:]:
                    pred_masks = F.interpolate(
                        pred_masks,
                        size=gt_masks.shape[-2:],
                        mode="bilinear",
                        align_corners=False,
                    )

                # K-perturbation for consistency loss
                perturbed_masks = None
                effective_K = self.exp_cfg.effective_K if self.exp_cfg.enable_mpcl else 0
                effective_lambda = self.exp_cfg.effective_lambda_con if self.exp_cfg.enable_mpcl else 0.0

                if effective_K > 0 and phase.lambda_con > 0.0 and effective_lambda > 0.0:
                    # Use wrapper's forward_k_perturbations
                    k_outputs = self.model.forward_k_perturbations(
                        image=images,
                        K=effective_K,
                        boxes=boxes,
                        points=points,
                        point_labels=point_labels,
                        masks=masks_input,
                        captions=captions,
                    )
                    p_masks_list = []
                    for out in k_outputs:
                        p_mask = out['pred_masks']
                        if p_mask.shape[-2:] != gt_masks.shape[-2:]:
                            p_mask = F.interpolate(
                                p_mask,
                                size=gt_masks.shape[-2:],
                                mode="bilinear",
                                align_corners=False,
                            )
                        p_masks_list.append(p_mask)
                    perturbed_masks = torch.stack(p_masks_list, dim=1)

                loss_dict = self.composer_loss(
                    pred_masks=pred_masks,
                    gt_masks=gt_masks,
                    perturbed_masks=perturbed_masks,
                )
                total_loss = loss_dict['total_loss']

            self.scaler.scale(total_loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            epoch_total_loss += total_loss.item()
            epoch_seg_loss += loss_dict['seg_loss'].item()
            con_loss_val = (
                loss_dict['consistency_loss'].item()
                if isinstance(loss_dict['consistency_loss'], torch.Tensor)
                else 0.0
            )
            epoch_con_loss += con_loss_val

            pbar.set_postfix({
                "L_tot": f"{total_loss.item():.4f}",
                "L_seg": f"{loss_dict['seg_loss'].item():.4f}",
                "L_con": f"{con_loss_val:.4f}",
            })

        num_batches = len(self.train_loader)
        return {
            "total_loss": epoch_total_loss / num_batches,
            "seg_loss": epoch_seg_loss / num_batches,
            "consistency_loss": epoch_con_loss / num_batches,
        }

    # ------------------------------------------------------------------
    # Override run() to add history tracking + final evaluation
    # ------------------------------------------------------------------

    def run(self):
        """Run full training with results tracking."""
        # Save experiment config snapshot
        self.exp_cfg.to_json(os.path.join(self.save_dir, "experiment_config.json"))

        self._write_log(
            f"[ABLATION] Kịch bản: {self.exp_cfg.name}\n"
            f"  Prompts: {self.exp_cfg.active_prompts}\n"
            f"  Components: {self.exp_cfg.active_components}\n"
            f"  K={self.exp_cfg.effective_K}, λ_con={self.exp_cfg.effective_lambda_con}"
        )
        self._write_log(
            f"BẮT ĐẦU HUẤN LUYỆN: {self.total_epochs} Epochs | "
            f"Lưu log tại: {self.log_file_path}"
        )

        for epoch in range(1, self.total_epochs + 1):
            phase = self.scheduler.get_current_phase(epoch)
            self.scheduler.apply_phase(
                model=self.model.model if hasattr(self.model, 'model') else self.model,
                epoch=epoch,
                optimizer=self.optimizer,
            )
            phase_name = getattr(phase, 'name', 'Phase')

            self._write_log(
                f"\n=== Epoch {epoch}/{self.total_epochs} | Phase: {phase_name} "
                f"| LR: {phase.lr} | lambda_con: {phase.lambda_con} ==="
            )

            train_info = self.train_one_epoch(epoch, phase)

            val_metrics = self.evaluate_fn(
                self.model, self.val_loader, device=self.device,
            )
            val_dice = val_metrics.get('dice', 0.0)
            val_miou = val_metrics.get('miou', 0.0)

            # Record epoch
            record = EpochRecord(
                epoch=epoch,
                phase=phase_name,
                lr=phase.lr,
                train_total_loss=train_info['total_loss'],
                train_seg_loss=train_info['seg_loss'],
                train_con_loss=train_info['consistency_loss'],
                val_dice=val_dice,
                val_miou=val_miou,
            )
            self.training_history.append(record)
            self.results_manager.save_training_history(
                self.save_dir, self.training_history,
            )

            log_str = (
                f"[LOG] | Epoch {epoch:03d} | {phase_name:10s} "
                f"| L_seg: {train_info['seg_loss']:.4f} "
                f"| L_con: {train_info['consistency_loss']:.4f} "
                f"| L_total: {train_info['total_loss']:.4f} "
                f"| val_dice: {val_dice:.4f} | val_miou: {val_miou:.4f}"
            )
            self._write_log(log_str)

            # Save checkpoints
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'val_dice': val_dice,
                'val_miou': val_miou,
                'scenario': self.exp_cfg.name,
            }
            torch.save(checkpoint, os.path.join(self.save_dir, "latest_model.pth"))

            if val_dice > self.best_val_dice:
                self.best_val_dice = val_dice
                torch.save(checkpoint, os.path.join(self.save_dir, "best_model.pth"))
                self._write_log(
                    f"[NEW BEST] val_dice: {self.best_val_dice:.4f}"
                )

        # Final evaluation
        self._write_log("\n" + "=" * 60)
        self._write_log("HUẤN LUYỆN HOÀN TẤT! ĐÁNH GIÁ CHUNG CUỘC...")
        self._run_final_evaluation()
        self._write_log("=" * 60 + "\n")

    # ------------------------------------------------------------------
    # Final evaluation
    # ------------------------------------------------------------------

    def _run_final_evaluation(self):
        """Load best checkpoint and evaluate on test set."""
        best_path = os.path.join(self.save_dir, "best_model.pth")
        if not os.path.exists(best_path):
            self._write_log("⚠️ Không tìm thấy best_model.pth")
            return

        checkpoint = torch.load(best_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        best_epoch = checkpoint.get('epoch', '?')
        self._write_log(f"Đã nạp trọng số tốt nhất từ Epoch {best_epoch}")

        # Evaluate on test set
        loader = self.test_loader if self.test_loader is not None else self.val_loader
        split_name = "test" if self.test_loader is not None else "val"
        self._write_log(f"ĐÁNH GIÁ TRÊN TẬP {split_name.upper()}...")

        final_metrics = self.evaluate_fn(self.model, loader, device=self.device, full_metrics=True)

        self._write_log(f"[KẾT QUẢ {split_name.upper()} — {self.exp_cfg.name}]")
        for key, val in final_metrics.items():
            if isinstance(val, float):
                self._write_log(f"   - {key}: {val:.6f}")
            else:
                self._write_log(f"   - {key}: {val}")

        # Save test results
        self.results_manager.save_test_results(
            save_dir=self.save_dir,
            metrics=final_metrics,
            scenario_name=self.exp_cfg.name,
        )
        self._write_log(f"✅ Kết quả đã lưu tại: {self.save_dir}/test_results.json")
