"""Train UMPA-SAM with E_sparse + E_fused + E_text (triple concat).

Architecture:
    Standard:      sparse_prompt_embs = cat([E_sparse, E_fused])
    This script:   sparse_prompt_embs = cat([E_sparse, E_fused, E_text])

Runs sequentially on all 5 polyp datasets with full logging.
Error-resilient: saves emergency checkpoint on crash.

Usage:
    python train_esparse_efused_etext.py
    python train_esparse_efused_etext.py --dry-run
    python train_esparse_efused_etext.py --dataset kvasir_seg
"""

import argparse
import gc
import json
import os
import sys
import time
import traceback
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from umpt_sam.config.model_config import UMPAModelConfig, UPFEConfig, MPPGConfig
from umpt_sam.config.train_config import TrainConfig, PhaseConfig
from umpt_sam.umpa_model import UMPAModel
from umpt_sam.losses.composer_loss import ComposerLoss
from umpt_sam.training.phase_scheduler import PhaseScheduler
from umpt_sam.training.evaluate import evaluate
from umpt_sam.data.polyp_dataset import PolypDataset, DatasetConfig, collate_fn
from umpt_sam.data.dataset_registry import get_dataset_config, list_datasets
from umpt_sam.data.polyp_transforms import POLYP_TRANSFORMS


# ======================================================================
# Model: E_sparse + E_fused + E_text (triple concat)
# ======================================================================

class ESparseEFusedETextModel(UMPAModel):
    """UMPAModel variant: Mask Decoder receives E_sparse + E_fused + E_text.

    Standard:   sparse_prompt_embeddings = cat([E_sparse, E_fused.unsqueeze(1)])
    This:       sparse_prompt_embeddings = cat([E_sparse, E_fused.unsqueeze(1), E_text])

    When E_text is None (no text prompt), falls back to standard behavior.
    """

    def forward(
        self,
        image: torch.Tensor,
        boxes: Optional[torch.Tensor] = None,
        points: Optional[torch.Tensor] = None,
        point_labels: Optional[torch.Tensor] = None,
        masks: Optional[torch.Tensor] = None,
        captions: Optional[List[str]] = None,
        text_embeddings: Optional[torch.Tensor] = None,
        multimask_output: bool = False,
        return_prompt_weights: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass: E_sparse + E_fused + E_text → mask decoder."""
        # --- Perturbation ---
        single_caption = captions[0] if captions and len(captions) == 1 else None
        perturbed = self.perturbation(
            bbox=boxes, points=points, point_labels=point_labels,
            mask=masks, text=single_caption, text_embeddings=text_embeddings,
        )

        boxes_p = perturbed.get("bbox", boxes)
        points_p = perturbed.get("points", points)
        point_labels_p = perturbed.get("point_labels", point_labels)
        masks_p = self._prepare_mask_prompt(perturbed.get("mask", masks))
        text_emb_p = perturbed.get("text_embedding", text_embeddings)

        if captions is None:
            captions_p = None
        elif len(captions) == 1 and "text" in perturbed:
            captions_p = [perturbed["text"]]
        else:
            captions_p = captions

        # --- Image encoder ---
        no_grad = not any(p.requires_grad for p in self.image_encoder.parameters())
        with torch.set_grad_enabled(not no_grad):
            backbone_out = self.image_encoder.forward_image(image)
        image_embeddings, high_res_features = self._select_sam_backbone_outputs(backbone_out)

        if high_res_features is not None and getattr(self.sam_mask_decoder, "use_high_res_features", False):
            conv_s0 = getattr(self.sam_mask_decoder, "conv_s0", None)
            conv_s1 = getattr(self.sam_mask_decoder, "conv_s1", None)
            if conv_s0 is not None and conv_s1 is not None:
                high_res_features = [conv_s0(high_res_features[0]), conv_s1(high_res_features[1])]

        image_pe = self.prompt_encoder.get_dense_pe()

        # --- Text encoder ---
        if captions_p is not None:
            text_out = self.image_encoder.forward_text(captions_p, device=image.device)
            text_emb_p = text_out["language_features"].permute(1, 0, 2)
        text_emb_p = self._project_text_embeddings(text_emb_p)

        # --- Prompt encoder ---
        point_input = (points_p, point_labels_p) if points_p is not None else None
        sparse_embs, dense_embs = self.prompt_encoder(
            points=point_input, boxes=boxes_p, masks=masks_p,
        )

        # --- UPFE fusion ---
        upfe_input = {
            "sparse_embeddings": sparse_embs,
            "mask_embeddings": dense_embs.flatten(2).permute(0, 2, 1),
            "text_embeddings": text_emb_p,
        }
        upfe_out = self.upfe_encoder(upfe_input, return_weights=return_prompt_weights)
        if return_prompt_weights:
            e_fused, prompt_weights = upfe_out
        else:
            e_fused = upfe_out
            prompt_weights = None

        # ============================================================
        # KEY: E_sparse + E_fused + E_text (triple concat)
        # Standard: cat([sparse_embs, e_fused.unsqueeze(1)])
        # Here:     cat([sparse_embs, e_fused.unsqueeze(1), E_text])
        # ============================================================
        parts = [sparse_embs, e_fused.unsqueeze(1)]

        if text_emb_p is not None:
            # text_emb_p: [B, N_text, D] — concat directly
            parts.append(text_emb_p)

        sparse_prompt_embeddings = torch.cat(parts, dim=1)

        # --- Mask decoder ---
        decoder_out = self.sam_mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_prompt_embeddings,
            dense_prompt_embeddings=dense_embs,
            multimask_output=multimask_output,
            repeat_image=False,
            high_res_features=high_res_features,
        )
        pred_masks, iou_predictions = decoder_out[:2]

        output = {"pred_masks": pred_masks, "iou_predictions": iou_predictions}
        if return_prompt_weights:
            output["prompt_weights"] = prompt_weights
        return output


# ======================================================================
# Trainer with error-resilient checkpoint saving
# ======================================================================

class FaultTolerantTrainer:
    """Trainer that saves emergency checkpoint on crash."""

    VARIANT = "esparse_efused_etext"

    def __init__(
        self, model, train_loader, val_loader, test_loader,
        optimizer, scheduler, train_config, composer_loss,
        evaluate_fn, save_dir, device, dataset_name,
    ):
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
        self.dataset_name = dataset_name

        self.best_val_dice = 0.0
        self.total_epochs = self.scheduler.total_epochs
        self.scaler = torch.amp.GradScaler(device, enabled=True)

        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.save_dir = os.path.join(save_dir, f"run_{run_id}")
        os.makedirs(self.save_dir, exist_ok=True)
        self.log_file_path = os.path.join(self.save_dir, "training_log.txt")
        self.history = []
        self._init_log_file()

    def _init_log_file(self):
        with open(self.log_file_path, "w", encoding="utf-8") as f:
            f.write(f"{'='*60}\n")
            f.write(f"UMPA-SAM Training — E_sparse + E_fused + E_text\n")
            f.write(f"Dataset: {self.dataset_name}\n")
            f.write(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total epochs: {self.total_epochs}\n")
            f.write(f"Save dir: {self.save_dir}\n")
            f.write(f"{'='*60}\n\n")

    def _log(self, msg):
        print(msg)
        with open(self.log_file_path, "a", encoding="utf-8") as f:
            f.write(msg + "\n")

    def _save_checkpoint(self, epoch, val_dice, val_miou, tag="latest"):
        ckpt = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "val_dice": val_dice,
            "val_miou": val_miou,
            "dataset": self.dataset_name,
            "variant": self.VARIANT,
        }
        path = os.path.join(self.save_dir, f"{tag}_model.pth")
        torch.save(ckpt, path)
        return path

    def _save_emergency_checkpoint(self, epoch, error):
        """Save model state when training crashes."""
        self._log(f"\n⚠️ EMERGENCY SAVE — crash at epoch {epoch}")
        self._log(f"   Error: {type(error).__name__}: {error}")
        try:
            ckpt = {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "dataset": self.dataset_name,
                "variant": self.VARIANT,
                "crash_error": f"{type(error).__name__}: {str(error)[:500]}",
                "crash_traceback": traceback.format_exc()[-2000:],
                "crash_time": datetime.now().isoformat(),
            }
            path = os.path.join(self.save_dir, "emergency_crash_model.pth")
            torch.save(ckpt, path)
            self._log(f"   ✅ Emergency checkpoint saved: {path}")

            # Also save error log
            error_path = os.path.join(self.save_dir, "error.log")
            with open(error_path, "w", encoding="utf-8") as f:
                f.write(f"Dataset: {self.dataset_name}\n")
                f.write(f"Variant: {self.VARIANT}\n")
                f.write(f"Epoch: {epoch}\n")
                f.write(f"Time: {datetime.now().isoformat()}\n")
                f.write(f"Error: {type(error).__name__}: {error}\n\n")
                f.write(traceback.format_exc())
            self._log(f"   📄 Error log: {error_path}")
        except Exception as save_err:
            self._log(f"   ❌ Failed to save emergency checkpoint: {save_err}")

    def _save_history(self):
        path = os.path.join(self.save_dir, "training_history.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.history, f, indent=2, ensure_ascii=False)

    def train_one_epoch(self, epoch, phase):
        self.model.train()
        epoch_total_loss = 0.0
        epoch_seg_loss = 0.0
        epoch_con_loss = 0.0

        self.composer_loss.config_loss['consistency_loss_weight'] = phase.lambda_con

        from tqdm import tqdm
        pbar = tqdm(self.train_loader, desc=f"[{self.dataset_name}] Epoch {epoch} [{phase.name}]", leave=False)

        for batch in pbar:
            images = batch['image'].to(self.device, non_blocking=True)
            gt_masks = batch['mask'].to(self.device, non_blocking=True)
            boxes = batch.get('bbox', None)
            points = batch.get('points', None)
            point_labels = batch.get('point_labels', None)
            captions = batch.get('text', None)

            if boxes is not None: boxes = boxes.to(self.device, non_blocking=True)
            if points is not None: points = points.to(self.device, non_blocking=True)
            if point_labels is not None: point_labels = point_labels.to(self.device, non_blocking=True)

            self.optimizer.zero_grad()

            with torch.amp.autocast(self.device, enabled=True):
                outputs = self.model(
                    image=images, boxes=boxes, points=points,
                    point_labels=point_labels, captions=captions,
                )
                pred_masks = outputs['pred_masks']
                if pred_masks.shape[-2:] != gt_masks.shape[-2:]:
                    pred_masks = F.interpolate(pred_masks, size=gt_masks.shape[-2:], mode="bilinear", align_corners=False)

                perturbed_masks = None
                if self.config.K > 0 and phase.lambda_con > 0.0:
                    k_outputs = self.model.forward_k_perturbations(
                        image=images, K=self.config.K,
                        boxes=boxes, points=points,
                        point_labels=point_labels, captions=captions,
                    )
                    p_list = []
                    for out in k_outputs:
                        p = out['pred_masks']
                        if p.shape[-2:] != gt_masks.shape[-2:]:
                            p = F.interpolate(p, size=gt_masks.shape[-2:], mode="bilinear", align_corners=False)
                        p_list.append(p)
                    perturbed_masks = torch.stack(p_list, dim=1)

                loss_dict = self.composer_loss(pred_masks=pred_masks, gt_masks=gt_masks, perturbed_masks=perturbed_masks)
                total_loss = loss_dict['total_loss']

            self.scaler.scale(total_loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            epoch_total_loss += total_loss.item()
            epoch_seg_loss += loss_dict['seg_loss'].item()
            con_val = loss_dict['consistency_loss'].item() if isinstance(loss_dict['consistency_loss'], torch.Tensor) else 0.0
            epoch_con_loss += con_val

            pbar.set_postfix({"L_tot": f"{total_loss.item():.4f}", "L_seg": f"{loss_dict['seg_loss'].item():.4f}"})

        n = len(self.train_loader)
        return {"total_loss": epoch_total_loss/n, "seg_loss": epoch_seg_loss/n, "consistency_loss": epoch_con_loss/n}

    def run(self):
        self._log(f"🚀 BẮT ĐẦU HUẤN LUYỆN: {self.total_epochs} epochs | {self.dataset_name}")
        current_epoch = 0

        try:
            for epoch in range(1, self.total_epochs + 1):
                current_epoch = epoch
                phase = self.scheduler.get_current_phase(epoch)
                self.scheduler.apply_phase(model=self.model, epoch=epoch, optimizer=self.optimizer)

                self._log(f"\n=== Epoch {epoch}/{self.total_epochs} | Phase: {phase.name} | LR: {phase.lr} | λ_con: {phase.lambda_con} ===")

                train_info = self.train_one_epoch(epoch, phase)

                val_metrics = self.evaluate_fn(self.model, self.val_loader, device=self.device)
                val_dice = val_metrics.get('dice', 0.0)
                val_miou = val_metrics.get('miou', 0.0)

                log_str = (
                    f"[LOG] Epoch {epoch:03d} | {phase.name:10s} "
                    f"| L_seg: {train_info['seg_loss']:.4f} "
                    f"| L_con: {train_info['consistency_loss']:.4f} "
                    f"| L_total: {train_info['total_loss']:.4f} "
                    f"| val_dice: {val_dice:.4f} | val_miou: {val_miou:.4f}"
                )
                self._log(log_str)

                # Save history
                self.history.append({
                    "epoch": epoch, "phase": phase.name,
                    "train_total_loss": train_info['total_loss'],
                    "train_seg_loss": train_info['seg_loss'],
                    "train_con_loss": train_info['consistency_loss'],
                    "val_dice": val_dice, "val_miou": val_miou,
                })
                self._save_history()

                # Save checkpoints
                self._save_checkpoint(epoch, val_dice, val_miou, tag="latest")
                if val_dice > self.best_val_dice:
                    self.best_val_dice = val_dice
                    self._save_checkpoint(epoch, val_dice, val_miou, tag="best")
                    self._log(f"[NEW BEST] val_dice: {self.best_val_dice:.4f}")

        except Exception as e:
            # ============================================================
            # EMERGENCY: Save model on crash
            # ============================================================
            self._save_emergency_checkpoint(current_epoch, e)
            raise  # Re-raise so caller can handle

        # --- Final evaluation ---
        self._log("\n" + "="*60)
        self._log("HUẤN LUYỆN HOÀN TẤT! ĐÁNH GIÁ CHUNG CUỘC...")

        best_path = os.path.join(self.save_dir, "best_model.pth")
        if os.path.exists(best_path):
            ckpt = torch.load(best_path, map_location=self.device, weights_only=False)
            self.model.load_state_dict(ckpt['model_state_dict'])
            self._log(f"Đã nạp best model từ epoch {ckpt.get('epoch', '?')}")

            loader = self.test_loader if self.test_loader is not None else self.val_loader
            split = "TEST" if self.test_loader is not None else "VAL"
            final = self.evaluate_fn(self.model, loader, device=self.device)
            self._log(f"[KẾT QUẢ {split}] Dice: {final.get('dice', 0):.6f} | mIoU: {final.get('miou', 0):.6f}")

            results_path = os.path.join(self.save_dir, "test_results.json")
            with open(results_path, "w", encoding="utf-8") as f:
                json.dump({"dataset": self.dataset_name, "variant": self.VARIANT, **final}, f, indent=2)

        self._log("="*60)


# ======================================================================
# Data helpers
# ======================================================================

def build_dataloaders(batch_size, dataset_name, dry_run=False):
    dataset_config = get_dataset_config(dataset_name)

    train_ds = PolypDataset(cfg=dataset_config, phase='train')
    train_ds.transform = POLYP_TRANSFORMS['train']
    val_ds = PolypDataset(cfg=dataset_config, phase='val')
    val_ds.transform = POLYP_TRANSFORMS['val']
    test_ds = PolypDataset(cfg=dataset_config, phase='test')
    test_ds.transform = POLYP_TRANSFORMS['val']

    if dry_run:
        from torch.utils.data import Subset
        train_ds = Subset(train_ds, range(min(4, len(train_ds))))
        val_ds = Subset(val_ds, range(min(4, len(val_ds))))
        test_ds = Subset(test_ds, range(min(4, len(test_ds))))

    kw = dict(batch_size=batch_size, collate_fn=collate_fn)
    train_loader = DataLoader(train_ds, shuffle=True, **kw)
    val_loader = DataLoader(val_ds, shuffle=False, **kw)
    test_loader = DataLoader(test_ds, shuffle=False, **kw)

    print(f"📦 [{dataset_name}] Train={len(train_ds)}, Val={len(val_ds)}, Test={len(test_ds)}")
    return train_loader, val_loader, test_loader


def build_model(sam_checkpoint, device):
    model_config = UMPAModelConfig(
        sam_checkpoint=sam_checkpoint,
        embed_dim=256,
        text_embed_dim=512,
        freeze_image_encoder=True,
        upfe=UPFEConfig(scoring_hidden_dim=256),
        mppg=MPPGConfig(),
    )
    # Build base UMPAModel then swap class to ESparseEFusedETextModel
    model = UMPAModel.from_config(model_config=model_config)
    model.__class__ = ESparseEFusedETextModel
    return model.to(device)


def clear_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


# ======================================================================
# Main
# ======================================================================

def train_one_dataset(dataset_name, args):
    """Train on a single dataset with error resilience."""
    print(f"\n{'█'*60}")
    print(f"█ E_sparse + E_fused + E_text — Dataset: {dataset_name}")
    print(f"{'█'*60}")

    train_config = TrainConfig()
    if args.dry_run:
        train_config = TrainConfig(
            batch_size=2, K=train_config.K,
            phase1=PhaseConfig(name="warmup", epochs=1, lambda_con=0.0,
                               freeze_image_encoder=True, freeze_prompt_encoder=True,
                               freeze_mask_decoder=True, lr=1e-4),
            phase2=PhaseConfig(name="adaptation", epochs=1, lambda_con=0.0,
                               freeze_image_encoder=True, freeze_prompt_encoder=False,
                               freeze_mask_decoder=True, lr=5e-5),
            phase3=PhaseConfig(name="consistency", epochs=1, lambda_con=0.5,
                               freeze_image_encoder=True, freeze_prompt_encoder=False,
                               freeze_mask_decoder=False, lr=1e-5),
        )

    bs = train_config.batch_size if not args.dry_run else 2
    train_loader, val_loader, test_loader = build_dataloaders(bs, dataset_name, args.dry_run)

    model = build_model(args.sam_checkpoint, args.device)
    composer_loss = ComposerLoss(config_loss=train_config.loss_weights).to(args.device)
    optimizer = optim.AdamW(model.parameters(), lr=train_config.phase1.lr, weight_decay=1e-4)
    scheduler = PhaseScheduler(train_config=train_config)

    save_dir = os.path.join(args.save_dir, dataset_name)
    trainer = FaultTolerantTrainer(
        model=model, train_loader=train_loader, val_loader=val_loader,
        test_loader=test_loader, optimizer=optimizer, scheduler=scheduler,
        train_config=train_config, composer_loss=composer_loss,
        evaluate_fn=evaluate, save_dir=save_dir, device=args.device,
        dataset_name=dataset_name,
    )
    trainer.run()
    return trainer


def main():
    parser = argparse.ArgumentParser(description="UMPA-SAM: Train with E_sparse + E_fused + E_text (5 datasets)")
    parser.add_argument("--dataset", type=str, choices=list_datasets(), help="Only this dataset")
    parser.add_argument("--save-dir", type=str, default="checkpoints/esparse_efused_etext")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--sam-checkpoint", type=str, default="model_trained/sam3.pt")
    parser.add_argument("--dry-run", action="store_true", help="Quick test: 1 epoch, 4 samples")
    args = parser.parse_args()

    datasets = [args.dataset] if args.dataset else list_datasets()
    results = []
    start_total = time.time()

    print(f"\n{'#'*60}")
    print(f"# UMPA-SAM — E_sparse + E_fused + E_text Training")
    print(f"# Datasets: {len(datasets)} — {', '.join(datasets)}")
    print(f"# Save dir: {os.path.abspath(args.save_dir)}")
    print(f"# Dry run: {args.dry_run}")
    print(f"{'#'*60}\n")

    for ds in datasets:
        start = time.time()
        record = {"dataset": ds, "status": "RUNNING", "elapsed_sec": 0, "error": None}
        try:
            train_one_dataset(ds, args)
            record["status"] = "OK"
            print(f"✅ {ds} — THÀNH CÔNG")
        except Exception as e:
            record["status"] = "FAILED"
            record["error"] = f"{type(e).__name__}: {str(e)[:300]}"
            print(f"❌ {ds} — LỖI: {record['error']}")
        finally:
            record["elapsed_sec"] = round(time.time() - start, 1)
            results.append(record)
            clear_gpu()

    elapsed = time.time() - start_total
    h, rem = divmod(int(elapsed), 3600)
    m, s = divmod(rem, 60)

    # Save overall report
    report = {
        "variant": "esparse_efused_etext",
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "total_elapsed_sec": round(elapsed, 1),
        "experiments": results,
    }
    os.makedirs(args.save_dir, exist_ok=True)
    report_path = os.path.join(args.save_dir, "experiment_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    ok = sum(1 for r in results if r["status"] == "OK")
    fail = sum(1 for r in results if r["status"] == "FAILED")
    print(f"\n📊 Tổng kết: {ok}/{len(results)} thành công, {fail}/{len(results)} lỗi")
    print(f"⏱️  Thời gian: {h}h {m}m {s}s")
    print(f"📄 Report: {report_path}")
    print(f"✅ HOÀN TẤT!")


if __name__ == "__main__":
    main()
