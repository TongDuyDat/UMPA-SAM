"""Train UMPA-SAM on all 5 polyp benchmark datasets with a single command.

Usage
-----
    # Train all 5 datasets sequentially (default config)
    python train_all.py

    # Train specific datasets only
    python train_all.py --datasets kvasir_seg cvc_clinicdb

    # Custom batch size + checkpoint
    python train_all.py --batch-size 4 --sam-checkpoint model_trained/sam3.pt

    # Dry-run (1 epoch per phase, 4 samples) to verify setup
    python train_all.py --dry-run

    # List available datasets
    python train_all.py --list
"""

import argparse
import csv
import gc
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
import torch.optim as optim
from torch.utils.data import DataLoader

# Ensure project root is in path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from umpt_sam.config.model_config import UMPAModelConfig, UPFEConfig, MPPGConfig
from umpt_sam.config.train_config import TrainConfig, PhaseConfig
from umpt_sam.umpa_model import UMPAModel
from umpt_sam.losses.composer_loss import ComposerLoss
from umpt_sam.training.phase_scheduler import PhaseScheduler
from umpt_sam.training.trainer import UMPATrainer
from umpt_sam.training.evaluate import evaluate
from umpt_sam.data.polyp_dataset import PolypDataset, DatasetConfig, collate_fn
from umpt_sam.data.dataset_registry import get_dataset_config, list_datasets
from umpt_sam.data.polyp_transforms import POLYP_TRANSFORMS
from umpt_sam.common import EpochRecord, ResultsManager


# ═══════════════════════════════════════════════════════════════════════
# Data helpers
# ═══════════════════════════════════════════════════════════════════════

def build_dataloaders(dataset_name: str, batch_size: int, num_workers: int = 2,
                      dry_run: bool = False):
    """Build train/val/test dataloaders for a named dataset."""
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

    loader_kwargs = dict(
        batch_size=batch_size,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True,
    )
    train_loader = DataLoader(train_ds, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_ds, shuffle=False, **loader_kwargs)

    _len = lambda d: len(d.dataset) if hasattr(d, 'dataset') else len(d)
    print(f"📦 [{dataset_name}] Train={_len(train_ds)}, Val={_len(val_ds)}, Test={_len(test_ds)}")

    return train_loader, val_loader, test_loader


# ═══════════════════════════════════════════════════════════════════════
# Config builders
# ═══════════════════════════════════════════════════════════════════════

def build_train_config(args, dry_run: bool = False) -> TrainConfig:
    """Build TrainConfig from CLI args."""
    if dry_run:
        return TrainConfig(
            batch_size=2, K=args.K,
            phase1=PhaseConfig(name="warmup", epochs=1, lr=args.lr1,
                               lambda_con=0.0, freeze_image_encoder=True,
                               freeze_prompt_encoder=True, freeze_mask_decoder=True),
            phase2=PhaseConfig(name="adaptation", epochs=1, lr=args.lr2,
                               lambda_con=0.0, freeze_image_encoder=True,
                               freeze_prompt_encoder=False, freeze_mask_decoder=True),
            phase3=PhaseConfig(name="consistency", epochs=1, lr=args.lr3,
                               lambda_con=args.lambda_con, freeze_image_encoder=True,
                               freeze_prompt_encoder=False, freeze_mask_decoder=False),
        )
    return TrainConfig(
        batch_size=args.batch_size, K=args.K,
        phase1=PhaseConfig(name="warmup", epochs=args.epochs1, lr=args.lr1,
                           lambda_con=0.0, freeze_image_encoder=True,
                           freeze_prompt_encoder=True, freeze_mask_decoder=True),
        phase2=PhaseConfig(name="adaptation", epochs=args.epochs2, lr=args.lr2,
                           lambda_con=0.0, freeze_image_encoder=True,
                           freeze_prompt_encoder=False, freeze_mask_decoder=True),
        phase3=PhaseConfig(name="consistency", epochs=args.epochs3, lr=args.lr3,
                           lambda_con=args.lambda_con, freeze_image_encoder=True,
                           freeze_prompt_encoder=False, freeze_mask_decoder=False),
    )


def build_model(args, device: str) -> UMPAModel:
    """Build UMPAModel from CLI args."""
    model_config = UMPAModelConfig(
        sam_checkpoint=args.sam_checkpoint,
        embed_dim=args.embed_dim,
        text_embed_dim=args.text_embed_dim,
        freeze_image_encoder=True,
        upfe=UPFEConfig(scoring_hidden_dim=args.embed_dim),
        mppg=MPPGConfig(),
    )
    return UMPAModel.from_config(model_config=model_config).to(device)


# ═══════════════════════════════════════════════════════════════════════
# Per-dataset training (follows train_ablation.py pattern)
# ═══════════════════════════════════════════════════════════════════════

def train_single_dataset(
    dataset_name: str,
    save_base_dir: str = "checkpoints/tmf_umpa",
    device: str = "cuda",
    dry_run: bool = False,
    sam_checkpoint: str = "model_trained/sam3.pt",
    args=None,
):
    """Train on a single dataset with full logging.

    Follows the same pattern as ``train_ablation.py:train_scenario()``:
    - Saves ``training_history.json`` (epoch-by-epoch metrics)
    - Saves ``test_results.json`` (final evaluation on test set)
    - Saves ``training_log.txt`` (readable log file)
    - Saves ``run_config.json`` (hyperparameters snapshot)
    """
    print(f"\n{'='*60}")
    print(f"🚀 TRAINING: {dataset_name}")
    print(f"{'='*60}\n")

    t0 = time.time()

    # 1. Config
    train_config = build_train_config(args, dry_run=dry_run)
    batch_size = train_config.batch_size

    # 2. Data
    train_loader, val_loader, test_loader = build_dataloaders(
        dataset_name, batch_size, num_workers=args.num_workers, dry_run=dry_run,
    )

    # 3. Model (fresh for each dataset — no weight leakage)
    model = build_model(args, device)

    total_p = sum(p.numel() for p in model.parameters())
    trainable_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"📊 Params: {total_p/1e6:.1f}M total, {trainable_p/1e6:.1f}M trainable")

    # 4. Loss + Optimizer + Scheduler
    composer_loss = ComposerLoss(config_loss=train_config.loss_weights).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=train_config.phase1.lr, weight_decay=1e-4)
    scheduler = PhaseScheduler(train_config=train_config)

    # 5. Save directory (per-dataset)
    dataset_save_dir = os.path.join(save_base_dir, dataset_name)

    # 6. Trainer (UMPATrainer creates run_YYYYMMDD_HHMMSS subdir)
    trainer = UMPATrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        train_config=train_config,
        composer_loss=composer_loss,
        evaluate_fn=evaluate,
        save_dir=dataset_save_dir,
        device=device,
    )

    # 7. Save run config snapshot
    run_config = {
        "dataset": dataset_name,
        "sam_checkpoint": sam_checkpoint,
        "embed_dim": args.embed_dim,
        "text_embed_dim": args.text_embed_dim,
        "batch_size": batch_size,
        "K": train_config.K,
        "total_epochs": train_config.total_epochs,
        "phase1": train_config.phase1.to_dict(),
        "phase2": train_config.phase2.to_dict(),
        "phase3": train_config.phase3.to_dict(),
        "dry_run": dry_run,
        "device": device,
        "timestamp": datetime.now().isoformat(),
    }
    config_path = os.path.join(trainer.save_dir, "run_config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(run_config, f, indent=2, ensure_ascii=False)

    # 8. Run training with epoch-by-epoch logging
    training_history: list[EpochRecord] = []

    trainer._write_log(
        f"[FULL TRAIN] Dataset: {dataset_name}\n"
        f"  Batch size: {batch_size}\n"
        f"  K: {train_config.K}\n"
        f"  Total epochs: {train_config.total_epochs}\n"
        f"  Phase1: {train_config.phase1.epochs}ep, lr={train_config.phase1.lr}\n"
        f"  Phase2: {train_config.phase2.epochs}ep, lr={train_config.phase2.lr}\n"
        f"  Phase3: {train_config.phase3.epochs}ep, lr={train_config.phase3.lr}, λ_con={train_config.phase3.lambda_con}"
    )

    for epoch in range(1, trainer.total_epochs + 1):
        phase = scheduler.get_current_phase(epoch)
        scheduler.apply_phase(model=model, epoch=epoch, optimizer=optimizer)
        phase_name = getattr(phase, 'name', 'Phase')

        trainer._write_log(
            f"\n=== Epoch {epoch}/{trainer.total_epochs} | Phase: {phase_name} "
            f"| LR: {phase.lr} | lambda_con: {phase.lambda_con} ==="
        )

        train_info = trainer.train_one_epoch(epoch, phase)

        val_metrics = evaluate(model, val_loader, device=device)
        val_dice = val_metrics.get('dice', 0.0)
        val_miou = val_metrics.get('miou', 0.0)

        # Record epoch (same as ablation_trainer)
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
        training_history.append(record)
        ResultsManager.save_training_history(trainer.save_dir, training_history)

        log_str = (
            f"[LOG] | Epoch {epoch:03d} | {phase_name:10s} "
            f"| L_seg: {train_info['seg_loss']:.4f} "
            f"| L_con: {train_info['consistency_loss']:.4f} "
            f"| L_total: {train_info['total_loss']:.4f} "
            f"| val_dice: {val_dice:.4f} | val_miou: {val_miou:.4f}"
        )
        trainer._write_log(log_str)

        # Save checkpoints
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_dice': val_dice,
            'val_miou': val_miou,
            'dataset': dataset_name,
        }
        torch.save(checkpoint, os.path.join(trainer.save_dir, "latest_model.pth"))

        if val_dice > trainer.best_val_dice:
            trainer.best_val_dice = val_dice
            torch.save(checkpoint, os.path.join(trainer.save_dir, "best_model.pth"))
            trainer._write_log(f"[NEW BEST] val_dice: {trainer.best_val_dice:.4f}")

    # 9. Final evaluation on test set (same as ablation_trainer._run_final_evaluation)
    trainer._write_log("\n" + "=" * 60)
    trainer._write_log("HUẤN LUYỆN HOÀN TẤT! ĐÁNH GIÁ CHUNG CUỘC...")

    best_path = os.path.join(trainer.save_dir, "best_model.pth")
    if os.path.exists(best_path):
        ckpt = torch.load(best_path, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        best_epoch = ckpt.get('epoch', '?')
        trainer._write_log(f"Đã nạp trọng số tốt nhất từ Epoch {best_epoch}")

        loader = test_loader if test_loader is not None else val_loader
        split_name = "test" if test_loader is not None else "val"
        trainer._write_log(f"ĐÁNH GIÁ TRÊN TẬP {split_name.upper()}...")

        final_metrics = evaluate(model, loader, device=device, full_metrics=True)

        trainer._write_log(f"[KẾT QUẢ {split_name.upper()} — {dataset_name}]")
        for key, val in final_metrics.items():
            if isinstance(val, float):
                trainer._write_log(f"   - {key}: {val:.6f}")
            else:
                trainer._write_log(f"   - {key}: {val}")

        # Save test results JSON
        ResultsManager.save_test_results(
            save_dir=trainer.save_dir,
            metrics=final_metrics,
            name=dataset_name,
        )
        trainer._write_log(f"✅ Kết quả đã lưu tại: {trainer.save_dir}/test_results.json")
    else:
        trainer._write_log("⚠️ Không tìm thấy best_model.pth để đánh giá.")
        final_metrics = {}

    trainer._write_log("=" * 60 + "\n")

    elapsed = time.time() - t0
    print(f"\n✅ {dataset_name} done | Best Dice: {trainer.best_val_dice:.4f} | {elapsed:.0f}s")

    return {
        'status': 'OK',
        'best_dice': trainer.best_val_dice,
        'test_metrics': final_metrics,
        'save_dir': trainer.save_dir,
        'elapsed': elapsed,
    }


# ═══════════════════════════════════════════════════════════════════════
# Summary helpers
# ═══════════════════════════════════════════════════════════════════════

def print_summary_table(results: dict):
    """Print a formatted results table to console."""
    print(f"\n{'='*90}")
    print(f"{'DATASET':<20s} | {'STATUS':<8s} | {'DICE':>10s} | {'mIoU':>10s} | "
          f"{'PREC':>10s} | {'REC':>10s} | {'TIME':>10s}")
    print(f"{'-'*90}")
    for ds_name, info in results.items():
        status = info['status']
        if status == 'OK':
            tm = info.get('test_metrics', {})
            dice = f"{tm.get('dice', info.get('best_dice', 0)):.4f}"
            miou = f"{tm.get('miou', 0):.4f}"
            prec = f"{tm.get('precision', 0):.4f}"
            rec = f"{tm.get('recall', 0):.4f}"
            m, s = divmod(int(info.get('elapsed', 0)), 60)
            h, m = divmod(m, 60)
            time_str = f"{h}h{m:02d}m{s:02d}s"
        else:
            dice = miou = prec = rec = "—"
            time_str = "—"
        print(f"{ds_name:<20s} | {status:<8s} | {dice:>10s} | {miou:>10s} | "
              f"{prec:>10s} | {rec:>10s} | {time_str:>10s}")
    print(f"{'='*90}")


def save_global_summary_csv(results: dict, save_dir: str):
    """Save cross-dataset summary as CSV."""
    csv_path = os.path.join(save_dir, "benchmark_summary.csv")
    cols = ["dataset", "status", "dice", "miou", "precision", "recall", "f2", "mask_ap", "elapsed_s"]

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(cols)
        for ds_name, info in results.items():
            if info['status'] == 'OK':
                tm = info.get('test_metrics', {})
                writer.writerow([
                    ds_name, "OK",
                    f"{tm.get('dice', 0):.6f}",
                    f"{tm.get('miou', 0):.6f}",
                    f"{tm.get('precision', 0):.6f}",
                    f"{tm.get('recall', 0):.6f}",
                    f"{tm.get('f2', 0):.6f}",
                    f"{tm.get('mask_ap', 0):.6f}",
                    f"{info.get('elapsed', 0):.1f}",
                ])
            else:
                writer.writerow([ds_name, "FAILED"] + [""] * 7)

    print(f"📊 Summary CSV: {csv_path}")
    return csv_path


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="UMPA-SAM Full Benchmark Training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"Available datasets: {', '.join(list_datasets())}",
    )

    group = p.add_mutually_exclusive_group()
    group.add_argument("--datasets", nargs="+", default=None,
                       choices=list_datasets(),
                       help="Specific datasets to train (default: all 5)")
    group.add_argument("--list", action="store_true",
                       help="List all available datasets and exit")

    # Paths
    p.add_argument("--sam-checkpoint", type=str, default="model_trained/sam3.pt",
                   help="Path to SAM3 pre-trained checkpoint")
    p.add_argument("--save-dir", type=str, default="checkpoints/tmf_umpa",
                   help="Base directory for saving results")

    # Model
    p.add_argument("--embed-dim", type=int, default=256)
    p.add_argument("--text-embed-dim", type=int, default=512)

    # Training schedule
    p.add_argument("--batch-size", type=int, default=4,
                   help="Batch size (4 for T4 15GB, 16+ for A100)")
    p.add_argument("--K", type=int, default=3,
                   help="Number of perturbations for consistency loss")
    p.add_argument("--epochs1", type=int, default=5, help="Phase 1 (warmup) epochs")
    p.add_argument("--epochs2", type=int, default=5, help="Phase 2 (adaptation) epochs")
    p.add_argument("--epochs3", type=int, default=10, help="Phase 3 (consistency) epochs")
    p.add_argument("--lr1", type=float, default=1e-4, help="Phase 1 learning rate")
    p.add_argument("--lr2", type=float, default=5e-5, help="Phase 2 learning rate")
    p.add_argument("--lr3", type=float, default=1e-5, help="Phase 3 learning rate")
    p.add_argument("--lambda-con", type=float, default=0.5,
                   help="Consistency loss weight in phase 3")

    # Runtime
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--dry-run", action="store_true",
                   help="Quick test: 1 epoch/phase, 4 samples per dataset")

    return p.parse_args()


def main():
    args = parse_args()

    if args.list:
        print("📋 Available datasets:\n")
        for name in list_datasets():
            ds_cfg = get_dataset_config(name)
            print(f"  {name:20s} | root: {ds_cfg.root}")
        return

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("⚠️  CUDA not available, falling back to CPU")
        device = "cpu"

    datasets = args.datasets or list_datasets()
    total_epochs = args.epochs1 + args.epochs2 + args.epochs3
    if args.dry_run:
        total_epochs = 3

    print(f"""
╔══════════════════════════════════════════════════════════════╗
║             UMPA-SAM FULL BENCHMARK TRAINING                ║
╠══════════════════════════════════════════════════════════════╣
║  Datasets:     {', '.join(datasets):<45s}║
║  Epochs/ds:    {total_epochs:<45d}║
║  Batch size:   {args.batch_size:<45d}║
║  Device:       {device:<45s}║
║  Save dir:     {args.save_dir:<45s}║
║  Dry-run:      {str(args.dry_run):<45s}║
╚══════════════════════════════════════════════════════════════╝
""")

    os.makedirs(args.save_dir, exist_ok=True)

    # Train each dataset sequentially
    all_results = {}
    total_start = time.time()

    for i, ds_name in enumerate(datasets, 1):
        print(f"\n{'#'*60}")
        print(f"# [{i}/{len(datasets)}] Dataset: {ds_name}")
        print(f"{'#'*60}")

        try:
            result = train_single_dataset(
                dataset_name=ds_name,
                save_base_dir=args.save_dir,
                device=device,
                dry_run=args.dry_run,
                sam_checkpoint=args.sam_checkpoint,
                args=args,
            )
        except Exception as e:
            result = {
                'status': 'FAILED',
                'error': str(e),
                'elapsed': 0,
            }
            print(f"\n❌ {ds_name} FAILED: {e}")
            import traceback
            traceback.print_exc()
        finally:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        all_results[ds_name] = result

        # Save intermediate global results
        interim_path = os.path.join(args.save_dir, "all_results.json")
        with open(interim_path, "w", encoding="utf-8") as f:
            # Convert non-serializable values
            serializable = {}
            for k, v in all_results.items():
                serializable[k] = {
                    sk: sv for sk, sv in v.items()
                    if not isinstance(sv, torch.Tensor)
                }
            json.dump(serializable, f, indent=2, ensure_ascii=False, default=str)

    # Final summary
    total_elapsed = time.time() - total_start
    h, rem = divmod(int(total_elapsed), 3600)
    m, s = divmod(rem, 60)
    print(f"\n⏱️  Total time: {h}h {m:02d}m {s:02d}s")

    print_summary_table(all_results)
    save_global_summary_csv(all_results, args.save_dir)

    # Exit code
    failed = [k for k, v in all_results.items() if v['status'] != 'OK']
    if failed:
        print(f"\n⚠️  {len(failed)} dataset(s) failed: {', '.join(failed)}")
        sys.exit(1)
    else:
        print("\n🎉 All datasets trained successfully!")


if __name__ == "__main__":
    main()
