"""Train UMPAv2 — Method 1: V1-style 3-phase training.

Usage::

    # Train all 5 datasets
    python train_v2_method1.py --sam-checkpoint model_trained/sam3.pt

    # Train specific datasets
    python train_v2_method1.py --datasets kvasir_seg cvc_clinicdb

    # Dry-run
    python train_v2_method1.py --dry-run

    # Custom epochs
    python train_v2_method1.py --epochs1 10 --epochs2 10 --epochs3 20
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from datetime import datetime

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from umpa_samv2.config import UMPAv2ModelConfig, LossConfig
from umpa_samv2.model import UMPAv2Model
from umpa_samv2.losses import ComposerV2Loss

from umpa_samv2.training.config_v2 import PhaseV2Config, TrainV2Config
from umpa_samv2.training.phase_scheduler_v2 import PhaseSchedulerV2
from umpa_samv2.training.trainer_v2 import UMPAv2Trainer
from umpa_samv2.training.evaluate_v2 import evaluate

from umpt_sam.data.polyp_dataset import PolypDataset, collate_fn
from umpt_sam.data.dataset_registry import get_dataset_config, list_datasets
from umpt_sam.data.polyp_transforms import POLYP_TRANSFORMS


# ═══════════════════════════════════════════════════════════════════════
# Data
# ═══════════════════════════════════════════════════════════════════════

def build_dataloaders(
    dataset_name: str,
    batch_size: int,
    num_workers: int = 2,
    dry_run: bool = False,
):
    ds_cfg = get_dataset_config(dataset_name)

    train_ds = PolypDataset(cfg=ds_cfg, phase="train")
    train_ds.transform = POLYP_TRANSFORMS["train"]

    val_ds = PolypDataset(cfg=ds_cfg, phase="val")
    val_ds.transform = POLYP_TRANSFORMS["val"]

    test_ds = PolypDataset(cfg=ds_cfg, phase="test")
    test_ds.transform = POLYP_TRANSFORMS["val"]

    if dry_run:
        from torch.utils.data import Subset

        train_ds = Subset(train_ds, range(min(4, len(train_ds))))
        val_ds = Subset(val_ds, range(min(4, len(val_ds))))
        test_ds = Subset(test_ds, range(min(4, len(test_ds))))

    kw = dict(
        batch_size=batch_size,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True,
    )
    train_loader = DataLoader(train_ds, shuffle=True, **kw)
    val_loader = DataLoader(val_ds, shuffle=False, **kw)
    test_loader = DataLoader(test_ds, shuffle=False, **kw)

    _len = lambda d: len(d.dataset) if hasattr(d, "dataset") else len(d)
    print(
        f"📦 [{dataset_name}] "
        f"Train={_len(train_ds)} Val={_len(val_ds)} Test={_len(test_ds)}"
    )
    return train_loader, val_loader, test_loader


# ═══════════════════════════════════════════════════════════════════════
# Config builders
# ═══════════════════════════════════════════════════════════════════════

def build_train_config(args, dry_run: bool = False) -> TrainV2Config:
    if dry_run:
        return TrainV2Config(
            batch_size=2,
            K=args.K,
            phase1=PhaseV2Config(name="warmup", epochs=1, lr=args.lr1),
            phase2=PhaseV2Config(
                name="adaptation", epochs=1, lr=args.lr2,
                freeze_transformer_encoder=False,
                freeze_transformer_decoder=False,
                freeze_segmentation_head=False,
                freeze_dot_prod_scoring=False,
            ),
            phase3=PhaseV2Config(
                name="consistency", epochs=1, lr=args.lr3,
                lambda_con=args.lambda_con,
                freeze_transformer_encoder=False,
                freeze_transformer_decoder=False,
                freeze_segmentation_head=False,
                freeze_dot_prod_scoring=False,
            ),
            consistency_weight=args.lambda_con,
            importance_reg_weight=args.lambda_reg,
        )
    return TrainV2Config(
        batch_size=args.batch_size,
        K=args.K,
        phase1=PhaseV2Config(
            name="warmup", epochs=args.epochs1, lr=args.lr1,
        ),
        phase2=PhaseV2Config(
            name="adaptation", epochs=args.epochs2, lr=args.lr2,
            freeze_transformer_encoder=False,
            freeze_transformer_decoder=False,
            freeze_segmentation_head=False,
            freeze_dot_prod_scoring=False,
        ),
        phase3=PhaseV2Config(
            name="consistency", epochs=args.epochs3, lr=args.lr3,
            lambda_con=args.lambda_con,
            freeze_transformer_encoder=False,
            freeze_transformer_decoder=False,
            freeze_segmentation_head=False,
            freeze_dot_prod_scoring=False,
        ),
        consistency_weight=args.lambda_con,
        importance_reg_weight=args.lambda_reg,
    )


def build_model(args, device: str) -> UMPAv2Model:
    config = UMPAv2ModelConfig(
        sam_checkpoint=args.sam_checkpoint,
        embed_dim=args.embed_dim,
        text_embed_dim=args.text_embed_dim,
    )
    return UMPAv2Model.from_config(config).to(device)


# ═══════════════════════════════════════════════════════════════════════
# Per-dataset training
# ═══════════════════════════════════════════════════════════════════════

def train_single_dataset(
    dataset_name: str,
    save_base_dir: str,
    device: str,
    dry_run: bool,
    args,
):
    print(f"\n{'='*60}")
    print(f"🚀 TRAINING: {dataset_name} (Method 1 — V1-style)")
    print(f"{'='*60}\n")

    t0 = time.time()

    train_config = build_train_config(args, dry_run=dry_run)
    train_loader, val_loader, test_loader = build_dataloaders(
        dataset_name, train_config.batch_size,
        num_workers=args.num_workers, dry_run=dry_run,
    )

    model = build_model(args, device)
    total_p = sum(p.numel() for p in model.parameters())
    trainable_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"📊 Params: {total_p/1e6:.1f}M total, {trainable_p/1e6:.1f}M trainable")

    loss_cfg = LossConfig(
        dice_weight=train_config.dice_weight,
        bce_weight=train_config.bce_weight,
        consistency_weight=train_config.consistency_weight,
        importance_reg_weight=train_config.importance_reg_weight,
    )
    composer_loss = ComposerV2Loss.from_config(loss_cfg).to(device)
    optimizer = optim.AdamW(
        model.parameters(), lr=train_config.phase1.lr, weight_decay=1e-4,
    )
    scheduler = PhaseSchedulerV2(train_config)

    dataset_dir = os.path.join(save_base_dir, dataset_name)
    trainer = UMPAv2Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        train_config=train_config,
        composer_loss=composer_loss,
        evaluate_fn=evaluate,
        save_dir=dataset_dir,
        device=device,
    )

    # Save config snapshot
    cfg_path = os.path.join(trainer.save_dir, "run_config.json")
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "method": "method1_v1style",
                "dataset": dataset_name,
                "config": train_config.to_dict(),
                "timestamp": datetime.now().isoformat(),
            },
            f, indent=2, ensure_ascii=False,
        )

    final_metrics = trainer.run()

    elapsed = time.time() - t0
    print(f"\n✅ {dataset_name} done | Best Dice: {trainer.best_val_dice:.4f} | {elapsed:.0f}s")

    return {
        "status": "OK",
        "best_dice": trainer.best_val_dice,
        "test_metrics": final_metrics,
        "save_dir": trainer.save_dir,
        "elapsed": elapsed,
    }


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="UMPAv2 Method 1 — V1-style 3-phase training",
    )
    p.add_argument("--datasets", nargs="+", default=None, choices=list_datasets())
    p.add_argument("--list", action="store_true")

    # Paths
    p.add_argument("--sam-checkpoint", default="model_trained/sam3.pt")
    p.add_argument("--save-dir", default="checkpoints/umpa_v2_method1")

    # Model
    p.add_argument("--embed-dim", type=int, default=256)
    p.add_argument("--text-embed-dim", type=int, default=512)

    # Training schedule
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--K", type=int, default=3)
    p.add_argument("--epochs1", type=int, default=5)
    p.add_argument("--epochs2", type=int, default=5)
    p.add_argument("--epochs3", type=int, default=10)
    p.add_argument("--lr1", type=float, default=1e-4)
    p.add_argument("--lr2", type=float, default=5e-5)
    p.add_argument("--lr3", type=float, default=1e-5)
    p.add_argument("--lambda-con", type=float, default=0.5)
    p.add_argument("--lambda-reg", type=float, default=0.1)

    # Runtime
    p.add_argument("--device", default="cuda")
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--dry-run", action="store_true")

    return p.parse_args()


def main():
    args = parse_args()

    if args.list:
        for name in list_datasets():
            print(f"  {name}")
        return

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("⚠️ CUDA not available, using CPU")
        device = "cpu"

    datasets = args.datasets or list_datasets()

    print(f"""
╔══════════════════════════════════════════════════════════════╗
║         UMPAv2 METHOD 1 — V1-STYLE 3-PHASE TRAINING        ║
╠══════════════════════════════════════════════════════════════╣
║  Datasets:   {', '.join(datasets):<47s}║
║  Device:     {device:<47s}║
║  Dry-run:    {str(args.dry_run):<47s}║
╚══════════════════════════════════════════════════════════════╝
""")

    os.makedirs(args.save_dir, exist_ok=True)
    all_results = {}
    total_start = time.time()

    for i, ds_name in enumerate(datasets, 1):
        print(f"\n{'#'*60}")
        print(f"# [{i}/{len(datasets)}] {ds_name}")
        print(f"{'#'*60}")

        try:
            result = train_single_dataset(
                dataset_name=ds_name,
                save_base_dir=args.save_dir,
                device=device,
                dry_run=args.dry_run,
                args=args,
            )
        except Exception as e:
            result = {"status": "FAILED", "error": str(e), "elapsed": 0}
            print(f"\n❌ {ds_name} FAILED: {e}")
            import traceback
            traceback.print_exc()
        finally:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        all_results[ds_name] = result

    total_elapsed = time.time() - total_start
    h, rem = divmod(int(total_elapsed), 3600)
    m, s = divmod(rem, 60)
    print(f"\n⏱️ Total time: {h}h {m:02d}m {s:02d}s")

    # Save global results
    results_path = os.path.join(args.save_dir, "all_results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(
            {k: {sk: sv for sk, sv in v.items() if not isinstance(sv, torch.Tensor)}
             for k, v in all_results.items()},
            f, indent=2, default=str,
        )

    failed = [k for k, v in all_results.items() if v["status"] != "OK"]
    if failed:
        print(f"\n⚠️ {len(failed)} failed: {', '.join(failed)}")
        sys.exit(1)
    else:
        print("\n🎉 All datasets trained successfully!")


if __name__ == "__main__":
    main()
