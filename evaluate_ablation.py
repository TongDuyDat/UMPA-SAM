"""CLI for evaluating ablation study checkpoints.

Usage
-----
    # Evaluate a single scenario
    python evaluate_ablation.py --scenario only_box \\
        --checkpoint checkpoints/ablation/only_box/run_.../best_model.pth

    # Evaluate all scenarios (auto-find latest best_model.pth)
    python evaluate_ablation.py --all --base-dir checkpoints/ablation

    # Just generate summary table from existing test_results.json
    python evaluate_ablation.py --summary --base-dir checkpoints/ablation
"""

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from umpt_sam.config.experiment_config import (
    ExperimentConfig,
    get_scenario,
    list_scenarios,
)
from umpt_sam.config.model_config import UMPAModelConfig, UPFEConfig, MPPGConfig
from umpt_sam.umpa_model import UMPAModel
from umpt_sam.ablation_wrapper import AblationModelWrapper
from umpt_sam.training.ablation_results import AblationResultsManager
from umpt_sam.training.evaluate import evaluate
from umpt_sam.data.polyp_dataset import PolypDataset, DatasetConfig, collate_fn
from umpt_sam.data.dataset_registry import get_dataset_config, list_datasets
from umpt_sam.data.polyp_transforms import POLYP_TRANSFORMS

from torch.utils.data import DataLoader


def build_test_loader(batch_size: int = 4, split: str = "test", dataset_name: str = "kvasir_seg"):
    """Build test dataloader."""
    dataset_config = get_dataset_config(dataset_name)
    try:
        dataset = PolypDataset(cfg=dataset_config, phase=split)
    except ValueError:
        print(f"⚠️ Split '{split}' không tìm thấy, fallback sang 'val'")
        split = "val"
        dataset = PolypDataset(cfg=dataset_config, phase=split)
    dataset.transform = POLYP_TRANSFORMS['val']

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )
    return loader, split


def evaluate_scenario(
    scenario_name: str,
    checkpoint_path: str,
    device: str = "cuda",
    batch_size: int = 4,
    sam_checkpoint: str = "model_trained/sam3.pt",
    save_dir: str = None,
    dataset_name: str = "kvasir_seg",
) -> dict:
    """Evaluate a single ablation scenario from checkpoint."""
    exp_cfg = get_scenario(scenario_name)
    print(f"\n🔬 Evaluating: {scenario_name}")
    print(f"   Checkpoint: {checkpoint_path}")

    # Build model
    model_config = UMPAModelConfig(
        sam_checkpoint=sam_checkpoint,
        checkpoint_path=checkpoint_path,
        embed_dim=256,
        text_embed_dim=512,
        freeze_image_encoder=True,
        upfe=UPFEConfig(scoring_hidden_dim=256),
        mppg=MPPGConfig(),
    )
    base_model = UMPAModel.from_config(
        model_config=model_config,
        checkpoint_path=checkpoint_path,
        map_location="cpu",
    ).to(device)

    # Wrap with ablation controls
    model = AblationModelWrapper.wrap(base_model, exp_cfg)

    # Build test loader
    test_loader, split = build_test_loader(batch_size, dataset_name=dataset_name)
    print(f"   Dataset: {dataset_name} | Split: {split} ({len(test_loader.dataset)} samples)")

    # Evaluate
    metrics = evaluate(model, test_loader, device=device)
    print(f"   Dice: {metrics.get('dice', 0):.6f}")
    print(f"   mIoU: {metrics.get('miou', 0):.6f}")

    # Save results if save_dir provided
    if save_dir:
        results_mgr = AblationResultsManager(base_dir=os.path.dirname(save_dir))
        results_mgr.save_test_results(
            save_dir=save_dir,
            metrics=metrics,
            scenario_name=scenario_name,
        )
        print(f"   ✅ Saved to: {save_dir}/test_results.json")

    return metrics


def evaluate_all(
    base_dir: str,
    device: str = "cuda",
    batch_size: int = 4,
    sam_checkpoint: str = "model_trained/sam3.pt",
    dataset_name: str = "kvasir_seg",
):
    """Evaluate all scenarios by finding their latest best_model.pth."""
    dataset_base_dir = os.path.join(base_dir, dataset_name)
    results_mgr = AblationResultsManager(base_dir=dataset_base_dir)
    evaluated = 0
    skipped = 0

    for scenario_name in list_scenarios():
        scenario_dir = os.path.join(dataset_base_dir, scenario_name)
        if not os.path.exists(scenario_dir):
            print(f"⏭️  {scenario_name}: không tìm thấy thư mục")
            skipped += 1
            continue

        # Find latest run directory
        from pathlib import Path
        latest_run = results_mgr._find_latest_run(Path(scenario_dir))
        if latest_run is None:
            print(f"⏭️  {scenario_name}: không có run directory")
            skipped += 1
            continue

        best_path = latest_run / "best_model.pth"
        if not best_path.exists():
            print(f"⏭️  {scenario_name}: không có best_model.pth")
            skipped += 1
            continue

        evaluate_scenario(
            scenario_name=scenario_name,
            checkpoint_path=str(best_path),
            device=device,
            batch_size=batch_size,
            sam_checkpoint=sam_checkpoint,
            save_dir=str(latest_run),
            dataset_name=dataset_name,
        )
        evaluated += 1

    print(f"\n📊 Evaluated: {evaluated}, Skipped: {skipped}")

    # Generate summary
    csv_path = results_mgr.generate_summary_csv()
    print(f"\n📄 Summary CSV: {csv_path}")
    print("\n" + results_mgr.generate_summary_table())


def main():
    parser = argparse.ArgumentParser(
        description="UMPT-SAM Ablation Study Evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"Available scenarios: {', '.join(list_scenarios())}",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--scenario",
        type=str,
        choices=list_scenarios(),
        help="Scenario name to evaluate",
    )
    group.add_argument(
        "--all",
        action="store_true",
        help="Evaluate all scenarios",
    )
    group.add_argument(
        "--summary",
        action="store_true",
        help="Only generate summary table from existing results",
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        help="Checkpoint path (required with --scenario)",
    )
    parser.add_argument(
        "--base-dir",
        type=str,
        default="checkpoints/ablation",
        help="Base directory with scenario results",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--sam-checkpoint",
        type=str,
        default="model_trained/sam3.pt",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="kvasir_seg",
        choices=list_datasets(),
        help="Polyp dataset name",
    )

    args = parser.parse_args()

    if args.summary:
        dataset_base = os.path.join(args.base_dir, args.dataset)
        results_mgr = AblationResultsManager(base_dir=dataset_base)
        csv_path = results_mgr.generate_summary_csv()
        print(f"📄 [{args.dataset}] Summary CSV: {csv_path}")
        print("\n" + results_mgr.generate_summary_table())
        return

    if args.all:
        evaluate_all(
            base_dir=args.base_dir,
            device=args.device,
            batch_size=args.batch_size,
            sam_checkpoint=args.sam_checkpoint,
            dataset_name=args.dataset,
        )
        return

    # Single scenario
    if not args.checkpoint:
        parser.error("--checkpoint is required with --scenario")

    evaluate_scenario(
        scenario_name=args.scenario,
        checkpoint_path=args.checkpoint,
        device=args.device,
        batch_size=args.batch_size,
        sam_checkpoint=args.sam_checkpoint,
        save_dir=os.path.dirname(args.checkpoint),
        dataset_name=args.dataset,
    )


if __name__ == "__main__":
    main()
