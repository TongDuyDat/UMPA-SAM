"""One-click runner: Train ALL 10 scenarios × ALL 5 datasets = 50 experiments.

Features
--------
- Fault-tolerant: if a scenario crashes, the error is logged and the next
  scenario starts immediately.
- Generates per-dataset summary CSVs + a cross-dataset error report.
- GPU memory is cleared between experiments to prevent OOM cascading.

Usage
-----
    # Train everything (50 experiments)
    python run_all_datasets.py

    # Dry-run (1 epoch, 4 samples per experiment — verify setup)
    python run_all_datasets.py --dry-run

    # Only 1 dataset, all scenarios
    python run_all_datasets.py --dataset kvasir_seg

    # Only 1 scenario, all datasets
    python run_all_datasets.py --scenario full_model

    # Override device / checkpoint
    python run_all_datasets.py --device cuda:1 --sam-checkpoint /path/to/sam3.pt
"""

import argparse
import gc
import json
import os
import sys
import time
import traceback
from datetime import datetime
from typing import List, Optional

import torch

# Ensure project root is in path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from umpt_sam.config.experiment_config import (
    ExperimentConfig,
    get_scenario,
    list_scenarios,
)
from umpt_sam.data.dataset_registry import list_datasets

# Re-use the existing train_scenario function
from train_ablation import train_scenario


# ======================================================================
# Error report
# ======================================================================

def _save_error_report(
    results: list,
    save_dir: str,
    elapsed_total: float,
) -> str:
    """Save a JSON report of all experiment outcomes."""
    report_path = os.path.join(save_dir, "experiment_report.json")
    report = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "total_elapsed_sec": round(elapsed_total, 1),
        "total_experiments": len(results),
        "succeeded": sum(1 for r in results if r["status"] == "OK"),
        "failed": sum(1 for r in results if r["status"] == "FAILED"),
        "skipped": sum(1 for r in results if r["status"] == "SKIPPED"),
        "experiments": results,
    }
    os.makedirs(save_dir, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    return report_path


def _print_summary_table(results: list) -> None:
    """Print a concise summary table to the console."""
    print(f"\n{'='*75}")
    print(f"{'Dataset':<18} {'Scenario':<18} {'Status':<10} {'Time':<12} {'Note'}")
    print(f"{'-'*75}")
    for r in results:
        t = f"{r['elapsed_sec']:.0f}s" if r["elapsed_sec"] else "—"
        note = r.get("error_type", "") or ""
        print(f"{r['dataset']:<18} {r['scenario']:<18} {r['status']:<10} {t:<12} {note}")
    print(f"{'='*75}")


def _clear_gpu():
    """Aggressively free GPU memory between experiments."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


# ======================================================================
# Main runner
# ======================================================================

def run_all(
    datasets: List[str],
    scenarios: List[str],
    save_dir: str,
    device: str,
    sam_checkpoint: str,
    dry_run: bool,
) -> list:
    """Train all requested experiments with error resilience."""
    results = []
    total = len(datasets) * len(scenarios)
    idx = 0

    for ds_name in datasets:
        print(f"\n{'█'*70}")
        print(f"█ DATASET: {ds_name}")
        print(f"{'█'*70}")

        for sc_name in scenarios:
            idx += 1
            exp_cfg = get_scenario(sc_name)

            print(f"\n[{idx}/{total}] {ds_name} / {sc_name}")
            print(f"   Prompts: {exp_cfg.active_prompts}")
            print(f"   Components: {exp_cfg.active_components}")
            print(f"   K={exp_cfg.effective_K}, λ_con={exp_cfg.effective_lambda_con}")

            record = {
                "dataset": ds_name,
                "scenario": sc_name,
                "status": "RUNNING",
                "elapsed_sec": 0,
                "error_type": None,
                "error_message": None,
                "traceback": None,
            }

            start = time.time()
            try:
                train_scenario(
                    exp_cfg=exp_cfg,
                    save_base_dir=save_dir,
                    device=device,
                    dry_run=dry_run,
                    sam_checkpoint=sam_checkpoint,
                    dataset_name=ds_name,
                )
                record["status"] = "OK"
                print(f"   ✅ {ds_name}/{sc_name} — THÀNH CÔNG")

            except Exception as e:
                elapsed = time.time() - start
                record["status"] = "FAILED"
                record["error_type"] = type(e).__name__
                record["error_message"] = str(e)[:500]
                record["traceback"] = traceback.format_exc()[-2000:]

                # Log error to file
                error_log_path = os.path.join(
                    save_dir, ds_name, sc_name, "error.log"
                )
                os.makedirs(os.path.dirname(error_log_path), exist_ok=True)
                with open(error_log_path, "w", encoding="utf-8") as f:
                    f.write(f"Dataset: {ds_name}\n")
                    f.write(f"Scenario: {sc_name}\n")
                    f.write(f"Time: {datetime.now().isoformat()}\n")
                    f.write(f"Error: {type(e).__name__}: {e}\n\n")
                    f.write(traceback.format_exc())

                print(f"   ❌ {ds_name}/{sc_name} — LỖI: {type(e).__name__}: {e}")
                print(f"   📄 Error log: {error_log_path}")

            finally:
                record["elapsed_sec"] = round(time.time() - start, 1)
                results.append(record)

                # Clear GPU memory before next experiment
                _clear_gpu()

    return results


def main():
    parser = argparse.ArgumentParser(
        description="UMPT-SAM: One-click train ALL datasets × ALL scenarios",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            f"Datasets ({len(list_datasets())}):  {', '.join(list_datasets())}\n"
            f"Scenarios ({len(list_scenarios())}): {', '.join(list_scenarios())}\n"
            f"\nTotal: {len(list_datasets())} × {len(list_scenarios())} = "
            f"{len(list_datasets()) * len(list_scenarios())} experiments"
        ),
    )
    parser.add_argument(
        "--dataset", type=str, choices=list_datasets(),
        help="Only this dataset (default: all)",
    )
    parser.add_argument(
        "--scenario", type=str, choices=list_scenarios(),
        help="Only this scenario (default: all)",
    )
    parser.add_argument(
        "--save-dir", type=str, default="checkpoints/ablation",
        help="Base save directory",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--sam-checkpoint", type=str, default="model_trained/sam3.pt")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Quick test: 1 epoch, 4 samples per experiment",
    )

    args = parser.parse_args()

    datasets = [args.dataset] if args.dataset else list_datasets()
    scenarios = [args.scenario] if args.scenario else list_scenarios()
    total = len(datasets) * len(scenarios)

    # Header
    print(f"\n{'#'*70}")
    print(f"#  UMPT-SAM — One-Click Experiment Runner")
    print(f"#")
    print(f"#  Datasets:     {len(datasets)} — {', '.join(datasets)}")
    print(f"#  Scenarios:    {len(scenarios)}")
    print(f"#  Total:        {total} experiments")
    print(f"#  Save dir:     {os.path.abspath(args.save_dir)}")
    print(f"#  Device:       {args.device}")
    print(f"#  SAM ckpt:     {args.sam_checkpoint}")
    print(f"#  Dry run:      {args.dry_run}")
    print(f"#  Started:      {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'#'*70}\n")

    start_time = time.time()

    results = run_all(
        datasets=datasets,
        scenarios=scenarios,
        save_dir=args.save_dir,
        device=args.device,
        sam_checkpoint=args.sam_checkpoint,
        dry_run=args.dry_run,
    )

    elapsed_total = time.time() - start_time
    hours, remainder = divmod(int(elapsed_total), 3600)
    minutes, seconds = divmod(remainder, 60)

    # Summary
    succeeded = sum(1 for r in results if r["status"] == "OK")
    failed = sum(1 for r in results if r["status"] == "FAILED")

    _print_summary_table(results)

    report_path = _save_error_report(results, args.save_dir, elapsed_total)

    print(f"\n📊 Tổng kết: {succeeded}/{total} thành công, {failed}/{total} lỗi")
    print(f"⏱️  Thời gian: {hours}h {minutes}m {seconds}s")
    print(f"📄 Report: {report_path}")

    if failed > 0:
        print(f"\n⚠️  Các kịch bản lỗi:")
        for r in results:
            if r["status"] == "FAILED":
                print(f"   ❌ {r['dataset']}/{r['scenario']}: {r['error_type']}: {r['error_message'][:100]}")

    print(f"\n✅ HOÀN TẤT!")


if __name__ == "__main__":
    main()
