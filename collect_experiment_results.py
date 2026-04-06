"""Thu thập kết quả thí nghiệm từ tất cả dataset cho bất kỳ variant nào.

Quét: checkpoints/{variant}/{dataset}/{variant}/run_*/test_results.json

Usage
-----
    # Efused only
    python collect_experiment_results.py --variant efused_only

    # Esparse + Efused + Etext
    python collect_experiment_results.py --variant esparse_efused_etext

    # Chỉ định thư mục khác
    python collect_experiment_results.py --variant efused_only --dir checkpoints/efused_only

    # Xuất JSON
    python collect_experiment_results.py --variant efused_only --output results/report.json
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path


METRICS = ["dice", "miou", "precision", "recall", "f2"]


def load_json(path: Path):
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def find_latest_run(directory: Path):
    """Tìm run_* mới nhất trong thư mục."""
    if not directory.exists():
        return None
    runs = sorted(
        [d for d in directory.iterdir() if d.is_dir() and d.name.startswith("run_")],
        key=lambda d: d.name,
        reverse=True,
    )
    return runs[0] if runs else None


def collect(base_dir: str, variant: str):
    """Quét checkpoints/{variant}/{dataset}/{variant}/run_*/"""
    base = Path(base_dir)
    if not base.exists():
        print(f"❌ Thư mục không tồn tại: {base_dir}")
        sys.exit(1)

    results = {}

    for dataset_dir in sorted(base.iterdir()):
        if not dataset_dir.is_dir():
            continue
        dataset_name = dataset_dir.name

        # Tìm scenario dir: {dataset}/{variant}/
        scenario_dir = dataset_dir / variant

        # Fallback 1: run_* trực tiếp trong dataset_dir (nếu không có tầng variant)
        if not scenario_dir.exists():
            scenario_dir = dataset_dir

        run_dir = find_latest_run(scenario_dir)

        # Fallback 2: test_results.json trực tiếp (không có run_*)
        if run_dir is None:
            if (scenario_dir / "test_results.json").exists():
                run_dir = scenario_dir
            else:
                print(f"  ⚠️  {dataset_name}: không tìm thấy run nào")
                continue

        test_data = load_json(run_dir / "test_results.json")
        history = load_json(run_dir / "training_history.json")
        exp_config = load_json(run_dir / "experiment_config.json")

        entry = {
            "dataset": dataset_name,
            "run_dir": str(run_dir),
            "test_metrics": test_data.get("metrics") if test_data else None,
        }

        # Best val từ history
        if history:
            best = max(history, key=lambda x: x.get("val_dice", 0))
            entry["best_val"] = {
                "epoch": best.get("epoch"),
                "val_dice": best.get("val_dice"),
                "val_miou": best.get("val_miou"),
            }
            entry["total_epochs"] = len(history)
        else:
            entry["best_val"] = None
            entry["total_epochs"] = None

        if exp_config:
            entry["config_summary"] = {
                "active_prompts": exp_config.get("active_prompts"),
                "active_components": exp_config.get("active_components"),
            }

        results[dataset_name] = entry

    return results


def print_table(results: dict, variant: str):
    """In bảng kết quả ra console."""
    print(f"\n{'='*100}")
    print(f"  UMPT-SAM — {variant} — Tổng hợp kết quả")
    print(f"{'='*100}")

    header = f"{'Dataset':<22}"
    for m in METRICS:
        header += f" {m.upper():>10}"
    header += f" {'BestEp':>8} {'ValDice':>10}"
    print(header)
    print("-" * 100)

    avg = {m: [] for m in METRICS}

    for ds_name, entry in sorted(results.items()):
        row = f"{ds_name:<22}"
        metrics = entry.get("test_metrics") or {}
        for m in METRICS:
            val = metrics.get(m)
            if isinstance(val, (int, float)):
                row += f" {val:>10.4f}"
                avg[m].append(val)
            else:
                row += f" {'N/A':>10}"

        best_val = entry.get("best_val") or {}
        best_ep = best_val.get("epoch", "N/A")
        best_dice = best_val.get("val_dice")
        row += f" {str(best_ep):>8}"
        row += f" {best_dice:>10.4f}" if isinstance(best_dice, (int, float)) else f" {'N/A':>10}"

        print(row)

    # Dòng trung bình
    print("-" * 100)
    avg_row = f"{'AVERAGE':<22}"
    for m in METRICS:
        vals = avg[m]
        if vals:
            avg_row += f" {sum(vals)/len(vals):>10.4f}"
        else:
            avg_row += f" {'N/A':>10}"
    avg_row += f" {'':>8} {'':>10}"
    print(avg_row)
    print(f"{'='*100}")


def main():
    parser = argparse.ArgumentParser(
        description="Thu thập kết quả thí nghiệm UMPT-SAM cho bất kỳ variant",
    )
    parser.add_argument(
        "--variant", type=str, required=True,
        help="Tên variant (efused_only, esparse_efused_etext, ...)",
    )
    parser.add_argument(
        "--dir", type=str, default=None,
        help="Thư mục gốc (default: checkpoints/<variant>)",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="File JSON xuất ra (default: <dir>/<variant>_results.json)",
    )
    args = parser.parse_args()

    base_dir = args.dir or f"checkpoints/{args.variant}"
    output_path = args.output or os.path.join(base_dir, f"{args.variant}_results.json")

    print(f"📂 Variant: {args.variant}")
    print(f"📂 Đang quét: {os.path.abspath(base_dir)}")
    results = collect(base_dir, args.variant)

    if not results:
        print("⚠️  Không tìm thấy kết quả nào!")
        sys.exit(0)

    # In bảng
    print_table(results, args.variant)

    # Lưu JSON
    report = {
        "variant": args.variant,
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "base_dir": os.path.abspath(base_dir),
        "total_datasets": len(results),
        "results": results,
    }
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Đã lưu: {os.path.abspath(output_path)}")
    print(f"   - {len(results)} datasets")


if __name__ == "__main__":
    main()
