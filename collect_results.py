"""Thu thập toàn bộ kết quả huấn luyện từ tất cả dataset × scenario.

Quét thư mục checkpoints, gom test_results.json + training_history.json
của mọi kịch bản vào 1 file JSON duy nhất + in bảng tổng hợp ra console.

Usage
-----
    # Mặc định quét checkpoints/ablation
    python collect_results.py

    # Chỉ định thư mục khác
    python collect_results.py --dir checkpoints/ablation

    # Xuất ra file cụ thể
    python collect_results.py --output results/all_results.json
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path


def find_latest_run(scenario_dir: Path):
    """Tìm thư mục run_* mới nhất trong 1 scenario."""
    if not scenario_dir.exists():
        return None
    runs = sorted(
        [d for d in scenario_dir.iterdir() if d.is_dir() and d.name.startswith("run_")],
        key=lambda d: d.name,
        reverse=True,
    )
    return runs[0] if runs else None


def load_json(path: Path):
    """Đọc file JSON, trả None nếu không tồn tại."""
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def collect(base_dir: str):
    """Quét toàn bộ base_dir và thu thập kết quả."""
    base = Path(base_dir)
    if not base.exists():
        print(f"❌ Thư mục không tồn tại: {base_dir}")
        sys.exit(1)

    all_results = {}

    # Duyệt qua các dataset
    for dataset_dir in sorted(base.iterdir()):
        if not dataset_dir.is_dir():
            continue
        dataset_name = dataset_dir.name

        # Bỏ qua nếu không có scenario con
        scenario_dirs = [d for d in dataset_dir.iterdir() if d.is_dir()]
        if not scenario_dirs:
            continue

        dataset_results = {}

        for scenario_dir in sorted(scenario_dirs):
            scenario_name = scenario_dir.name

            # Tìm run mới nhất
            run_dir = find_latest_run(scenario_dir)
            if run_dir is None:
                continue

            # Đọc test_results.json
            test_results = load_json(run_dir / "test_results.json")
            # Đọc experiment_config.json
            exp_config = load_json(run_dir / "experiment_config.json")
            # Đọc training_history.json (lấy epoch cuối)
            history = load_json(run_dir / "training_history.json")

            entry = {
                "run_dir": str(run_dir),
                "scenario": scenario_name,
            }

            # Metrics từ test_results.json
            if test_results and "metrics" in test_results:
                entry["test_metrics"] = test_results["metrics"]
            else:
                entry["test_metrics"] = None

            # Config tóm tắt
            if exp_config:
                entry["config"] = {
                    "active_prompts": exp_config.get("active_prompts"),
                    "active_components": exp_config.get("active_components"),
                    "K": exp_config.get("effective_K"),
                    "lambda_con": exp_config.get("effective_lambda_con"),
                }
            else:
                entry["config"] = None

            # Best val metrics từ history (epoch cuối cùng có val_dice cao nhất)
            if history:
                best_epoch = max(history, key=lambda x: x.get("val_dice", 0))
                entry["best_val"] = {
                    "epoch": best_epoch.get("epoch"),
                    "val_dice": best_epoch.get("val_dice"),
                    "val_miou": best_epoch.get("val_miou"),
                }
                entry["total_epochs"] = len(history)
            else:
                entry["best_val"] = None
                entry["total_epochs"] = None

            dataset_results[scenario_name] = entry

        if dataset_results:
            all_results[dataset_name] = dataset_results

    return all_results


def print_table(all_results: dict):
    """In bảng tổng hợp ra console."""
    METRICS = ["dice", "miou", "precision", "recall", "f2"]

    for dataset_name, scenarios in all_results.items():
        print(f"\n{'='*90}")
        print(f" DATASET: {dataset_name}")
        print(f"{'='*90}")

        # Header
        header = f"{'Scenario':<22}"
        for m in METRICS:
            header += f" {m.upper():>10}"
        header += f" {'BestEpoch':>10} {'ValDice':>10}"
        print(header)
        print("-" * 90)

        for sc_name, entry in sorted(scenarios.items()):
            row = f"{sc_name:<22}"
            metrics = entry.get("test_metrics") or {}
            for m in METRICS:
                val = metrics.get(m)
                if val is not None and isinstance(val, (int, float)):
                    row += f" {val:>10.4f}"
                else:
                    row += f" {'N/A':>10}"
            
            best_val = entry.get("best_val") or {}
            best_ep = best_val.get("epoch", "N/A")
            best_dice = best_val.get("val_dice")
            row += f" {str(best_ep):>10}"
            if best_dice is not None:
                row += f" {best_dice:>10.4f}"
            else:
                row += f" {'N/A':>10}"
            
            print(row)

        print(f"{'='*90}")


def main():
    parser = argparse.ArgumentParser(
        description="Thu thập toàn bộ kết quả huấn luyện UMPT-SAM",
    )
    parser.add_argument(
        "--dir", type=str, default="checkpoints/ablation",
        help="Thư mục gốc chứa kết quả (default: checkpoints/ablation)",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="File JSON xuất ra (default: <dir>/all_results.json)",
    )
    args = parser.parse_args()

    output_path = args.output or os.path.join(args.dir, "all_results.json")

    print(f"📂 Đang quét: {os.path.abspath(args.dir)}")
    all_results = collect(args.dir)

    if not all_results:
        print("⚠️  Không tìm thấy kết quả nào!")
        sys.exit(0)

    # Tạo báo cáo
    report = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "base_dir": os.path.abspath(args.dir),
        "total_datasets": len(all_results),
        "total_experiments": sum(len(v) for v in all_results.values()),
        "results": all_results,
    }

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    # In bảng
    print_table(all_results)

    print(f"\n✅ Đã lưu toàn bộ kết quả vào: {os.path.abspath(output_path)}")
    print(f"   - {report['total_datasets']} datasets")
    print(f"   - {report['total_experiments']} experiments")


if __name__ == "__main__":
    main()
