"""Gom tất cả training_log.txt vào 1 file duy nhất.

Quét: checkpoints/{variant}/{dataset}/{variant}/run_*/training_log.txt

Usage
-----
    python collect_logs.py --variant efused_only
    python collect_logs.py --variant esparse_efused_etext
    python collect_logs.py --variant efused_only esparse_efused_etext
    python collect_logs.py --variant efused_only --output logs/all_logs.txt
"""

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path


def collect_logs(base_dir: Path, variant: str):
    """Tìm tất cả training_log.txt trong base_dir."""
    logs = []
    if not base_dir.exists():
        return logs

    for dataset_dir in sorted(base_dir.iterdir()):
        if not dataset_dir.is_dir():
            continue
        dataset_name = dataset_dir.name

        # Tìm trong {dataset}/{variant}/run_*/
        scenario_dir = dataset_dir / variant
        if not scenario_dir.exists():
            scenario_dir = dataset_dir

        # Duyệt tất cả run_* (không chỉ latest)
        run_dirs = sorted(
            [d for d in scenario_dir.iterdir() if d.is_dir() and d.name.startswith("run_")],
            key=lambda d: d.name,
        ) if scenario_dir.exists() else []

        # Fallback: log nằm trực tiếp
        if not run_dirs and (scenario_dir / "training_log.txt").exists():
            run_dirs = [scenario_dir]

        for run_dir in run_dirs:
            log_path = run_dir / "training_log.txt"
            if log_path.exists():
                logs.append({
                    "variant": variant,
                    "dataset": dataset_name,
                    "run": run_dir.name,
                    "path": log_path,
                })

    return logs


def main():
    parser = argparse.ArgumentParser(description="Gom training logs vào 1 file")
    parser.add_argument(
        "--variant", type=str, nargs="+", required=True,
        help="Tên variant(s): efused_only, esparse_efused_etext, ...",
    )
    parser.add_argument(
        "--dir", type=str, default=None,
        help="Thư mục gốc (default: checkpoints/<variant>)",
    )
    parser.add_argument(
        "--output", type=str, default="checkpoints/all_training_logs.txt",
        help="File output (default: checkpoints/all_training_logs.txt)",
    )
    args = parser.parse_args()

    all_logs = []
    for variant in args.variant:
        base_dir = Path(args.dir) if args.dir else Path(f"checkpoints/{variant}")
        logs = collect_logs(base_dir, variant)
        all_logs.extend(logs)
        print(f"📂 {variant}: tìm thấy {len(logs)} log(s) trong {base_dir}")

    if not all_logs:
        print("⚠️  Không tìm thấy training_log.txt nào!")
        sys.exit(0)

    # Gom vào 1 file
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as out:
        out.write(f"{'#'*80}\n")
        out.write(f"#  UMPT-SAM — Tổng hợp Training Logs\n")
        out.write(f"#  Variants: {', '.join(args.variant)}\n")
        out.write(f"#  Tổng: {len(all_logs)} logs\n")
        out.write(f"#  Thời gian: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        out.write(f"{'#'*80}\n\n")

        for i, entry in enumerate(all_logs, 1):
            out.write(f"\n{'='*80}\n")
            out.write(f"[{i}/{len(all_logs)}] {entry['variant']} / {entry['dataset']} / {entry['run']}\n")
            out.write(f"File: {entry['path']}\n")
            out.write(f"{'='*80}\n\n")

            with open(entry["path"], "r", encoding="utf-8") as log_f:
                out.write(log_f.read())

            out.write("\n")

    print(f"\n✅ Đã gom {len(all_logs)} logs vào: {os.path.abspath(args.output)}")


if __name__ == "__main__":
    main()
