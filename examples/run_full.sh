#!/bin/bash
# run_full.sh — Run full inference + perturbation demo
set -e

source /opt/conda/etc/profile.d/conda.sh
conda activate sam3
cd /mnt/d/NCKH/NCKH2025/polyp/sam3/examples

echo "=============================="
echo "Step 1: Full inference (all test images)"
echo "=============================="
python best_result_demo.py --mode full

echo ""
echo "=============================="
echo "Step 2: Perturbation degradation demo"
echo "=============================="
python best_result_perturbed.py

echo ""
echo "=============================="
echo "ALL DONE!"
echo "=============================="
