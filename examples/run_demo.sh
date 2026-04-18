#!/bin/bash
# run_demo.sh — wrapper to run best_result_demo.py in sam3 env
source /opt/conda/etc/profile.d/conda.sh
conda activate sam3
cd /mnt/d/NCKH/NCKH2025/polyp/sam3/examples
python best_result_demo.py "$@"
