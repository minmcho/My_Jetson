#!/usr/bin/env bash
# Export, benchmark and report for one model. Usage: bash run_all.sh [yolo11n.pt]
set -euo pipefail
MODEL="${1:-yolo11n.pt}"
STEM="$(basename "$MODEL" .pt)"

python3 scripts/export.py --model "$MODEL"
python3 scripts/benchmark.py --model "$MODEL"
python3 scripts/make_report.py --model "$STEM"
