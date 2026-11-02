#!/usr/bin/env bash
# Real-time pipeline experiments. Run inside the container from the repo root:
#   bash realtime/run_realtime.sh 0            # USB webcam
#   bash realtime/run_realtime.sh traffic.mp4  # video file
# Needs the engines from case study 1 (bash run_all.sh).
set -euo pipefail
SRC="${1:?usage: bash realtime/run_realtime.sh <source: 0 | csi | video.mp4 | rtsp://...>}"
M=models
P="python3 realtime/pipeline.py --source $SRC"

# Part A: capacity. How many FPS can the whole pipeline do? (files only)
if [ -f "$SRC" ]; then
  $P --model $M/yolo11n.pt          --max-speed --label "A1 PyTorch FP32"
  $P --model $M/yolo11n_fp16.engine --max-speed --label "A2 TensorRT FP16"
  $P --model $M/yolo11n_int8.engine --max-speed --label "A3 TensorRT INT8"
  $P --model $M/yolo11n_fp16.engine --max-speed --track --label "A4 FP16 + tracking"
fi

# Part B: real time. Frames arrive at camera speed; what latency does the viewer see?
$P --model $M/yolo11n.pt          --sync          --label "B1 PyTorch, naive reader"
$P --model $M/yolo11n.pt                          --label "B2 PyTorch, threaded reader"
$P --model $M/yolo11n_fp16.engine --track         --label "B3 FP16 + tracking, threaded"
$P --model $M/yolo11n_fp16.engine --track --save realtime/results/demo.mp4 --frames 300 \
   --label "B4 FP16 + tracking + video out"

python3 realtime/make_realtime_report.py
