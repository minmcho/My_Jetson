# Jetson YOLO Optimization Benchmark

Case study #1 for the QSInfomatic edge-AI portfolio: take a stock YOLO11 model and show the
speed-up from PyTorch → ONNX → TensorRT FP16 → TensorRT INT8 on an NVIDIA Jetson, with the
accuracy cost of each step measured.

## What it produces
- `results/results.csv` — latency, FPS, mAP50-95 and average board power for every format
- `results/report.md` — a ready-to-publish results table
- `results/fps_vs_accuracy.png` — the chart for your README / LinkedIn post

## Quick start (on the Jetson)
1. Do the one-time setup in [SETUP.md](SETUP.md).
2. Start the container from this folder:
   ```bash
   sudo docker run -it --rm --ipc=host --runtime=nvidia \
     -v "$PWD":/work -w /work \
     -v /usr/bin/tegrastats:/usr/bin/tegrastats \
     ultralytics/ultralytics:latest-jetson-jetpack6
   ```
3. Inside the container:
   ```bash
   bash run_all.sh            # default model: yolo11n.pt
   bash run_all.sh yolo11s.pt # try a bigger model too
   ```
   The first run takes a while (TensorRT engine builds + INT8 calibration). Later runs reuse
   the engines in `models/`.

## Files
| File | Purpose |
|---|---|
| `scripts/export.py` | Exports ONNX, TensorRT FP16 and TensorRT INT8 versions of the model |
| `scripts/benchmark.py` | Measures latency/FPS, mAP and power for each format |
| `scripts/make_report.py` | Turns `results.csv` into a markdown table + chart |
| `run_all.sh` | Runs all three in order |

## Publishing checklist
- [ ] Note the board, JetPack version and power mode in the README (benchmark.py records them)
- [ ] Paste `results/report.md` into the README
- [ ] Add `results/fps_vs_accuracy.png`
- [ ] One-paragraph takeaway: "X× faster with <Y% mAP loss"
- [ ] Push to GitHub, then post on LinkedIn
