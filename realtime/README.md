# Case study 2: Real-time detection + tracking pipeline

Case study 1 measured the **model** alone. A real product runs a **pipeline**: grab a frame
from the camera, pre-process, run the model, run NMS and tracking, draw and output. This
experiment measures every stage, so you can see where the time actually goes and what
latency a viewer experiences.

## What it shows
- **Capacity (Part A):** the maximum FPS of the full pipeline with PyTorch, TensorRT FP16
  and INT8, and the extra cost of ByteTrack multi-object tracking.
- **Real time (Part B):** frames arrive at camera speed (30 FPS). Compares:
  - a **naive reader** (read → process → read). When the model is slow, stale frames
    pile up in the camera buffer, so latency balloons.
  - a **threaded reader** that always hands the model the newest frame and drops the rest.
    This is how production pipelines keep latency low.
- **End-to-end latency** (p50/p95), dropped frames and board power for every run.

## Run it
You need the engines from case study 1 first (`bash run_all.sh` in the repo root).

**1. Get a video source.** Either:
- a **USB webcam** (source `0`), or
- a **video file**, for example a 720p or 1080p street or traffic clip. Free stock-video
  sites have plenty, or record 30 seconds on your phone. Copy it into the repo folder as
  `traffic.mp4`.

**2. Start the container** (same as case study 1, plus the camera device if you use one):
```bash
sudo docker run -it --rm --ipc=host --runtime=nvidia \
  -v "$PWD":/work -w /work \
  -v /usr/bin/tegrastats:/usr/bin/tegrastats \
  -v /proc/device-tree/model:/host/model:ro \
  -v /var/lib/nvpmodel:/var/lib/nvpmodel:ro \
  -v /etc/nvpmodel.conf:/etc/nvpmodel.conf:ro \
  --device /dev/video0 \
  ultralytics/ultralytics:latest-jetson-jetpack6
```
Drop the `--device /dev/video0` line if you have no webcam.

**3. Run all experiments** (about 5 minutes):
```bash
bash realtime/run_realtime.sh traffic.mp4   # or: bash realtime/run_realtime.sh 0
```

**Outputs** in `realtime/results/`:
| File | What it is |
|---|---|
| `realtime_results.csv` | one row per run: FPS, per-stage ms, latency, dropped frames, power |
| `realtime_report.md` | ready-to-paste results table |
| `stage_breakdown.png` | stacked bar chart of where each frame's time goes |
| `demo.mp4` | annotated video with boxes, track IDs and a live FPS/latency overlay |

## Single runs
```bash
python3 realtime/pipeline.py --source traffic.mp4 --model models/yolo11n_fp16.engine --track
python3 realtime/pipeline.py --source 0 --model models/yolo11n_int8.engine --show   # needs a display
python3 realtime/pipeline.py --help
```

## How latency is measured
- **Camera:** from the moment the frame is handed to the application until the result
  (boxes drawn and written or shown) is ready.
- **Video files** are played at their real frame rate to imitate a camera. With the naive
  reader, a 4-frame camera buffer is simulated, which matches typical USB (V4L2) camera
  behaviour.
- The first 30 frames are warm-up and are not counted.
