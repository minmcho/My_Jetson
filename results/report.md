## Results: yolo11n on Jetson Orin Nano Super

L4T: `R39.2.1` · Power mode: `not recorded` · Image size: 640

| Format | Mean latency (ms) | p95 (ms) | FPS | Speed-up | mAP50-95 | Δ mAP | Avg power (W) | Energy / frame (mJ) |
|---|---|---|---|---|---|---|---|---|
| PyTorch FP32 | 30.36 | 30.89 | 32.9 | 1.0× | 0.508 | +0.0 pts | 8.79 | 267 |
| ONNX FP32 | 24.19 | 24.46 | 41.3 | 1.3× | 0.505 | -0.2 pts | 10.23 | 248 |
| TensorRT FP16 | 8.94 | 9.01 | 111.9 | 3.4× | 0.504 | -0.4 pts | 9.59 | 86 |
| TensorRT INT8 | 7.85 | 8.12 | 127.4 | 3.9× | 0.475 | -3.2 pts | 8.89 | 70 |

**Takeaway:** TensorRT INT8 runs 3.9× faster than PyTorch FP32 with a 3.2-point mAP50-95 change, using 70 mJ per frame vs 267 mJ.
