"""Benchmark every exported format: latency, FPS, mAP50-95 and average board power.

Usage: python3 scripts/benchmark.py --model yolo11n.pt [--imgsz 640] [--data coco128.yaml]
Appends one row per format to results/results.csv.
"""
import argparse
import csv
import platform
import re
import statistics
import subprocess
import threading
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from ultralytics import YOLO

WARMUP = 30
RUNS = 300


class PowerSampler:
    """Reads total board power (mW) from tegrastats while a benchmark runs."""

    def __init__(self, interval_ms=200):
        self.interval_ms = interval_ms
        self.samples = []
        self.proc = None
        self.thread = None

    def start(self):
        try:
            self.proc = subprocess.Popen(
                ["tegrastats", "--interval", str(self.interval_ms)],
                stdout=subprocess.PIPE, text=True)
        except FileNotFoundError:
            print("[warn] tegrastats not found; power will be blank")
            return
        self.thread = threading.Thread(target=self._read, daemon=True)
        self.thread.start()

    def _read(self):
        for line in self.proc.stdout:
            # e.g. "VDD_IN 5123mW/5000mW" (Orin) or "POM_5V_IN 4321/4000" (older)
            m = re.search(r"(?:VDD_IN|POM_5V_IN)\s+(\d+)", line)
            if m:
                self.samples.append(int(m.group(1)))

    def stop(self):
        if self.proc:
            self.proc.terminate()
            self.proc.wait(timeout=5)
        return round(statistics.mean(self.samples) / 1000, 2) if self.samples else ""


def board_info():
    info = {"board": platform.node(), "jetpack": "", "power_mode": ""}
    model = Path("/proc/device-tree/model")
    if model.exists():
        info["board"] = model.read_text().strip("\x00").strip()
    rel = Path("/etc/nv_tegra_release")
    if rel.exists():
        info["jetpack"] = rel.read_text().splitlines()[0][:60]
    try:
        out = subprocess.run(["nvpmodel", "-q"], capture_output=True, text=True).stdout
        m = re.search(r"NV Power Mode:\s*(\S+)", out)
        info["power_mode"] = m.group(1) if m else ""
    except FileNotFoundError:
        pass
    return info


def time_inference(model, imgsz):
    img = (np.random.rand(imgsz, imgsz, 3) * 255).astype(np.uint8)
    for _ in range(WARMUP):
        model.predict(img, imgsz=imgsz, verbose=False)
    lat = []
    for _ in range(RUNS):
        t0 = time.perf_counter()
        model.predict(img, imgsz=imgsz, verbose=False)
        lat.append((time.perf_counter() - t0) * 1000)
    lat.sort()
    return statistics.mean(lat), lat[int(0.95 * len(lat)) - 1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="yolo11n.pt")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--data", default="coco128.yaml",
                    help="dataset yaml for mAP (use coco.yaml for publishable numbers)")
    a = ap.parse_args()

    stem = Path(a.model).stem
    models = Path("models")
    formats = [
        ("PyTorch FP32", models / f"{stem}.pt"),
        ("ONNX FP32", models / f"{stem}_fp32.onnx"),
        ("TensorRT FP16", models / f"{stem}_fp16.engine"),
        ("TensorRT INT8", models / f"{stem}_int8.engine"),
    ]

    info = board_info()
    print(f"[board] {info}")
    out = Path("results")
    out.mkdir(exist_ok=True)
    csv_path = out / "results.csv"
    fields = ["timestamp", "model", "format", "imgsz", "mean_ms", "p95_ms", "fps",
              "map50_95", "avg_power_w", "board", "jetpack", "power_mode"]
    new_file = not csv_path.exists()

    with csv_path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if new_file:
            w.writeheader()
        for name, path in formats:
            if not path.exists():
                print(f"[skip] {name}: {path} missing (run export.py first)")
                continue
            print(f"\n=== {name} ({path.name}) ===")
            model = YOLO(str(path), task="detect")

            power = PowerSampler()
            power.start()
            mean_ms, p95_ms = time_inference(model, a.imgsz)
            avg_w = power.stop()

            metrics = model.val(data=a.data, imgsz=a.imgsz, batch=1,
                                device=0, verbose=False, plots=False)
            row = {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "model": stem, "format": name, "imgsz": a.imgsz,
                "mean_ms": round(mean_ms, 2), "p95_ms": round(p95_ms, 2),
                "fps": round(1000 / mean_ms, 1),
                "map50_95": round(float(metrics.box.map), 4),
                "avg_power_w": avg_w, **info,
            }
            w.writerow(row)
            f.flush()
            print(row)


if __name__ == "__main__":
    main()
