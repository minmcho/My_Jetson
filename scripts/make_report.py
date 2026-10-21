"""Turn results/results.csv into results/report.md and results/fps_vs_accuracy.png.

Usage: python3 scripts/make_report.py [--model yolo11n]
Uses the most recent row for each format of the chosen model.
"""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ORDER = ["PyTorch FP32", "ONNX FP32", "TensorRT FP16", "TensorRT INT8"]


def load(csv_path, model):
    latest = {}
    with csv_path.open() as f:
        for row in csv.DictReader(f):
            if model and row["model"] != model:
                continue
            latest[row["format"]] = row  # later rows overwrite earlier ones
    return [latest[k] for k in ORDER if k in latest]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="yolo11n")
    a = ap.parse_args()

    out = Path("results")
    rows = load(out / "results.csv", a.model)
    if not rows:
        raise SystemExit(f"No results for model '{a.model}'. Run benchmark.py first.")

    base = rows[0]
    base_fps = float(base["fps"])
    base_map = float(base["map50_95"])

    lines = [
        f"## Results: {a.model} on {base['board']}",
        "",
        f"JetPack: `{base['jetpack']}` · Power mode: `{base['power_mode']}` · "
        f"Image size: {base['imgsz']}",
        "",
        "| Format | Mean latency (ms) | p95 (ms) | FPS | Speed-up | mAP50-95 | Δ mAP | Avg power (W) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        speed = float(r["fps"]) / base_fps
        dmap = (float(r["map50_95"]) - base_map) * 100
        lines.append(
            f"| {r['format']} | {r['mean_ms']} | {r['p95_ms']} | {r['fps']} | "
            f"{speed:.1f}× | {float(r['map50_95']):.3f} | {dmap:+.1f} pts | "
            f"{r['avg_power_w'] or '–'} |")

    best = max(rows, key=lambda r: float(r["fps"]))
    lines += [
        "",
        f"**Takeaway:** {best['format']} runs {float(best['fps']) / base_fps:.1f}× faster "
        f"than PyTorch FP32 with a {(base_map - float(best['map50_95'])) * 100:.1f}-point "
        f"mAP50-95 change.",
    ]
    (out / "report.md").write_text("\n".join(lines) + "\n")

    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=150)
    for r in rows:
        x, y = float(r["fps"]), float(r["map50_95"])
        ax.scatter(x, y, s=80)
        ax.annotate(r["format"], (x, y), textcoords="offset points", xytext=(6, 6), fontsize=9)
    ax.set_xlabel("Throughput (FPS, batch 1)")
    ax.set_ylabel("mAP50-95")
    ax.set_title(f"{a.model}: speed vs accuracy on {base['board']}", fontsize=10)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "fps_vs_accuracy.png")
    print("Wrote results/report.md and results/fps_vs_accuracy.png")


if __name__ == "__main__":
    main()
