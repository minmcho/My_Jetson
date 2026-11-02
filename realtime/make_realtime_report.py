"""Turn realtime/results/realtime_results.csv into a markdown table and a stage-breakdown chart.

Usage: python3 realtime/make_realtime_report.py [--board "Jetson Orin Nano Super"]
Uses the most recent row for each run label.
"""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parent / "results"
STAGES = [("wait", "Waiting for frame"), ("pre", "Pre-process"), ("infer", "Inference"),
          ("post", "NMS + tracking"), ("output", "Draw / output")]
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]  # validated categorical slots 1-5
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e4e3de"


def load():
    latest = {}
    with (OUT / "realtime_results.csv").open() as f:
        for row in csv.DictReader(f):
            latest[row["label"]] = row  # keep insertion order of first appearance
    return list(latest.values())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--board", help="board name to show (overrides the CSV value)")
    a = ap.parse_args()
    rows = load()
    if not rows:
        raise SystemExit("No results yet. Run realtime/pipeline.py first.")
    board = a.board or rows[0]["board"]

    lines = [
        f"## Real-time pipeline results on {board}",
        "",
        "| Run | Source | Pipeline FPS | Wait | Pre | Infer | NMS+track | Output | "
        "Latency p50 / p95 (ms) | Dropped | Power (W) |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        src = "max speed" if r["source_fps"] == "max" else f"{r['source_fps']} FPS"
        lines.append(
            f"| {r['label']} | {src} | {r['fps']} | {r['wait_ms']} | {r['pre_ms']} | "
            f"{r['infer_ms']} | {r['post_ms']} | {r['output_ms']} | "
            f"{r['e2e_p50_ms']} / {r['e2e_p95_ms']} | "
            f"{'–' if r['source_fps'] == 'max' else r['dropped_pct'] + '%'} | "
            f"{r['avg_power_w'] or '–'} |")
    lines += ["", "Stage columns are mean milliseconds per frame. "
              "\"Max speed\" runs read a video file back-to-back to measure pipeline capacity; "
              "the other runs receive frames at camera speed."]
    (OUT / "realtime_report.md").write_text("\n".join(lines) + "\n")

    # Stacked horizontal bars: where each frame's time goes
    labels = [r["label"] for r in rows][::-1]
    data = {k: [float(r[f"{k}_ms"]) for r in rows][::-1] for k, _ in STAGES}
    fig, ax = plt.subplots(figsize=(8, 0.55 * len(rows) + 1.6), dpi=150)
    left = [0.0] * len(rows)
    for (key, name), color in zip(STAGES, COLORS):
        ax.barh(labels, data[key], left=left, color=color, label=name, height=0.6,
                edgecolor="white", linewidth=2)
        left = [l + v for l, v in zip(left, data[key])]
    fps = [r["fps"] for r in rows][::-1]
    lat = [float(r["e2e_p50_ms"]) for r in rows][::-1]
    for y, (total, f, l) in enumerate(zip(left, fps, lat)):
        ax.text(total + max(left) * 0.01, y, f"{f} FPS · {l:.0f} ms latency",
                va="center", fontsize=8, color=INK)
    ax.set_xlim(0, max(left) * 1.4)
    ax.set_xlabel("Milliseconds per frame (mean)", color=MUTED, fontsize=9)
    ax.set_title(f"Where the time goes: real-time YOLO pipeline on {board}",
                 fontsize=10, color=INK, loc="left")
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.legend(ncol=5, fontsize=7.5, frameon=False, loc="upper left",
              bbox_to_anchor=(0, -0.12 if len(rows) > 3 else -0.3))
    fig.tight_layout()
    fig.savefig(OUT / "stage_breakdown.png")
    print(f"Wrote {OUT / 'realtime_report.md'} and {OUT / 'stage_breakdown.png'}")


if __name__ == "__main__":
    main()
