"""Turn pathology/results/pathology_results.csv into a report table and the
deployability chart (throughput vs out-of-distribution AUC).

Usage: python3 pathology/make_pathology_report.py [--board "Jetson Orin Nano Super"]
Uses the most recent row for each backbone x format.
"""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

OUT = Path(__file__).resolve().parent / "results"
ORDER = ["resnet18", "vit_tiny", "phikon", "conch", "uni", "virchow"]
COLORS = dict(zip(ORDER, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]))
FORMATS = ["PyTorch FP32", "PyTorch FP16", "TensorRT FP16", "TensorRT INT8"]
MARKERS = {"PyTorch FP32": "o", "PyTorch FP16": "s", "TensorRT FP16": "^", "TensorRT INT8": "D"}
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e4e3de"


def load():
    latest = {}
    with (OUT / "pathology_results.csv").open() as f:
        for row in csv.DictReader(f):
            latest[(row["backbone"], row["format"])] = row
    return sorted(latest.values(), key=lambda r: (ORDER.index(r["backbone"]),
                                                  FORMATS.index(r["format"])))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--board", help="board name to show (overrides the CSV value)")
    a = ap.parse_args()
    rows = load()
    if not rows:
        raise SystemExit("No results yet. Run pathology/run_pathology.py first.")
    board = a.board or rows[0]["board"]
    tps = rows[0]

    lines = [
        f"## Pathology backbones on {board}",
        "",
        f"CAMELYON17-WILDS tumor vs normal. Linear probe trained on FP32 features of "
        f"{tps['n_train']} id_train tiles, tested on {tps['n_test']} id_val (in-distribution) and "
        f"{tps['n_test']} ood_test (unseen hospital) tiles. Throughput at batch {tps['batch']}.",
        "",
        "| Backbone | Format | Params (M) | Input | Tiles/s | Latency b1 (ms) | Power (W) | "
        "Energy / tile (mJ) | Min / WSI* | Cosine vs FP32 | AUC ID | AUC OOD | Δ AUC OOD [95% CI] |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        d = "—" if r["format"] == "PyTorch FP32" else \
            f"{float(r['d_auc_ood']):+.3f} [{float(r['d_auc_ood_ci_lo']):+.3f}, {float(r['d_auc_ood_ci_hi']):+.3f}]"
        lines.append(
            f"| {r['label']} | {r['format']} | {r['params_m']} | {r['input_px']} | {r['tiles_per_s']} | "
            f"{r['lat_b1_ms']} | {r['avg_power_w'] or '–'} | {r['mj_per_tile'] or '–'} | "
            f"{r['min_per_slide']} | {float(r['cos_vs_fp32']):.4f} | {float(r['auc_id']):.3f} | "
            f"{float(r['auc_ood']):.3f} | {d} |")
    lines += ["", "*Minutes per whole-slide image, assuming 10,000 tissue tiles per slide "
              "(model time only; tile reading and decoding not included)."]
    (OUT / "pathology_report.md").write_text("\n".join(lines) + "\n")

    # Chart: deployability (throughput) vs robustness (OOD AUC)
    fig, ax = plt.subplots(figsize=(9.5, 5), dpi=150)
    for bb in ORDER:
        pts = [r for r in rows if r["backbone"] == bb]
        if not pts:
            continue
        xs = [float(r["tiles_per_s"]) for r in pts]
        ys = [float(r["auc_ood"]) for r in pts]
        ax.plot(xs, ys, color=COLORS[bb], linewidth=1.2, alpha=0.6, zorder=1)
        for r, x, y in zip(pts, xs, ys):
            ax.scatter(x, y, s=64, marker=MARKERS[r["format"]], color=COLORS[bb],
                       edgecolor="white", linewidth=1.5, zorder=3)
    ax.set_xscale("log")
    ax.set_xlabel(f"Throughput on {board} (tiles/s, log scale)", color=MUTED, fontsize=9)
    ax.set_ylabel("AUC on unseen hospital (OOD)", color=MUTED, fontsize=9)
    ax.set_title("Pathology backbones: edge speed vs out-of-distribution accuracy",
                 fontsize=10, color=INK, loc="left")
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    labels = {r["backbone"]: r["label"] for r in rows}
    bb_handles = [Line2D([], [], color=COLORS[b], linewidth=3, label=labels[b])
                  for b in ORDER if b in labels]
    present = [f for f in FORMATS if any(r["format"] == f for r in rows)]
    fmt_handles = [Line2D([], [], marker=MARKERS[f], linestyle="", color=MUTED, markersize=7,
                          label=f) for f in present]
    leg = ax.legend(handles=bb_handles, title="Backbone", fontsize=7.5, title_fontsize=8,
                    frameon=False, loc="upper left", bbox_to_anchor=(1.02, 1.0))
    ax.add_artist(leg)
    ax.legend(handles=fmt_handles, title="Format", fontsize=7.5, title_fontsize=8,
              frameon=False, loc="lower left", bbox_to_anchor=(1.02, 0.0))
    xmin, xmax = ax.get_xlim()
    ax.set_xlim(xmin / 1.6, xmax * 1.2)
    fig.tight_layout()
    fig.savefig(OUT / "speed_vs_ood_auc.png", bbox_inches="tight")
    print(f"Wrote {OUT / 'pathology_report.md'} and {OUT / 'speed_vs_ood_auc.png'}")


if __name__ == "__main__":
    main()
