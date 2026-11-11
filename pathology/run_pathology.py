"""Edge-deployment benchmark for pathology foundation-model backbones.

For each backbone and each format (PyTorch FP32, PyTorch FP16, TensorRT FP16, TensorRT INT8):
  * speed:     latency at batch 1, throughput at batch N (tiles/s)
  * cost:      average board power, energy per tile, minutes per whole-slide image
  * fidelity:  cosine similarity of embeddings vs the FP32 reference
  * accuracy:  a linear probe trained on FP32 features of CAMELYON17-WILDS id_train is
               applied unchanged to each format's features. AUC on id_val (in-distribution)
               and ood_test (unseen hospital), with a paired bootstrap 95% CI for the
               OOD AUC change vs FP32. This answers: "if I train on a workstation and
               deploy a compressed backbone on the edge, does accuracy hold up?"

Usage (inside the Jetson container, from the repo root):
  python3 pathology/run_pathology.py                         # open models only
  HF_TOKEN=hf_xxx python3 pathology/run_pathology.py --backbones uni conch
  python3 pathology/run_pathology.py --data-dir /data/camelyon17   # reuse your downloaded data

Results are appended to pathology/results/pathology_results.csv.
"""
import argparse
import csv
import gc
import os
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "scripts"))

from backbones import BACKBONES, count_params, load_backbone  # noqa: E402
from data import load_tiles  # noqa: E402
import trt_utils  # noqa: E402

try:
    from benchmark import PowerSampler, board_info  # Jetson power + board helpers
except Exception:  # running somewhere without ultralytics / tegrastats
    import platform

    class PowerSampler:
        def start(self):
            pass

        def stop(self):
            return ""

    def board_info():
        name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else platform.processor()
        return {"board": name, "jetpack": "", "power_mode": ""}

FORMATS = ["torch_fp32", "torch_fp16", "trt_fp16", "trt_int8"]
FORMAT_LABEL = {"torch_fp32": "PyTorch FP32", "torch_fp16": "PyTorch FP16",
                "trt_fp16": "TensorRT FP16", "trt_int8": "TensorRT INT8"}


# ---------------------------------------------------------------- helpers
def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def to_batches(images, bs, device):
    for i in range(0, len(images), bs):
        x = torch.from_numpy(images[i:i + bs]).to(device, non_blocking=True)
        yield x.permute(0, 3, 1, 2).float().div_(255.0)


@torch.no_grad()
def embed(fn, images, bs, device, half=False):
    out = []
    for x in to_batches(images, bs, device):
        y = fn(x.half() if half else x)
        out.append(y.float().cpu().numpy())
    return np.concatenate(out)


@torch.no_grad()
def time_fn(fn, x, device, min_iters=20, min_seconds=5.0, warmup=10):
    for _ in range(warmup):
        fn(x)
    sync(device)
    lat, t_end = [], time.perf_counter() + min_seconds
    while len(lat) < min_iters or time.perf_counter() < t_end:
        t0 = time.perf_counter()
        fn(x)
        sync(device)
        lat.append((time.perf_counter() - t0) * 1000)
    lat.sort()
    return statistics.mean(lat), lat[int(0.95 * (len(lat) - 1))]


def auc(y, p):
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y, p))


def paired_bootstrap_delta(y, p_ref, p_new, n_boot=1000, seed=0):
    rng = np.random.RandomState(seed)
    deltas = []
    for _ in range(n_boot):
        i = rng.randint(0, len(y), len(y))
        if y[i].min() == y[i].max():
            continue
        deltas.append(auc(y[i], p_new[i]) - auc(y[i], p_ref[i]))
    return float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5))


def cosine(a, b):
    a = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)
    b = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-12)
    return float((a * b).sum(1).mean())


# ---------------------------------------------------------------- main
def run_backbone(name, a, device, writer, info):
    spec = BACKBONES[name]
    if spec["gated"] and not os.environ.get("HF_TOKEN"):
        print(f"[skip] {name}: gated model, set HF_TOKEN (see pathology/README.md)")
        return
    print(f"\n=== {spec['label']} ===")
    try:
        model, spec = load_backbone(name)
    except Exception as e:  # missing access, package or memory
        print(f"[skip] {name}: could not load ({e})")
        return
    size, bs = spec["size"], a.batch
    params_m = count_params(model) / 1e6

    data = {s: load_tiles(s, a.n_train if s == "id_train" else a.n_test, size,
                          a.data_dir, a.cache_dir) for s in ("id_train", "id_val", "ood_test")}
    (Xtr, ytr), (Xid, yid), (Xood, yood) = data["id_train"], data["id_val"], data["ood_test"]

    fmts = [f for f in a.formats if f in FORMATS]
    use_trt = device.type == "cuda" and trt_utils.trt is not None
    work = Path(a.workdir) / name
    onnx_path = work / f"{name}_fp32.onnx"
    if use_trt and any(f.startswith("trt") for f in fmts):
        print("[onnx] exporting")
        trt_utils.export_onnx(model, size, onnx_path)

    feats, rows = {}, []
    # torch formats first (they need the PyTorch model), then TensorRT (model freed)
    order = sorted(set(["torch_fp32"] + fmts), key=FORMATS.index)
    for fmt in order:
        half = fmt == "torch_fp16"
        if fmt == "torch_fp16" and device.type != "cuda":
            print("[skip] torch_fp16 needs a GPU")
            continue
        if fmt.startswith("trt") and not use_trt:
            print(f"[skip] {fmt}: TensorRT/GPU not available")
            continue
        try:
            if fmt.startswith("torch"):
                model = model.to(device)
                model = model.half() if half else model.float()
                fn = model
                size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / 2**20
            else:
                if model is not None:  # free PyTorch weights before building engines
                    model = None
                gc.collect()
                torch.cuda.empty_cache()
                calib = None
                if fmt == "trt_int8":
                    n_cal = min(len(Xtr), a.calib_batches * bs)
                    calib = list(to_batches(Xtr[:n_cal], bs, device))
                engine = trt_utils.build_engine(onnx_path, work / f"{name}_{fmt}.engine",
                                                fmt.split("_")[1], size, bs, bs, calib,
                                                a.workspace_gb)
                del calib
                fn = trt_utils.TRTRunner(engine)
                size_mb = engine.stat().st_size / 2**20

            x1 = torch.from_numpy(Xtr[:1]).to(device).permute(0, 3, 1, 2).float().div(255)
            xb = torch.from_numpy(Xtr[:bs]).to(device).permute(0, 3, 1, 2).float().div(255)
            if half:
                x1, xb = x1.half(), xb.half()
            print(f"[{fmt}] timing")
            lat1, _ = time_fn(fn, x1, device)
            power = PowerSampler()
            power.start()
            latb, latb_p95 = time_fn(fn, xb, device, min_seconds=a.seconds)
            watts = power.stop()

            print(f"[{fmt}] embedding {len(Xtr) + len(Xid) + len(Xood)} tiles")
            feats[fmt] = {s: embed(fn, X, bs, device, half)
                          for s, X in (("train", Xtr), ("id", Xid), ("ood", Xood))}
        except Exception as e:
            print(f"[skip] {fmt}: {type(e).__name__}: {e}")
            if fmt == "torch_fp32":
                return
            continue

        tps = bs / (latb / 1000)
        rows.append({
            "backbone": name, "label": spec["label"], "format": FORMAT_LABEL[fmt],
            "input_px": size, "embed_dim": feats[fmt]["id"].shape[1],
            "params_m": round(params_m, 1), "model_mb": round(size_mb, 1),
            "batch": bs, "lat_b1_ms": round(lat1, 2), "lat_batch_ms": round(latb, 2),
            "lat_batch_p95_ms": round(latb_p95, 2), "tiles_per_s": round(tps, 1),
            "avg_power_w": watts,
            "mj_per_tile": round(watts / tps * 1000, 2) if watts != "" else "",
            "min_per_slide": round(a.tiles_per_slide / tps / 60, 2),
        })
        del fn
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # ---- accuracy: FP32-trained linear probe applied to every format
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    ref = feats["torch_fp32"]
    scaler = StandardScaler().fit(ref["train"])
    clf = LogisticRegression(max_iter=5000).fit(scaler.transform(ref["train"]), ytr)
    prob = {f: {s: clf.predict_proba(scaler.transform(feats[f][s]))[:, 1] for s in ("id", "ood")}
            for f in feats}
    fmt_key = {FORMAT_LABEL[k]: k for k in FORMAT_LABEL}
    for r in rows:
        f = fmt_key[r["format"]]
        lo, hi = (0.0, 0.0) if f == "torch_fp32" else \
            paired_bootstrap_delta(yood, prob["torch_fp32"]["ood"], prob[f]["ood"])
        r.update({
            "cos_vs_fp32": round(cosine(np.concatenate([feats[f]["id"], feats[f]["ood"]]),
                                        np.concatenate([ref["id"], ref["ood"]])), 5),
            "auc_id": round(auc(yid, prob[f]["id"]), 4),
            "auc_ood": round(auc(yood, prob[f]["ood"]), 4),
            "d_auc_ood": round(auc(yood, prob[f]["ood"]) - auc(yood, prob["torch_fp32"]["ood"]), 4),
            "d_auc_ood_ci_lo": round(lo, 4), "d_auc_ood_ci_hi": round(hi, 4),
            "n_train": len(ytr), "n_test": len(yid),
            "timestamp": datetime.now().isoformat(timespec="seconds"), **info,
        })
        writer(r)
        print(f"  {r['format']:<14} {r['tiles_per_s']:>8} tiles/s  "
              f"b1 {r['lat_b1_ms']:>7} ms  AUC id {r['auc_id']:.3f}  ood {r['auc_ood']:.3f} "
              f"(Δ {r['d_auc_ood']:+.3f})  cos {r['cos_vs_fp32']:.4f}  "
              f"{r['avg_power_w'] or '-'} W")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--backbones", nargs="+", default=list(BACKBONES), choices=list(BACKBONES))
    ap.add_argument("--formats", nargs="+", default=FORMATS, choices=FORMATS)
    ap.add_argument("--n-train", type=int, default=400, help="probe-training / calibration tiles")
    ap.add_argument("--n-test", type=int, default=400, help="tiles per test split (ID and OOD)")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--seconds", type=float, default=8.0, help="duration of each throughput run")
    ap.add_argument("--calib-batches", type=int, default=16, help="INT8 calibration batches")
    ap.add_argument("--tiles-per-slide", type=int, default=10000,
                    help="tissue tiles in a typical WSI, for the minutes-per-slide estimate")
    ap.add_argument("--data-dir", help="folder with CAMELYON17-WILDS parquet files")
    ap.add_argument("--cache-dir", default=str(HERE / "data_cache"))
    ap.add_argument("--workdir", default=str(HERE / "engines"))
    ap.add_argument("--workspace-gb", type=float, default=2.0)
    ap.add_argument("--board", help="board name to record (overrides auto-detect)")
    a = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    info = board_info()
    if a.board:
        info["board"] = a.board
    print(f"[device] {device} | {info}")

    out = HERE / "results"
    out.mkdir(exist_ok=True)
    path = out / "pathology_results.csv"

    def writer(row):
        new = not path.exists()
        with path.open("a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(row))
            if new:
                w.writeheader()
            w.writerow(row)

    for name in a.backbones:
        run_backbone(name, a, device, writer, info)
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    print(f"\nResults: {path}\nNext: python3 pathology/make_pathology_report.py")


if __name__ == "__main__":
    main()
