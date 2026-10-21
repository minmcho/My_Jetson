"""Export a YOLO model to ONNX, TensorRT FP16 and TensorRT INT8.

Usage: python3 scripts/export.py --model yolo11n.pt [--imgsz 640] [--calib coco128.yaml]
Outputs go to models/<stem>_<format>.<ext>; existing files are skipped.
"""
import argparse
import shutil
from pathlib import Path

from ultralytics import YOLO


def export_one(weights, out_path, **kwargs):
    if out_path.exists():
        print(f"[skip] {out_path} already exists")
        return
    print(f"[export] {out_path.name}  {kwargs}")
    produced = YOLO(weights).export(**kwargs)
    shutil.move(str(produced), out_path)
    print(f"[done] {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="yolo11n.pt")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--calib", default="coco128.yaml",
                    help="dataset yaml used for INT8 calibration")
    a = ap.parse_args()

    out = Path("models")
    out.mkdir(exist_ok=True)
    weights = out / Path(a.model).name
    if not weights.exists():
        # Ultralytics downloads official weights on first use
        YOLO(a.model)
        shutil.copy(Path(a.model).name, weights)
    stem = weights.stem
    common = dict(imgsz=a.imgsz, device=0)

    export_one(weights, out / f"{stem}_fp32.onnx", format="onnx", simplify=True, **common)
    export_one(weights, out / f"{stem}_fp16.engine", format="engine", half=True, **common)
    export_one(weights, out / f"{stem}_int8.engine", format="engine", int8=True,
               data=a.calib, **common)


if __name__ == "__main__":
    main()
