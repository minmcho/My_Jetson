"""Real-time detection (+ optional tracking) pipeline with per-stage timing.

Runs live video through a YOLO model and measures, frame by frame:
  wait     time spent waiting for the next frame from the camera / video
  pre      resize + normalize (Ultralytics preprocess)
  infer    model forward pass (PyTorch or TensorRT)
  post     NMS + tracking (ByteTrack when --track is on)
  output   drawing boxes, writing video, showing the window
  e2e      end-to-end latency: frame handed to the app -> result ready

Examples (inside the Ultralytics Jetson container, from the repo root):
  python3 realtime/pipeline.py --source 0 --model models/yolo11n_fp16.engine
  python3 realtime/pipeline.py --source traffic.mp4 --model models/yolo11n_int8.engine --track
  python3 realtime/pipeline.py --source traffic.mp4 --model models/yolo11n.pt --sync --save out.mp4

One summary row per run is appended to realtime/results/realtime_results.csv.
"""
import argparse
import csv
import statistics
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import cv2
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from benchmark import PowerSampler, board_info  # noqa: E402

STAGES = ["wait", "pre", "infer", "post", "output"]


# ---------------------------------------------------------------- sources
def open_capture(source, width, height, fps):
    if source == "csi":
        # Jetson CSI camera (e.g. IMX219). Needs OpenCV built with GStreamer and,
        # in Docker, -v /tmp/argus_socket:/tmp/argus_socket.
        gst = (f"nvarguscamerasrc ! video/x-raw(memory:NVMM),width={width},height={height},"
               f"framerate={fps}/1 ! nvvidconv ! video/x-raw,format=BGRx ! videoconvert ! "
               f"video/x-raw,format=BGR ! appsink drop=1 max-buffers=1")
        cap = cv2.VideoCapture(gst, cv2.CAP_GSTREAMER)
    elif source.isdigit():  # USB webcam index
        cap = cv2.VideoCapture(int(source), cv2.CAP_V4L2)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_FPS, fps)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    else:  # video file, RTSP or HTTP stream
        cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise SystemExit(f"Could not open source: {source}")
    return cap


def is_file(source):
    return Path(source).is_file()


class SyncReader:
    """Naive approach: read the next frame only after the previous one is processed.
    A camera keeps a small queue of frames (V4L2 typically holds ~4), so when the
    model is slower than the camera you get stale frames: latency grows by up to a
    full buffer. For video files this camera behaviour is simulated."""

    CAMERA_BUFFER = 4

    def __init__(self, cap, source, loop, src_fps):
        self.cap, self.loop = cap, loop
        self.pace = 1.0 / src_fps if (is_file(source) and src_fps) else 0.0
        self.t0, self.i = None, 0
        self.dropped = 0

    def _read_raw(self):
        ok, frame = self.cap.read()
        if not ok and self.loop:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = self.cap.read()
        return frame if ok else None

    def read(self):
        if not self.pace:  # live camera / stream, or --max-speed
            frame = self._read_raw()
            return (frame, time.perf_counter()) if frame is not None else (None, None)
        now = time.perf_counter()
        if self.t0 is None:
            self.t0 = now
        newest = int((now - self.t0) / self.pace)  # newest frame the "camera" has produced
        if newest - self.i >= self.CAMERA_BUFFER:  # buffer full: oldest frames overwritten
            skip = newest - self.CAMERA_BUFFER + 1 - self.i
            for _ in range(skip):
                self.cap.grab()
            self.dropped += skip
            self.i += skip
        t_frame = self.t0 + self.i * self.pace  # when this frame was "captured"
        if t_frame > now:
            time.sleep(t_frame - now)
        frame = self._read_raw()
        self.i += 1
        return (frame, t_frame) if frame is not None else (None, None)

    def stop(self):
        pass


class ThreadedReader:
    """Real-time approach: a background thread grabs frames continuously and keeps
    only the newest one. The model always works on the freshest frame; frames it
    can't keep up with are dropped (and counted) instead of piling up as latency."""

    def __init__(self, cap, source, loop, src_fps):
        self.cap, self.loop = cap, loop
        self.pace = 1.0 / src_fps if (is_file(source) and src_fps) else 0.0
        self.cond = threading.Condition()
        self.frame, self.t, self.seq, self.taken = None, None, 0, 0
        self.dropped = 0
        self.running = True
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        next_t = time.perf_counter()
        while self.running:
            if self.pace:
                delay = next_t - time.perf_counter()
                if delay > 0:
                    time.sleep(delay)
                next_t += self.pace
            ok, frame = self.cap.read()
            if not ok and self.loop:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ok, frame = self.cap.read()
            with self.cond:
                if not ok:
                    self.running = False
                    self.cond.notify_all()
                    return
                if self.frame is not None and self.taken < self.seq:
                    self.dropped += 1  # previous frame was never processed
                self.frame, self.t = frame, time.perf_counter()
                self.seq += 1
                self.cond.notify_all()

    def read(self):
        with self.cond:
            while self.running and self.taken >= self.seq:
                self.cond.wait(timeout=1.0)
            if self.taken >= self.seq:
                return None, None
            self.taken = self.seq
            return self.frame, self.t

    def stop(self):
        self.running = False
        self.thread.join(timeout=2)


# ---------------------------------------------------------------- helpers
def pct(values, p):
    s = sorted(values)
    return s[min(len(s) - 1, int(p / 100 * len(s)))]


def draw_hud(img, fps, e2e_ms):
    text = f"{fps:5.1f} FPS | {e2e_ms:5.1f} ms latency"
    cv2.rectangle(img, (0, 0), (360, 32), (0, 0, 0), -1)
    cv2.putText(img, text, (8, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--source", required=True,
                    help="0 (USB webcam), csi (Jetson CSI camera), video file, or rtsp:// URL")
    ap.add_argument("--model", default="models/yolo11n_fp16.engine")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--track", action="store_true", help="add ByteTrack multi-object tracking")
    ap.add_argument("--sync", action="store_true",
                    help="use the naive synchronous reader instead of the threaded one")
    ap.add_argument("--frames", type=int, default=600, help="frames to measure (after warm-up)")
    ap.add_argument("--warmup", type=int, default=30)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--fps", type=int, default=30, help="camera FPS (files use their own FPS)")
    ap.add_argument("--max-speed", action="store_true",
                    help="read video files as fast as possible (measures pipeline capacity)\n"
                         "instead of playing them at real-time speed like a camera")
    ap.add_argument("--no-loop", action="store_true", help="stop at the end of a video file")
    ap.add_argument("--show", action="store_true", help="show a window (needs a display)")
    ap.add_argument("--save", help="write annotated video to this .mp4 path")
    ap.add_argument("--label", help="name for this run in the results table")
    a = ap.parse_args()

    cap = open_capture(a.source, a.width, a.height, a.fps)
    src_fps = cap.get(cv2.CAP_PROP_FPS) or a.fps
    if src_fps <= 0 or src_fps > 240:
        src_fps = a.fps
    # Capacity runs read frames back-to-back; real-time runs use the threaded reader
    # unless --sync asks for the naive one.
    Reader = SyncReader if (a.sync or a.max_speed) else ThreadedReader
    reader = Reader(cap, a.source, loop=not a.no_loop, src_fps=0 if a.max_speed else src_fps)

    model = YOLO(a.model, task="detect")
    model_name = Path(a.model).name
    label = a.label or (f"{model_name}" + (" +track" if a.track else "")
                        + (" (sync)" if a.sync else "") + (" max" if a.max_speed else ""))
    pace = "max speed" if a.max_speed else f"{src_fps:.0f} FPS"
    print(f"[run] {label} | source={a.source} @ {pace} | reader={Reader.__name__}")

    writer = None
    draw = a.show or a.save
    rows = {k: [] for k in STAGES + ["e2e"]}
    total = a.warmup + a.frames
    n = 0
    fps_ema = 0.0
    power = PowerSampler()
    t_last = time.perf_counter()
    t_start = None

    try:
        while n < total:
            t_wait0 = time.perf_counter()
            frame, t_cap = reader.read()
            if frame is None:
                print("[info] source ended")
                break
            t0 = time.perf_counter()

            if a.track:
                res = model.track(frame, imgsz=a.imgsz, persist=True, verbose=False,
                                  tracker="bytetrack.yaml")[0]
            else:
                res = model.predict(frame, imgsz=a.imgsz, verbose=False)[0]
            t1 = time.perf_counter()

            now = time.perf_counter()
            inst = 1.0 / max(now - t_last, 1e-6)
            fps_ema = inst if fps_ema == 0 else 0.9 * fps_ema + 0.1 * inst
            t_last = now

            if draw:
                img = res.plot()
                draw_hud(img, fps_ema, (t1 - t_cap) * 1000)
                if a.save:
                    if writer is None:
                        h, w = img.shape[:2]
                        writer = cv2.VideoWriter(a.save, cv2.VideoWriter_fourcc(*"mp4v"),
                                                 src_fps, (w, h))
                    writer.write(img)
                if a.show:
                    cv2.imshow("realtime", img)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
            t2 = time.perf_counter()

            n += 1
            if n == a.warmup:
                t_start = time.perf_counter()
                reader.dropped = 0
                power.start()
            if n > a.warmup:
                sp = res.speed  # ms: preprocess / inference / postprocess
                model_ms = (t1 - t0) * 1000
                pre, inf = sp.get("preprocess", 0.0), sp.get("inference", 0.0)
                rows["wait"].append((t0 - t_wait0) * 1000)
                rows["pre"].append(pre)
                rows["infer"].append(inf)
                rows["post"].append(max(model_ms - pre - inf, 0.0))  # NMS + tracking + overhead
                rows["output"].append((t2 - t1) * 1000)
                rows["e2e"].append((t2 - t_cap) * 1000)
    except KeyboardInterrupt:
        print("\n[info] stopped by user")
    finally:
        reader.stop()
        avg_w = power.stop()
        cap.release()
        if writer:
            writer.release()
        if a.show:
            cv2.destroyAllWindows()

    measured = len(rows["e2e"])
    if measured < 10:
        raise SystemExit("Not enough frames measured. Use a longer video or loop it.")
    elapsed = time.perf_counter() - t_start
    fps = measured / elapsed
    drop_pct = 100 * reader.dropped / (reader.dropped + measured)

    summary = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "label": label, "model": model_name, "track": a.track,
        "reader": "sync" if Reader is SyncReader else "threaded", "source": a.source,
        "source_fps": "max" if a.max_speed else round(src_fps, 1), "imgsz": a.imgsz, "frames": measured,
        "fps": round(fps, 1),
        **{f"{k}_ms": round(statistics.mean(rows[k]), 2) for k in STAGES},
        "e2e_p50_ms": round(pct(rows["e2e"], 50), 2),
        "e2e_p95_ms": round(pct(rows["e2e"], 95), 2),
        "dropped_pct": round(drop_pct, 1),
        "avg_power_w": avg_w,
        **board_info(),
    }

    out = ROOT / "realtime" / "results"
    out.mkdir(parents=True, exist_ok=True)
    path = out / "realtime_results.csv"
    new = not path.exists()
    with path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary))
        if new:
            w.writeheader()
        w.writerow(summary)

    print(f"\n=== {label} ===")
    print(f"  pipeline FPS : {summary['fps']}  (source {summary['source_fps']} FPS)")
    print("  stage ms     : " + "  ".join(f"{k}={summary[k + '_ms']}" for k in STAGES))
    print(f"  latency      : p50 {summary['e2e_p50_ms']} ms, p95 {summary['e2e_p95_ms']} ms")
    print(f"  dropped      : {summary['dropped_pct']}% of frames")
    print(f"  power        : {avg_w or '-'} W")


if __name__ == "__main__":
    main()
