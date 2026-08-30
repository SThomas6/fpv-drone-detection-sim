#!/usr/bin/env python3
"""Live drone detection on any camera, any laptop. No simulator needed.

`camera/detect_live.py` subscribes to a Gazebo topic, so it only runs where
the simulator runs. This runs the SAME detector and the SAME tracker against
a webcam, a video file, or a folder of frames - which is what you need to
point a real camera at the sky and watch it work.

    python camera/watch.py --source 0                    # built-in webcam
    python camera/watch.py --source drone.mp4            # a video
    python camera/watch.py --source data/clips/eval_birds  # sim frames

Press q to quit, s to save the current frame, SPACE to pause.

On an Apple-silicon Mac the detector runs on the GPU through Metal (MPS)
automatically. Expect roughly 15-30 FPS at 1280x720 on an M-series chip;
if it lags, --imgsz 640 costs a little sensitivity on tiny targets and
roughly doubles the rate.

A caution worth having before you point this at the sky: the model was
trained on simulator imagery plus the Anti-UAV sequences, and a webcam sees
neither. A phone screen playing drone footage, or a small quadcopter at
20-50 m, is a fair test. A speck at 500 m is not - at that range this
project's own measurements say the wide camera cannot resolve it, and no
amount of running it live will change that.
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from camera.detector import DroneDetector  # noqa: E402
from camera.tracking import CentroidTracker  # noqa: E402

C_DET = (60, 60, 255)
C_TRK = (120, 255, 120)
C_TXT = (245, 245, 245)
C_DIM = (150, 150, 150)


def put(img, text, xy, scale=0.55, colour=C_TXT, thick=1):
    cv2.putText(img, text, xy, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0),
                thick + 2, cv2.LINE_AA)
    cv2.putText(img, text, xy, cv2.FONT_HERSHEY_SIMPLEX, scale, colour,
                thick, cv2.LINE_AA)


def frames_from(source: str):
    """Yield frames from a webcam index, a video file, or a frame folder."""
    p = Path(source)
    if p.is_dir():
        files = sorted(list(p.glob("frames/*.png")) or list(p.glob("*.png"))
                       or list(p.glob("*.jpg")))
        if not files:
            raise SystemExit(f"no images in {p}")
        print(f"source: {len(files)} frames from {p}")
        for f in files:
            img = cv2.imread(str(f))
            if img is not None:
                yield img
        return

    cap = cv2.VideoCapture(int(source) if source.isdigit() else source)
    if not cap.isOpened():
        raise SystemExit(
            f"could not open source {source!r}. On macOS the first webcam "
            f"is 0; if this is the first run, grant camera access when the "
            f"prompt appears (or System Settings > Privacy & Security > "
            f"Camera) and try again.")
    print(f"source: {'webcam ' + source if source.isdigit() else source}")
    while True:
        ok, img = cap.read()
        if not ok:
            break
        yield img
    cap.release()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="0",
                    help="webcam index (0), video file, or clip folder")
    ap.add_argument("--conf", type=float, default=0.25,
                    help="detection threshold. The sim benchmarks run at "
                         "0.10, but a real scene has far more to be wrong "
                         "about, so this starts stricter")
    ap.add_argument("--imgsz", type=int, default=None)
    ap.add_argument("--device", default=None, help="mps|cuda|cpu (auto)")
    ap.add_argument("--no-track", action="store_true")
    ap.add_argument("--zoom", type=int, default=6,
                    help="magnification of the inset around the best target")
    ap.add_argument("--record", default=None, help="write an .mp4 here")
    ap.add_argument("--max-width", type=int, default=1280)
    args = ap.parse_args()

    det = DroneDetector(conf=args.conf, device=args.device,
                        **({"imgsz": args.imgsz} if args.imgsz else {}))
    det.warmup()
    print(f"detector ready on {getattr(det, 'device', '?')} "
          f"({Path(det.weights).name})")
    tracker = None if args.no_track else CentroidTracker(
        class_consistent=True, suppress_spawn_near_coasting=True)

    writer = None
    times = deque(maxlen=30)
    paused = False
    saved = 0
    t0 = time.time()
    n = 0

    for frame in frames_from(args.source):
        if frame.shape[1] > args.max_width:
            s = args.max_width / frame.shape[1]
            frame = cv2.resize(frame, (args.max_width, int(frame.shape[0] * s)))
        h, w = frame.shape[:2]
        t = time.time()
        dets = [d for d in det.detect(frame) if d.confidence >= args.conf]
        times.append(time.time() - t)
        n += 1

        best = max(dets, key=lambda d: d.confidence, default=None)
        for d in dets:
            x1, y1, x2, y2 = [int(v) for v in d.xyxy]
            cv2.rectangle(frame, (x1 - 2, y1 - 2), (x2 + 2, y2 + 2), C_DET, 2)
            put(frame, f"{d.cls_name} {d.confidence:.2f}",
                (x1 - 2, y1 - 8), 0.5, C_DET, 1)

        held = 0
        if tracker is not None:
            for tr in tracker.update(dets, timestamp=time.time() - t0):
                if tr.misses:
                    continue
                x1, y1, x2, y2 = [int(v) for v in tr.box]
                cv2.rectangle(frame, (x1 - 6, y1 - 6), (x2 + 6, y2 + 6),
                              C_TRK, 2)
                put(frame, f"T{tr.track_id}", (x1 - 6, y2 + 20), 0.55,
                    C_TRK, 1)
                held += 1

        # magnified inset, so a target only a few pixels across is visible
        if best is not None and args.zoom > 1:
            cx = int((best.xyxy[0] + best.xyxy[2]) / 2)
            cy = int((best.xyxy[1] + best.xyxy[3]) / 2)
            half = 200 // (2 * args.zoom)
            x0 = int(np.clip(cx - half, 0, w - 2 * half))
            y0 = int(np.clip(cy - half, 0, h - 2 * half))
            crop = frame[y0:y0 + 2 * half, x0:x0 + 2 * half]
            if crop.size:
                ins = cv2.resize(crop, (200, 200),
                                 interpolation=cv2.INTER_NEAREST)
                frame[8:208, w - 208:w - 8] = ins
                cv2.rectangle(frame, (w - 208, 8), (w - 8, 208), C_DIM, 1)
                put(frame, f"x{args.zoom}", (w - 204, 226), 0.45, C_DIM)

        fps = 1.0 / (np.mean(times) + 1e-9)
        bar = np.full((44, w, 3), 22, np.uint8)
        put(bar, f"{fps:4.1f} FPS", (10, 29), 0.6, C_TXT, 1)
        put(bar, f"detections {len(dets)}", (130, 29), 0.6,
            C_DET if dets else C_DIM, 1)
        if tracker is not None:
            put(bar, f"tracks {held}", (320, 29), 0.6,
                C_TRK if held else C_DIM, 1)
        put(bar, "q quit   s save   SPACE pause", (w - 330, 29), 0.5, C_DIM)
        view = np.vstack([bar, frame])

        if args.record:
            if writer is None:
                writer = cv2.VideoWriter(
                    args.record, cv2.VideoWriter_fourcc(*"mp4v"), 20,
                    (view.shape[1], view.shape[0]))
            writer.write(view)

        cv2.imshow("drone watch", view)
        k = cv2.waitKey(1 if not paused else 0) & 0xFF
        if k == ord("q"):
            break
        if k == ord(" "):
            paused = not paused
        if k == ord("s"):
            cv2.imwrite(f"watch_{saved:03d}.png", view)
            print(f"saved watch_{saved:03d}.png")
            saved += 1

    if writer is not None:
        writer.release()
        print(f"wrote {args.record}")
    cv2.destroyAllWindows()
    print(f"{n} frames, mean {1.0/(np.mean(times)+1e-9):.1f} FPS")


if __name__ == "__main__":
    main()
