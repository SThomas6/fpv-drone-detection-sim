#!/usr/bin/env python3
"""Render the system working, as video, so it can be judged by eye.

Numbers in a table are not evidence you can see. This replays a captured
clip through the SAME fusion the benchmark uses - cached per-sensor streams
into CentroidTracker - and draws what each channel contributes, frame by
frame, with the true range printed alongside.

Layout: the wide camera on the left (the drone is a handful of pixels there,
which is itself the point), a magnified inset on the right so a 3 px target
is actually visible, the thermal band below it, and a HUD naming which
sensors fired this frame and whether the tracker is holding the target.

The inset is centred on GROUND TRUTH, not on a detection - otherwise a frame
where everything missed would show an empty crop and look like a frame where
nothing was there. Centring on truth means a miss looks like a miss.

    python scripts/render_demo.py --clip data/clips/range_1km_hr --out demo.mp4
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from camera.detector import Detection, merge_close  # noqa: E402
from camera.evaluate import is_hit  # noqa: E402
from camera.fuse_eval import fuse_measurements  # noqa: E402
from camera.tracking import CentroidTracker  # noqa: E402

# BGR
C_GT = (80, 255, 80)
C_RGB = (60, 60, 255)
C_IR = (0, 165, 255)
C_MOV = (255, 255, 0)
C_PT = (255, 0, 255)
C_TRK_OK = (120, 255, 120)
C_TRK_BAD = (80, 80, 255)
C_TXT = (240, 240, 240)
C_DIM = (150, 150, 150)

W, H = 1280, 700
VIEW_W, VIEW_H = 880, 495
INSET = 376


def load(clip: Path, name: str):
    p = clip / f"detections_{name}.jsonl"
    if not p.exists():
        return {}
    return {r["frame"]: r["detections"]
            for r in (json.loads(l) for l in open(p))}


def put(img, text, xy, scale=0.5, colour=C_TXT, thick=1):
    cv2.putText(img, text, xy, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0),
                thick + 2, cv2.LINE_AA)
    cv2.putText(img, text, xy, cv2.FONT_HERSHEY_SIMPLEX, scale, colour,
                thick, cv2.LINE_AA)


def thermal_view(path: Path, size):
    """L16 kelvin counts -> a false-colour image a human can read."""
    if not path.exists():
        return np.zeros((size[1], size[0], 3), np.uint8)
    raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if raw is None:
        return np.zeros((size[1], size[0], 3), np.uint8)
    raw = raw.astype(np.float32)
    lo, hi = np.percentile(raw, 1), np.percentile(raw, 99.9)
    norm = np.clip((raw - lo) / max(hi - lo, 1e-6), 0, 1)
    img = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
    return cv2.resize(img, size, interpolation=cv2.INTER_NEAREST)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--step", type=int, default=3,
                    help="render every Nth frame; the TRACKER still runs on "
                         "every frame, because skipping frames would change "
                         "the very behaviour being shown")
    ap.add_argument("--fps", type=int, default=25)
    ap.add_argument("--conf", type=float, default=0.10)
    ap.add_argument("--zoom", type=int, default=8)
    ap.add_argument("--title", default=None)
    args = ap.parse_args()

    clip = Path(args.clip)
    labels = [json.loads(l) for l in open(clip / "labels.jsonl")]
    meta = json.loads((clip / "meta.json").read_text())
    streams = {n: load(clip, n)
               for n in ("full", "sahi", "ir", "motion", "point", "acoustic")}
    have = [n for n, v in streams.items() if v]
    print(f"{clip.name}: {len(labels)} frames, streams = {', '.join(have)}")

    # --- pass 1: replay the tracker over EVERY frame -----------------------
    tracker = CentroidTracker(class_consistent=True,
                              suppress_spawn_near_coasting=True)
    tracks_by_frame = {}
    for rec in labels:
        f = rec["frame"]
        rgb = [Detection(*d["xyxy"], confidence=d["conf"],
                         cls_name=d.get("cls", "drone"))
               for d in streams["full"].get(f, []) if d["conf"] >= 0.05]
        aux = [Detection(*d["xyxy"], confidence=d["conf"], cls_name="hotspot")
               for d in streams["ir"].get(f, []) if d["conf"] >= 0.2]
        aux += [Detection(*d["xyxy"], confidence=d["conf"], cls_name="mover")
                for d in streams["motion"].get(f, []) if d["conf"] >= 0.10]
        aux += [Detection(*d["xyxy"], confidence=d["conf"], cls_name="point")
                for d in streams["point"].get(f, []) if d["conf"] >= 0.0]
        dets, _ = fuse_measurements(merge_close(rgb), aux)
        tracks_by_frame[f] = [(t.track_id, tuple(t.box), t.coasting)
                              for t in
                              tracker.update(dets, timestamp=rec["t"])]

    # --- pass 2: draw ------------------------------------------------------
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(args.out, fourcc, args.fps, (W, H))
    if not vw.isOpened():
        raise SystemExit(f"could not open {args.out} for writing")
    title = args.title or clip.name
    n_drawn = 0
    for rec in labels[::args.step]:
        f = rec["frame"]
        frame = cv2.imread(str(clip / "frames" / f))
        if frame is None:
            continue
        fh, fw = frame.shape[:2]
        canvas = np.full((H, W, 3), 18, np.uint8)

        gx = (rec["bbox"][0] + rec["bbox"][2]) / 2
        gy = (rec["bbox"][1] + rec["bbox"][3]) / 2
        gw = rec["bbox"][2] - rec["bbox"][0]
        vis = rec.get("visible")

        # ---- magnified inset, centred on truth ----
        half = INSET // (2 * args.zoom)
        x0 = int(np.clip(gx - half, 0, fw - 2 * half))
        y0 = int(np.clip(gy - half, 0, fh - 2 * half))
        inset = frame[y0:y0 + 2 * half, x0:x0 + 2 * half]
        inset = cv2.resize(inset, (INSET, INSET), interpolation=cv2.INTER_NEAREST)

        def to_inset(px, py):
            return int((px - x0) * args.zoom), int((py - y0) * args.zoom)

        # ---- draw detections on BOTH the wide view and the inset ----
        fired = []
        for name, colour, thr, tag in (
                ("full", C_RGB, args.conf, "RGB"),
                ("sahi", C_RGB, args.conf, "tiled"),
                ("ir", C_IR, 0.2, "thermal"),
                ("motion", C_MOV, 0.10, "motion"),
                ("point", C_PT, 0.0, "point")):
            hit_any = False
            for d in streams[name].get(f, []):
                if d["conf"] < thr:
                    continue
                x1, y1, x2, y2 = [int(v) for v in d["xyxy"]]
                cv2.rectangle(frame, (x1 - 3, y1 - 3), (x2 + 3, y2 + 3),
                              colour, 1)
                a, b = to_inset(d["xyxy"][0], d["xyxy"][1])
                c, e = to_inset(d["xyxy"][2], d["xyxy"][3])
                cv2.rectangle(inset, (a - 4, b - 4), (c + 4, e + 4), colour, 2)
                if vis and is_hit(d["xyxy"], rec):
                    hit_any = True
            if hit_any:
                fired.append((tag, colour))

        # ---- ground truth, drawn last so it is never hidden ----
        if vis:
            cv2.rectangle(frame,
                          (int(rec["bbox"][0]) - 6, int(rec["bbox"][1]) - 6),
                          (int(rec["bbox"][2]) + 6, int(rec["bbox"][3]) + 6),
                          C_GT, 1)
            a, b = to_inset(rec["bbox"][0], rec["bbox"][1])
            c, e = to_inset(rec["bbox"][2], rec["bbox"][3])
            cv2.rectangle(inset, (a - 6, b - 6), (c + 6, e + 6), C_GT, 1)

        # ---- tracker ----
        held = False
        for tid, box, coasting in tracks_by_frame.get(f, []):
            ok = vis and is_hit(box, rec)
            held = held or ok
            col = C_TRK_OK if ok else C_TRK_BAD
            x1, y1, x2, y2 = [int(v) for v in box]
            cv2.rectangle(frame, (x1 - 8, y1 - 8), (x2 + 8, y2 + 8), col, 2)
            put(frame, f"T{tid}" + ("~" if coasting else ""),
                (x1 - 8, y1 - 14), 0.5, col, 1)

        view = cv2.resize(frame, (VIEW_W, VIEW_H), interpolation=cv2.INTER_AREA)
        canvas[8:8 + VIEW_H, 8:8 + VIEW_W] = view
        canvas[8:8 + INSET, 896:896 + INSET] = inset
        cv2.rectangle(canvas, (896, 8), (896 + INSET, 8 + INSET), C_DIM, 1)
        canvas[392:392 + 300, 896:896 + 376] = thermal_view(
            clip / "frames_ir" / f, (376, 300))
        cv2.rectangle(canvas, (896, 392), (896 + 376, 392 + 300), C_DIM, 1)

        put(canvas, title, (12, 528), 0.62, C_TXT, 1)
        rng = rec.get("range_m")
        put(canvas, f"RANGE {rng:.0f} m" if rng else "RANGE  -",
            (12, 566), 0.95, (120, 220, 255), 2)
        put(canvas, f"target {gw:.1f} px wide", (330, 566), 0.6, C_DIM, 1)
        put(canvas, f"x{args.zoom} magnified view", (900, 400 - 8), 0.45, C_DIM)
        put(canvas, "thermal (LWIR)", (900, 392 + 316), 0.45, C_DIM)

        if fired:
            put(canvas, "SEEN BY:", (12, 600), 0.55, C_DIM)
            x = 110
            for tag, colour in fired:
                put(canvas, tag, (x, 600), 0.62, colour, 2)
                x += 22 + 14 * len(tag)
        else:
            put(canvas, "SEEN BY:  nothing this frame", (12, 600), 0.55,
                (90, 90, 200))
        status = ("TRACKING" if held else
                  ("COASTING / LOST" if tracks_by_frame.get(f) else "NO TRACK"))
        put(canvas, status, (12, 640), 0.8,
            C_TRK_OK if held else (80, 80, 255), 2)
        put(canvas, "green = truth   red = RGB   orange = thermal   "
                    "cyan = motion   magenta = point", (12, 674), 0.45, C_DIM)

        vw.write(canvas)
        n_drawn += 1
        if n_drawn % 100 == 0:
            print(f"  {n_drawn} frames rendered", flush=True)
    vw.release()
    print(f"wrote {args.out} ({n_drawn} frames, {n_drawn / args.fps:.0f} s)")


if __name__ == "__main__":
    main()
