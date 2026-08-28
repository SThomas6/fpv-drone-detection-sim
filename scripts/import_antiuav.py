#!/usr/bin/env python3
"""Convert Anti-UAV sequences (real footage) into this project's clip format.

Anti-UAV300 sequences ship as RGB.mp4 + IR.mp4 with JSON labels
({"exist": [...0/1...], "gt_rect": [[x, y, w, h], ...]}). This writes our
standard clip layout (frames/*.png + labels.jsonl + meta.json) so
camera/evaluate.py and the track-level tooling run on real data unchanged.

Honesty notes baked in:
  · range_m is a PSEUDO-RANGE estimated from box width assuming a 0.34 m
    airframe and our camera's focal constant — real range is unknown; the
    field exists so the analyzer's range table keeps working, and meta.json
    records that it is estimated.
  · Real labels are hand-annotated and imperfect (unlike the simulator's).

    python scripts/import_antiuav.py --root data/external/anti_uav/unpacked \
        --modality RGB --limit 12 --fps 5 --out-prefix data/clips/real_rgb_
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from simulator.station_camera import FX, size_bucket  # noqa: E402

DRONE_SIZE_M = 0.34


def import_sequence(seq_dir: Path, modality: str, out: Path, fps: float):
    stem = "visible" if modality == "RGB" else "infrared"
    video = seq_dir / f"{stem}.mp4"
    label = seq_dir / f"{stem}.json"
    if not video.exists() or not label.exists():
        return None

    lab = json.loads(label.read_text())
    exist = lab.get("exist", [])
    rects = lab.get("gt_rect", [])

    cap = cv2.VideoCapture(str(video))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    step = max(1, round(src_fps / fps))
    (out / "frames").mkdir(parents=True, exist_ok=True)

    kept = 0
    with open(out / "labels.jsonl", "w") as fh:
        idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if idx % step:
                idx += 1
                continue
            name = f"{kept:06d}.png"
            cv2.imwrite(str(out / "frames" / name), frame)
            rec = {"frame": name, "t": round(idx / src_fps, 3),
                   "pos": None, "quat": None, "birds": []}
            vis = idx < len(exist) and exist[idx] == 1 and idx < len(rects)
            r = rects[idx] if vis else None
            if vis and r and r[2] > 0 and r[3] > 0:
                x, y, w, h = r
                rec.update({
                    "bbox": [x, y, x + w, y + h],
                    "centre": [x + w / 2, y + h / 2],
                    "px_width": w,
                    "bucket": size_bucket(w),
                    # pseudo-range from box width; see module docstring
                    "range_m": round(FX * DRONE_SIZE_M / max(w, 0.5), 1),
                    "visible": True,
                })
            else:
                rec.update({"bbox": None, "visible": False, "range_m": None})
            fh.write(json.dumps(rec) + "\n")
            kept += 1
            idx += 1
    cap.release()

    (out / "meta.json").write_text(json.dumps({
        "world": f"REAL:anti-uav:{seq_dir.name}", "source_fps": src_fps,
        "interval_s": step / src_fps, "frames": kept,
        "range_m_is_estimated": True,
        "note": f"Anti-UAV {modality} sequence {seq_dir.name}; "
                "hand-annotated labels; pseudo-range from box width",
    }, indent=2))
    return kept


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True,
                    help="directory containing Anti-UAV sequence folders")
    ap.add_argument("--modality", default="RGB", choices=["RGB", "IR"])
    ap.add_argument("--limit", type=int, default=12)
    ap.add_argument("--fps", type=float, default=5.0)
    ap.add_argument("--out-prefix", default="data/clips/real_rgb_")
    args = ap.parse_args()

    seqs = sorted(p for p in Path(args.root).iterdir() if p.is_dir())
    done = 0
    for seq in seqs:
        if done >= args.limit:
            break
        out = Path(f"{args.out_prefix}{seq.name}")
        n = import_sequence(seq, args.modality, out, args.fps)
        if n:
            print(f"  {seq.name}: {n} frames -> {out}", flush=True)
            done += 1
    print(f"imported {done} sequences")


if __name__ == "__main__":
    main()
