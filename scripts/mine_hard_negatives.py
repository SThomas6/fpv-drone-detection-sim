#!/usr/bin/env python3
"""Mine hard-negative tiles: run the detector over clips and harvest every
false positive as a background training tile.

The clip labels say where the drone actually is; any detection elsewhere is a
confirmed mistake, and a 640px tile around it becomes a negative example with
an empty label file — no manual review needed.

    python scripts/mine_hard_negatives.py --clips data/clips/realtrain_rgb_* \
        --out data/finetune_hardneg --conf 0.10
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from camera.detector import DroneDetector  # noqa: E402
from camera.evaluate import is_hit  # noqa: E402

TILE = 640


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", nargs="+", required=True)
    ap.add_argument("--out", default="data/finetune_hardneg")
    ap.add_argument("--conf", type=float, default=0.10)
    ap.add_argument("--max-per-clip", type=int, default=60)
    args = ap.parse_args()

    clips = []
    for pat in args.clips:
        clips += [Path(p) for p in glob.glob(pat)]
    out_img = Path(args.out) / "images" / "train"
    out_lab = Path(args.out) / "labels" / "train"
    out_img.mkdir(parents=True, exist_ok=True)
    out_lab.mkdir(parents=True, exist_ok=True)

    det = DroneDetector(conf=args.conf)
    det.warmup()
    total = 0
    import numpy as np
    for clip in clips:
        labels = {r["frame"]: r for r in
                  (json.loads(l) for l in open(clip / "labels.jsonl"))}
        mined = 0
        for fname, gt in sorted(labels.items()):
            if mined >= args.max_per_clip:
                break
            frame = np.array(Image.open(clip / "frames" / fname).convert("RGB"))
            fps = [d for d in det.infer(frame)
                   if d.cls_name == "drone"
                   and not (gt["visible"] and is_hit(d.xyxy, gt))]
            for d in fps[:2]:
                cx, cy = d.centre
                h, w = frame.shape[:2]
                x0 = int(min(max(cx - TILE / 2, 0), w - TILE))
                y0 = int(min(max(cy - TILE / 2, 0), h - TILE))
                # skip tiles that would contain the real drone
                if gt["visible"] and gt.get("bbox"):
                    bx = (gt["bbox"][0] + gt["bbox"][2]) / 2
                    by = (gt["bbox"][1] + gt["bbox"][3]) / 2
                    if x0 <= bx <= x0 + TILE and y0 <= by <= y0 + TILE:
                        continue
                name = f"hn_{clip.name}_{fname[:-4]}_{int(cx)}"
                Image.fromarray(frame[y0:y0 + TILE, x0:x0 + TILE]).save(
                    out_img / f"{name}.png")
                (out_lab / f"{name}.txt").write_text("")
                mined += 1
                total += 1
        print(f"  {clip.name}: {mined} negatives", flush=True)
    print(f"mined {total} hard-negative tiles into {args.out}")


if __name__ == "__main__":
    main()
