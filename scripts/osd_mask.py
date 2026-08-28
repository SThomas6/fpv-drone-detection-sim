#!/usr/bin/env python3
"""Detect and neutralise burned-in OSD overlays (reticle, text, timestamp) in
real tracking-camera footage.

The turret pans, so scene pixels change while OSD strokes stay put: a pixel
that is high-contrast in the time-MEAN image but has low temporal variance is
overlay, not world. The mask is dilated and filled with neutral grey (114 —
YOLO's letterbox colour) BEFORE inference, so the detector never sees the
glyphs. Writes a masked copy of the clip so the standard evaluator runs
unchanged; raw and masked numbers must be reported side by side (the mask
estimates deployment on an OSD-free camera; raw measures this dataset).

    python scripts/osd_mask.py --clip data/clips/real_rgb_X --out data/clips/real_rgb_X_masked
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

STD_MAX = 6.0        # max temporal std for an OSD pixel
GRAD_MIN = 18.0      # min spatial gradient (mean image) for an OSD stroke
DILATE = 2
FILL = 114


def build_mask(frames: list[np.ndarray]) -> np.ndarray:
    stack = np.stack([f.astype(np.float32).mean(axis=2) for f in frames])
    mean = stack.mean(axis=0)
    std = stack.std(axis=0)
    gy, gx = np.gradient(mean)
    grad = np.hypot(gx, gy)
    mask = (std < STD_MAX) & (grad > GRAD_MIN)
    # timestamp digits change every second: catch them via a static-text
    # neighbourhood — dilate pulls in the strokes around them anyway
    for _ in range(DILATE):
        m = mask.copy()
        m[1:, :] |= mask[:-1, :]
        m[:-1, :] |= mask[1:, :]
        m[:, 1:] |= mask[:, :-1]
        m[:, :-1] |= mask[:, 1:]
        mask = m
    return mask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sample", type=int, default=40,
                    help="frames sampled to build the mask")
    args = ap.parse_args()
    clip, out = Path(args.clip), Path(args.out)
    files = sorted((clip / "frames").glob("*.png"))
    step = max(1, len(files) // args.sample)
    sample = [np.array(Image.open(f).convert("RGB")) for f in files[::step]]
    mask = build_mask(sample)
    print(f"mask covers {mask.mean() * 100:.2f}% of the frame")

    (out / "frames").mkdir(parents=True, exist_ok=True)
    Image.fromarray((mask * 255).astype(np.uint8)).save(out / "osd_mask.png")
    for f in files:
        a = np.array(Image.open(f).convert("RGB"))
        a[mask] = FILL
        Image.fromarray(a).save(out / "frames" / f.name)
    shutil.copy2(clip / "labels.jsonl", out / "labels.jsonl")
    meta = json.loads((clip / "meta.json").read_text())
    meta["osd_masked"] = True
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"wrote masked clip to {out} ({len(files)} frames)")


if __name__ == "__main__":
    main()
