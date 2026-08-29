#!/usr/bin/env python3
"""Repaint the drone in existing training tiles: colour-augmented training.

Measured problem (scripts/colour_robustness.py): the detector FINDS drones of
any colour, but it NAMES a white one 'bird' 96% of the time - and the
bird-mute policy would then silence a real target. The cause is the training
set, not the architecture: every sim airframe is near-black (model.sdf
diffuse 0.08) and the only light drones it ever saw were the real Anti-UAV
sequences. 'Drone' has partly come to mean 'dark blob'.

Same remedy that worked for motion blur (m5): synthesise the missing
appearance rather than re-weight what exists. For each drone-bearing tile,
build a mask of the airframe pixels and repaint them white / mid-grey /
camouflage, keeping geometry, background, blur and label untouched. Bird
tiles are NEVER repainted - the point is to break 'dark = drone', not to
teach 'light = drone'.

    python scripts/build_colour_tiles.py --src data/finetune --out data/finetune_colour
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.colour_robustness import airframe_mask, repaint  # noqa: E402

DRONE_CLASS = 0


def yolo_boxes(label_path: Path, w: int, h: int):
    """(cls, x1,y1,x2,y2) in pixels from a YOLO label file."""
    out = []
    if not label_path.exists():
        return out
    for line in label_path.read_text().splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        c, cx, cy, bw, bh = int(parts[0]), *map(float, parts[1:])
        out.append((c, (cx - bw / 2) * w, (cy - bh / 2) * h,
                    (cx + bw / 2) * w, (cy + bh / 2) * h))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", nargs="+", required=True,
                    help="finetune dirs whose images/train tiles are repainted")
    ap.add_argument("--out", default="data/finetune_colour")
    ap.add_argument("--modes", nargs="+",
                    default=["white", "midgrey", "camo"])
    ap.add_argument("--per-mode", type=int, default=450,
                    help="max tiles per repaint mode (keeps the mix balanced)")
    ap.add_argument("--seed", type=int, default=17)
    args = ap.parse_args()

    out = Path(args.out)
    (out / "images/train").mkdir(parents=True, exist_ok=True)
    (out / "labels/train").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    # collect tiles that actually contain a drone box
    cands = []
    for srcd in args.src:
        idir = Path(srcd) / "images/train"
        ldir = Path(srcd) / "labels/train"
        for ip in sorted(idir.glob("*.png")):
            lp = ldir / (ip.stem + ".txt")
            if not lp.exists():
                continue
            txt = lp.read_text()
            if any(l.startswith("0 ") for l in txt.splitlines()):
                cands.append((ip, lp))
    print(f"{len(cands)} drone-bearing source tiles")

    made = 0
    for mode in args.modes:
        idx = rng.permutation(len(cands))[:args.per_mode]
        for k in idx:
            ip, lp = cands[int(k)]
            im = np.array(Image.open(ip).convert("RGB"))
            h, w = im.shape[:2]
            touched = False
            for c, x1, y1, x2, y2 in yolo_boxes(lp, w, h):
                if c != DRONE_CLASS:
                    continue          # never repaint birds
                x1i, y1i = max(0, int(x1) - 3), max(0, int(y1) - 3)
                x2i, y2i = min(w, int(x2) + 3), min(h, int(y2) + 3)
                patch = im[y1i:y2i, x1i:x2i]
                if patch.size == 0:
                    continue
                m = airframe_mask(patch)
                if not m.any():
                    continue
                im[y1i:y2i, x1i:x2i] = repaint(patch, m, mode, rng)
                touched = True
            if not touched:
                continue
            name = f"{ip.stem}_{mode}"
            Image.fromarray(im).save(out / "images/train" / f"{name}.png")
            (out / "labels/train" / f"{name}.txt").write_text(lp.read_text())
            made += 1
        print(f"  {mode}: cumulative {made} tiles", flush=True)
    print(f"wrote {made} colour-augmented tiles to {out}")


if __name__ == "__main__":
    main()
