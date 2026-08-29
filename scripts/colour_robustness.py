#!/usr/bin/env python3
"""Does the detector depend on the drone's COLOUR? Measure, don't assume.

The training imagery covers two extremes by accident, not design: the sim
airframe is near-black (model.sdf diffuse 0.08) and the real Anti-UAV drone
is bright white against grey sky. Nothing in the set is mid-tone, camouflaged
or deliberately sky-matched - so "will it see a black/camo drone?" is an open
question, and the honest way to answer it is to repaint the drone in real
frames and re-run the deployed detector.

Method: inside each ground-truth box, build a mask of the pixels that are
actually the airframe (those differing from the box-border background), then
recolour ONLY those pixels. Geometry, motion blur, size and background are
untouched, so any recall change is attributable to appearance alone.

Repaints:
  original    - control
  black       - the classic low-light/stealth airframe
  white       - bright gloss shell
  midgrey     - mid-tone, the weakest contrast against most skies
  skymatch    - painted the local background's own median colour: the
                adversarial worst case (a target trying to disappear)
  camo        - green/brown mottle, the anti-terrain paint scheme
  invert      - luminance-inverted; a sanity control

    python scripts/colour_robustness.py --clip data/clips/terrain_ir_canopy
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from camera.detector import DroneDetector  # noqa: E402
from camera.evaluate import is_hit  # noqa: E402


def airframe_mask(patch: np.ndarray) -> np.ndarray:
    """Pixels inside the box that are the aircraft, not the background."""
    if patch.size == 0:
        return np.zeros(patch.shape[:2], dtype=bool)
    lum = patch.astype(np.float32).mean(axis=2)
    # background estimated from the box border ring
    ring = np.concatenate([lum[0, :], lum[-1, :], lum[:, 0], lum[:, -1]])
    bg = float(np.median(ring))
    spread = float(np.median(np.abs(ring - bg))) + 1e-3
    return np.abs(lum - bg) > max(3.0 * spread, 6.0)


def repaint(patch: np.ndarray, mask: np.ndarray, mode: str,
            rng: np.random.Generator) -> np.ndarray:
    out = patch.copy()
    if not mask.any() or mode == "original":
        return out
    lum = patch.astype(np.float32).mean(axis=2)
    ring = np.concatenate([lum[0, :], lum[-1, :], lum[:, 0], lum[:, -1]])
    bg = float(np.median(ring))
    # keep internal shading so the airframe still looks like an object
    shade = lum[mask]
    shade = (shade - shade.min()) / (np.ptp(shade) + 1e-6)      # 0..1
    if mode == "black":
        base = np.array([18, 18, 20], np.float32); amp = 26.0
    elif mode == "white":
        base = np.array([214, 216, 220], np.float32); amp = 34.0
    elif mode == "midgrey":
        base = np.array([124, 126, 128], np.float32); amp = 26.0
    elif mode == "skymatch":
        ring_rgb = np.concatenate([patch[0, :], patch[-1, :],
                                   patch[:, 0], patch[:, -1]], axis=0)
        base = np.median(ring_rgb, axis=0).astype(np.float32); amp = 10.0
    elif mode == "camo":
        base = np.array([74, 88, 58], np.float32); amp = 40.0
    elif mode == "invert":
        base = np.array([255 - bg] * 3, np.float32); amp = 30.0
    else:
        raise SystemExit(f"unknown mode {mode}")
    px = base[None, :] + (shade[:, None] - 0.5) * amp
    if mode == "camo":
        blotch = rng.normal(0.0, 22.0, size=(px.shape[0], 3))
        px = px + blotch
    out[mask] = np.clip(px, 0, 255).astype(np.uint8)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", required=True)
    ap.add_argument("--conf", type=float, default=0.10)
    ap.add_argument("--limit", type=int, default=150)
    ap.add_argument("--modes", nargs="+",
                    default=["original", "black", "white", "midgrey",
                             "camo", "skymatch", "invert"])
    args = ap.parse_args()

    clip = Path(args.clip)
    labels = [json.loads(l) for l in open(clip / "labels.jsonl")
              if json.loads(l).get("visible")][:args.limit]
    det = DroneDetector(conf=0.03)
    det.warmup()
    rng = np.random.default_rng(3)

    print(f"\n{clip.name}: {len(labels)} drone-visible frames, "
          f"deployed detector, conf {args.conf}")
    print(f"{'repaint':>10} {'recall':>8} {'named drone':>12} {'med conf':>9}")
    for mode in args.modes:
        seen = 0
        named = tot = 0
        confs = []
        for rec in labels:
            im = np.array(Image.open(clip / "frames" / rec["frame"]
                                     ).convert("RGB"))
            x1, y1, x2, y2 = [int(round(v)) for v in rec["bbox"]]
            pad = 4
            x1p, y1p = max(0, x1 - pad), max(0, y1 - pad)
            x2p, y2p = min(im.shape[1], x2 + pad), min(im.shape[0], y2 + pad)
            patch = im[y1p:y2p, x1p:x2p]
            if patch.size and mode != "original":
                m = airframe_mask(patch)
                im[y1p:y2p, x1p:x2p] = repaint(patch, m, mode, rng)
            dets = [d for d in det.detect(im) if d.confidence >= args.conf]
            hits = [d for d in dets if is_hit(d.xyxy, rec)]
            if hits:
                seen += 1
                best = max(hits, key=lambda d: d.confidence)
                confs.append(best.confidence)
                tot += 1
                named += best.cls_name == "drone"
        r = seen / max(len(labels), 1)
        print(f"{mode:>10} {r:>8.3f} {named / max(tot, 1):>12.3f} "
              f"{(np.median(confs) if confs else 0):>9.3f}", flush=True)


if __name__ == "__main__":
    main()
