#!/usr/bin/env python3
"""Does cloud break the detector? Composite real sky conditions and measure.

The simulator's sky is a flat colour and its `overcast` lighting preset only
dims the sun - so every sim number in this project was measured against a
sky with NO STRUCTURE. That flatters the detector twice: a textured sky
both hides targets (contrast varies across the frame) and invents them
(cloud edges and blobs are exactly the small dark/bright shapes the
detector hunts). No clip in the set tests it.

This composites procedural cloud fields over the sky of existing clips and
re-runs the deployed detector, so recall AND false positives are measured
under each condition without a new capture. Conditions:

  clear         - control, the frame as captured
  overcast      - uniform luminance lift, low contrast (the easy cloud case)
  broken        - fair-weather cumulus: mid-scale blobs with hard edges
                  (the dangerous case - blob edges mimic targets)
  stormy        - dark, high-contrast, streaky
  bright_edge   - thin bright cloud with strong rims against blue

Cloud is composited only where the frame is sky-like (bright, low-texture)
and never inside a ground-truth box, so the target itself is unmodified and
any recall change comes from the surrounding scene, not from painting over
the drone.

    python scripts/sky_conditions.py --clip data/clips/eval_birds
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


def fractal_noise(h, w, rng, octaves=4, persistence=0.5):
    """Value-noise cloud field in [0,1], DOMINATED BY LARGE SCALES.

    Real cloud is smooth and metres-to-hundreds-of-metres across; an
    equal-amplitude octave stack looks like television static and makes the
    test unrealistically hard (measured: fine-grain static dropped recall to
    0.37, which says more about the synthetic than about cloud). Start from a
    coarse 6-cell grid and let each finer octave contribute half as much.
    """
    total = np.zeros((h, w), np.float32)
    amp, norm = 1.0, 0.0
    cells = 6
    for _ in range(octaves):
        grid = rng.random((cells + 1, max(2, int(cells * w / h) + 1))
                          ).astype(np.float32)
        img = np.asarray(Image.fromarray((grid * 255).astype(np.uint8)
                                         ).resize((w, h), Image.BICUBIC),
                         np.float32) / 255.0
        total += amp * img
        norm += amp
        amp *= persistence
        cells *= 2
    total /= max(norm, 1e-6)
    return np.clip((total - total.min()) / (np.ptp(total) + 1e-6), 0, 1)


def sky_mask(img: np.ndarray) -> np.ndarray:
    """Sky-like pixels: bright and locally smooth (terrain is neither)."""
    lum = img.astype(np.float32).mean(axis=2)
    # >= not >: the sim sky is a CONSTANT luminance, so a strict compare
    # against a percentile that equals that constant excludes the entire sky
    # (measured: 0% mask coverage, every condition scored identically)
    bright = lum >= np.percentile(lum, 25)
    gx = np.abs(np.diff(lum, axis=1, prepend=lum[:, :1]))
    gy = np.abs(np.diff(lum, axis=0, prepend=lum[:1, :]))
    smooth = (gx + gy) < 12.0
    return bright & smooth


_CLOUD_CACHE: dict = {}


def cloud_field(h, w, rng, drift_px: float = 0.0):
    """The SAME cloud field every frame, slowly drifting.

    Regenerating noise per frame makes the sky flicker completely between
    consecutive frames - meaningless for a single-frame detector test but
    catastrophic (and wrong) for tracking and the motion channel, which
    reasonably assume the world is temporally coherent. Real cloud drifts a
    few pixels a second.
    """
    key = (h, w)
    if key not in _CLOUD_CACHE:
        _CLOUD_CACHE[key] = fractal_noise(h, w, rng)
    base = _CLOUD_CACHE[key]
    if drift_px:
        return np.roll(base, int(drift_px), axis=1)
    return base


def apply_sky(img: np.ndarray, mode: str, rng, boxes,
              drift_px: float = 0.0) -> np.ndarray:
    if mode == "clear":
        return img
    h, w = img.shape[:2]
    n = cloud_field(h, w, rng, drift_px)
    out = img.astype(np.float32)
    m = sky_mask(img).astype(np.float32)
    # never touch the target itself - recall changes must come from context
    # Feathered protection: a hard rectangle leaves a clean-sky patch that
    # is itself a giveaway. Physically the cloud is BEHIND the aircraft, so
    # the target keeps its own pixels and the halo fades out.
    for (x1, y1, x2, y2) in boxes:
        cx_, cy_ = (x1 + x2) / 2, (y1 + y2) / 2
        r_ = max(x2 - x1, y2 - y1) * 0.75 + 3.0
        yy, xx = np.ogrid[0:h, 0:w]
        d_ = np.sqrt((xx - cx_) ** 2 + (yy - cy_) ** 2)
        m *= np.clip((d_ - r_) / max(r_, 1.0), 0.0, 1.0).astype(np.float32)
    m = m[:, :, None]
    if mode == "overcast":
        cloud = 200 + 18 * (n - 0.5)
        alpha = 0.55
    elif mode == "broken":
        blob = (n > 0.55).astype(np.float32)
        soft = np.asarray(Image.fromarray((blob * 255).astype(np.uint8)
                                          ).resize((w // 8, h // 8)
                                                   ).resize((w, h),
                                                            Image.BILINEAR),
                          np.float32) / 255.0
        cloud = 150 + 95 * soft + 20 * (n - 0.5)
        alpha = 0.6
    elif mode == "stormy":
        cloud = 80 + 80 * n
        alpha = 0.7
    elif mode == "bright_edge":
        edge = np.abs(np.diff(n, axis=1, prepend=n[:, :1])) * 14.0
        cloud = 165 + 80 * np.clip(edge, 0, 1) + 30 * (n - 0.5)
        alpha = 0.65
    else:
        raise SystemExit(f"unknown mode {mode}")
    cloud = np.repeat(cloud[:, :, None], 3, axis=2)
    out = out * (1 - alpha * m) + cloud * (alpha * m)
    return np.clip(out, 0, 255).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", required=True)
    ap.add_argument("--conf", type=float, default=0.10)
    ap.add_argument("--limit", type=int, default=120)
    ap.add_argument("--modes", nargs="+",
                    default=["clear", "overcast", "broken", "stormy",
                             "bright_edge"])
    ap.add_argument("--save-example", default=None)
    args = ap.parse_args()

    clip = Path(args.clip)
    recs = [json.loads(l) for l in open(clip / "labels.jsonl")][:args.limit]
    vis = [r for r in recs if r.get("visible")]
    det = DroneDetector(conf=0.03)
    det.warmup()
    meta = json.loads((clip / "meta.json").read_text())
    minutes = len(recs) * meta.get("interval_s", 0.5) / 60.0

    print(f"\n{clip.name}: {len(recs)} frames ({len(vis)} with drone), "
          f"deployed detector, conf {args.conf}")
    print(f"{'sky':>12} {'recall':>8} {'FP/min':>8} {'med conf':>9}")
    for mode in args.modes:
        rng = np.random.default_rng(5)
        seen = fp = 0
        confs = []
        for rec in recs:
            im = np.array(Image.open(clip / "frames" / rec["frame"]
                                     ).convert("RGB"))
            boxes = []
            if rec.get("visible") and rec.get("bbox"):
                boxes.append(rec["bbox"])
            for b in (rec.get("birds") or []):
                boxes.append(b["bbox"])
            im = apply_sky(im, mode, rng, boxes)
            if args.save_example and rec is recs[len(recs) // 2]:
                Image.fromarray(im).save(f"{args.save_example}_{mode}.png")
            dets = [d for d in det.detect(im) if d.confidence >= args.conf]
            hits = [d for d in dets if rec.get("visible")
                    and is_hit(d.xyxy, rec)]
            if rec.get("visible") and hits:
                seen += 1
                confs.append(max(h.confidence for h in hits))
            fp += len(dets) - len(hits)
        print(f"{mode:>12} {seen / max(len(vis), 1):>8.3f} "
              f"{fp / max(minutes, 1e-6):>8.1f} "
              f"{(np.median(confs) if confs else 0):>9.3f}", flush=True)


if __name__ == "__main__":
    main()
