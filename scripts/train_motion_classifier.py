#!/usr/bin/env python3
"""Retrain the track-level motion classifier on FUSED tracks.

Why this exists. `camera/classify.py train` builds its training set by
replaying clips through the RGB detector alone, over clear-sky clips. But at
inference the classifier judges tracks from `camera/fuse_eval.py`, which are
fed by RGB + IR + the motion channel over terrain. Those tracks are sampled
differently — occlusion fragments them, aux channels fill gaps the detector
missed, and the position sequence is a Kalman estimate over a mixed
measurement stream. That is a train/serve mismatch, and it is expensive:
measured on the canopy clip, some stream sees the drone in 96% of frames and
the tracker holds it in 93%, but the motion classifier passes only 69%. The
detector is not the bottleneck there; this classifier is.

So: build the training set from the same fused pipeline that will consume it,
using cached detections only (no GPU), and keep the clear-sky clips in the mix
so nothing that already works is forgotten.

    python scripts/train_motion_classifier.py \
        --clips data/clips/baseline data/clips/birds_only \
                data/clips/train_terrain_canopy data/clips/train_terrain_mission \
        --holdout data/clips/eval_birds data/clips/terrain_ir_canopy \
        --out camera/motion_classifier_v2.json

Held-out clips are scored but never trained on. The terrain matrix clips
(terrain_ir_*, terrain_canopy, terrain_birds, eval_birds) are the campaign's
evaluation set — keep them out of --clips.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from camera.classify import (FEATURE_SETS, MIN_HISTORY,  # noqa: E402
                             MotionClassifier)
from camera.evaluate import hits_object, is_hit  # noqa: E402
from camera.fuse_eval import fuse_measurements, load_clip  # noqa: E402
from camera.tracking import CentroidTracker  # noqa: E402


def samples_from_clip(clip: Path, rgb_conf: float, ir_conf: float,
                      stride: int, feature_set: str = "v1"):
    """(features, label) for every track-window in one clip.

    label 1 = the track is on the drone, 0 = on a bird. Clutter tracks are
    skipped: this classifier's job is drone-vs-bird, and the travel gate and
    static-clutter map handle non-moving clutter. Feeding clutter in as
    negatives would teach it to reject anything with a short history, which is
    most of a fragmented terrain track.
    """
    rows = load_clip(clip, rgb_conf, ir_conf)
    tracker = CentroidTracker(class_consistent=True,
                              suppress_spawn_near_coasting=True)
    out = []
    for i, (f, gt, rgb_dets, aux) in enumerate(rows):
        dets, _sources = fuse_measurements(rgb_dets, aux)
        for tr in tracker.update(dets, timestamp=gt["t"]):
            if len(tr.history) < MIN_HISTORY or i % stride:
                continue
            on_drone = bool(gt["visible"] and is_hit(tr.box, gt))
            on_bird = any(hits_object(tr.box, b) for b in (gt.get("birds") or []))
            if on_drone == on_bird:          # neither, or ambiguous overlap
                continue
            feats = FEATURE_SETS[feature_set][1](tr.history)
            if feats is not None:
                out.append((feats, 1.0 if on_drone else 0.0))
    return out


def collect(clips, rgb_conf, ir_conf, stride, feature_set="v1"):
    X, y, per_clip = [], [], {}
    for c in clips:
        s = samples_from_clip(Path(c), rgb_conf, ir_conf, stride, feature_set)
        n_d = sum(1 for _, lab in s if lab > 0.5)
        per_clip[Path(c).name] = (n_d, len(s) - n_d)
        print(f"  {Path(c).name:28s} {n_d:>5} drone  {len(s) - n_d:>5} bird",
              flush=True)
        for f, lab in s:
            X.append(f)
            y.append(lab)
    return np.array(X), np.array(y), per_clip


def score(clf, X, y, thr=0.5):
    if len(X) == 0:
        return None
    p = np.array([clf.probability(f) for f in X])
    pred = (p >= thr).astype(float)
    drone = y == 1
    bird = y == 0
    return {
        "n": int(len(y)),
        "drone_kept": float(pred[drone].mean()) if drone.any() else None,
        "bird_rejected": float((1 - pred[bird]).mean()) if bird.any() else None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", nargs="+", required=True)
    ap.add_argument("--holdout", nargs="*", default=[])
    ap.add_argument("--rgb-conf", type=float, default=0.05)
    ap.add_argument("--ir-conf", type=float, default=0.2)
    ap.add_argument("--stride", type=int, default=3,
                    help="sample every Nth frame; consecutive track-windows "
                         "are near-duplicates and just inflate the set")
    ap.add_argument("--feature-set", default="v1", choices=list(FEATURE_SETS),
                    help="v1 = the original six sky-fitted motion features; "
                         "v3 = the domain-stable set (see camera/classify.py)")
    ap.add_argument("--out", default="camera/motion_classifier_v2.json")
    args = ap.parse_args()

    print("building training set from fused tracks...")
    X, y, _ = collect(args.clips, args.rgb_conf, args.ir_conf, args.stride,
                      args.feature_set)
    if len(X) < 20:
        raise SystemExit("not enough track samples to train")
    print(f"\n{len(X)} samples: {int(y.sum())} drone, {int(len(y) - y.sum())} bird")

    clf = MotionClassifier(feature_set=args.feature_set).fit(X, y)
    old = MotionClassifier.load()

    print("\nlearned weights (positive pushes toward 'drone'):")
    for name, wv in sorted(zip(clf.feature_names, clf.w),
                           key=lambda kv: -abs(kv[1])):
        print(f"  {name:>16}: {wv:+.3f}")

    tr = score(clf, X, y)
    print(f"\ntrain: drone kept {tr['drone_kept']:.3f}, "
          f"birds rejected {tr['bird_rejected']:.3f}")

    if args.holdout:
        print("\nheld-out clips (never trained on) — new vs current model:")
        for c in args.holdout:
            Xh, yh, _ = collect([c], args.rgb_conf, args.ir_conf,
                                args.stride, args.feature_set)
            Xo, yo, _ = ((Xh, yh, None) if args.feature_set == old.feature_set
                         else collect([c], args.rgb_conf, args.ir_conf,
                                      args.stride, old.feature_set))
            if not len(Xh):
                continue
            n, o = score(clf, Xh, yh), score(old, Xo, yo)
            def fmt(s, k):
                return "  n/a " if s is None or s[k] is None else f"{s[k]:.3f}"
            print(f"  {Path(c).name:28s} drone kept {fmt(o,'drone_kept')} -> "
                  f"{fmt(n,'drone_kept')}   birds rejected "
                  f"{fmt(o,'bird_rejected')} -> {fmt(n,'bird_rejected')}")

    clf.save(args.out)
    print(f"\nwrote {args.out}")
    print("NOT installed as the live classifier — verify with "
          "camera/fuse_eval.py first, then copy over "
          "camera/motion_classifier.json")


if __name__ == "__main__":
    main()
