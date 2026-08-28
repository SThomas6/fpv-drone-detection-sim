#!/usr/bin/env python3
"""Score m3 "bird-namer" checkpoints: drone recall AND class naming, per pass.

`scripts/sweep_checkpoints.py` picks by drone recall / FP-per-min alone,
which is right for a recall campaign and wrong for this one: m3 exists to
make the detector NAME birds correctly (bird-mute only mutes what the
detector calls bird), and naming quality is invisible to that sweeper. This
one scores, per checkpoint, on the selection clip (train_ir_sweep - a
TRAINING-POOL clip, never an eval clip):

  drone frame-recall  (centre-distance, conf 0.10, any class counts as seen)
  drone naming        (share of drone-hitting detections with cls 'drone')
  bird naming         (share of bird-hitting detections with cls 'bird')

at BOTH inference scales - full-frame and SAHI - because the SAHI pass is
where naming was broken (79.5% vs 93.2% at native scale) and the SAHI-scale
training tiles are the change most likely to move it.

A guard clip (default: a realtrain_* real-footage clip) catches real-domain
forgetting: report drone recall there and flag any checkpoint that sags.

    python scripts/sweep_m3_naming.py \
        --runs runs/experiments/m3_birdnamer runs/experiments/m3_birdnamer_b \
        --stride 3
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
from camera.evaluate import hits_object, is_hit  # noqa: E402

SELECT_CLIP = "data/clips/train_ir_sweep"
GUARD_CLIP = "data/clips/realtrain_rgb_20190925_133630_1_1"
# Sky naming clips — TRAINING-POOL sky captures, never eval_birds. Added
# after the first m3 pick (epoch12) turned out to trade sky drone naming
# without the sweep ever measuring it: baseline has the drone against sky
# (drone naming), birds_only has only birds (bird naming).
SKY_DRONE_CLIP = "data/clips/baseline"
SKY_BIRD_CLIP = "data/clips/birds_only"
CONF = 0.10


def score_clip(det: DroneDetector, clip: Path, sahi: bool,
               limit: int | None = None):
    labels = [json.loads(l) for l in open(clip / "labels.jsonl")]
    if limit:
        labels = labels[:limit]
    seen = vis = 0
    on_drone = {"drone": 0, "other": 0}
    on_bird = {"bird": 0, "other": 0}
    for rec in labels:
        frame = np.array(Image.open(clip / "frames" / rec["frame"]).convert("RGB"))
        dets = det.detect_sliced(frame) if sahi else det.detect(frame)
        dets = [d for d in dets if d.confidence >= CONF]
        if rec.get("visible"):
            vis += 1
            if any(is_hit(d.xyxy, rec) for d in dets):
                seen += 1
        for d in dets:
            hit_d = rec.get("visible") and is_hit(d.xyxy, rec)
            hit_b = any(hits_object(d.xyxy, b) for b in (rec.get("birds") or []))
            if hit_d and not hit_b:
                on_drone["drone" if d.cls_name == "drone" else "other"] += 1
            elif hit_b and not hit_d:
                on_bird["bird" if d.cls_name == "bird" else "other"] += 1
    nd = sum(on_drone.values())
    nb = sum(on_bird.values())
    return {
        "recall": seen / vis if vis else None,
        "drone_named": on_drone["drone"] / nd if nd else None,
        "bird_named": on_bird["bird"] / nb if nb else None,
        "n_drone_dets": nd, "n_bird_dets": nb,
    }


def fmt(v):
    return "  n/a" if v is None else f"{v:.3f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True,
                    help="experiment dirs whose weights/epoch*.pt are scored")
    ap.add_argument("--stride", type=int, default=3,
                    help="score every Nth epoch checkpoint (plus the last)")
    ap.add_argument("--select-clip", default=SELECT_CLIP)
    ap.add_argument("--guard-clip", default=GUARD_CLIP)
    ap.add_argument("--guard-limit", type=int, default=120,
                    help="frames of the guard clip to score (speed)")
    ap.add_argument("--sky-limit", type=int, default=200,
                    help="frames of each sky naming clip to score")
    ap.add_argument("--baseline", default="camera/weights/drone_bird_v1.pt",
                    help="deployed weights scored first as the reference row; "
                         "'' skips")
    ap.add_argument("--out", default="runs/m3_naming_sweep.json")
    args = ap.parse_args()

    ckpts = []
    if args.baseline:
        ckpts.append(("baseline", Path(args.baseline)))
    for run in args.runs:
        wdir = Path(run) / "weights"
        eps = sorted(wdir.glob("epoch*.pt"),
                     key=lambda p: int(p.stem.replace("epoch", "")))
        picked = [p for i, p in enumerate(eps)
                  if i % args.stride == 0 or i == len(eps) - 1]
        ckpts += [(Path(run).name, p) for p in picked]
    print(f"scoring {len(ckpts)} checkpoints on {args.select_clip} "
          f"(+ guard {args.guard_clip})\n")

    sel = Path(args.select_clip)
    guard = Path(args.guard_clip)
    rows = []
    print(f"{'checkpoint':>28} | {'recall':>7} {'droneNm':>8} {'birdNm':>7} | "
          f"{'sahiRec':>8} {'sahiBird':>9} | {'guardRec':>9} | "
          f"{'skyDrone':>8} {'skyBird':>8}")
    for run_name, ck in ckpts:
        det = DroneDetector(weights=str(ck), conf=0.03)
        det.warmup()
        full = score_clip(det, sel, sahi=False)
        sahi = score_clip(det, sel, sahi=True)
        g = score_clip(det, guard, sahi=False, limit=args.guard_limit) \
            if guard.exists() else {"recall": None}
        skyd = score_clip(det, Path(SKY_DRONE_CLIP), sahi=False,
                          limit=args.sky_limit)
        skyb = score_clip(det, Path(SKY_BIRD_CLIP), sahi=False,
                          limit=args.sky_limit)
        label = f"{run_name}/{ck.stem}"
        print(f"{label:>28} | {fmt(full['recall']):>7} "
              f"{fmt(full['drone_named']):>8} {fmt(full['bird_named']):>7} | "
              f"{fmt(sahi['recall']):>8} {fmt(sahi['bird_named']):>9} | "
              f"{fmt(g['recall']):>9} | {fmt(skyd['drone_named']):>8} "
              f"{fmt(skyb['bird_named']):>8}", flush=True)
        rows.append({"run": run_name, "ckpt": str(ck), "full": full,
                     "sahi": sahi, "guard": g,
                     "sky_drone": skyd, "sky_bird": skyb})

    Path(args.out).write_text(json.dumps(rows, indent=2))
    print(f"\nwrote {args.out}")
    print("Pick by hand: want bird naming up (both passes) at recall and "
          "guard within a point or two of the deployed model's row.")


if __name__ == "__main__":
    main()
