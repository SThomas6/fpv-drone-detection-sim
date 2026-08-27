#!/usr/bin/env python3
"""Score the detector against the simulator's ground truth.

Two stages, so the expensive part happens once:

  1. inference — run the detector over a clip at a LOW confidence floor and
     cache every raw detection with its score.
  2. analysis  — score those cached detections at any number of confidence
     thresholds, bucketed by target size and range.

    python camera/evaluate.py run     --clip data/clips/baseline --mode full
    python camera/evaluate.py analyze --clip data/clips/baseline --mode full

Matching: at these sizes a 6 px box that is 3 px off has near-zero IoU while
being a perfectly good detection, so a hit is IoU >= 0.3 OR a centre within
max(12 px, 2x the true box width). Both criteria are reported.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from camera.detector import DroneDetector  # noqa: E402
from simulator.station_camera import size_bucket  # noqa: E402

CONF_FLOOR = 0.03           # cache everything above this; sweep upward later
SWEEP = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]
IOU_HIT = 0.3
RANGE_BINS = [(0, 20), (20, 40), (40, 60), (60, 80), (80, 100), (100, 130),
              (130, 160), (160, 190), (190, 220), (220, 260), (260, 1e9)]


def iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    ua = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / ua if ua > 0 else 0.0


def hits_object(det_box, obj, tol_floor=12.0):
    """Does this detection correspond to the given labelled object?"""
    gx1, gy1, gx2, gy2 = obj["bbox"]
    if iou(det_box, obj["bbox"]) >= IOU_HIT:
        return True
    dcx, dcy = (det_box[0] + det_box[2]) / 2, (det_box[1] + det_box[3]) / 2
    gcx, gcy = (gx1 + gx2) / 2, (gy1 + gy2) / 2
    tol = max(tol_floor, 2.0 * (gx2 - gx1))
    return math.hypot(dcx - gcx, dcy - gcy) <= tol


def is_hit(det_box, gt):
    gx1, gy1, gx2, gy2 = gt["bbox"]
    if iou(det_box, gt["bbox"]) >= IOU_HIT:
        return True
    dcx, dcy = (det_box[0] + det_box[2]) / 2, (det_box[1] + det_box[3]) / 2
    gcx, gcy = (gx1 + gx2) / 2, (gy1 + gy2) / 2
    tol = max(12.0, 2.0 * (gx2 - gx1))
    return math.hypot(dcx - gcx, dcy - gcy) <= tol


# ---------------------------------------------------------------- stage 1

def run(clip: Path, mode: str, device: str, limit: int | None,
        min_range: float | None = None, weights: str | None = None,
        tag: str | None = None):
    labels = [json.loads(l) for l in open(clip / "labels.jsonl")]
    if min_range is not None:
        labels = [r for r in labels
                  if r.get("range_m") is not None and r["range_m"] >= min_range]
    if limit:
        labels = labels[:limit]

    det = DroneDetector(device=device, conf=CONF_FLOOR,
                        use_sahi=(mode == "sahi"), weights=weights)
    det.warmup()
    print(f"mode={mode} device={device} weights={Path(det.weights).name} "
          f"classes={det.names} frames={len(labels)}", flush=True)

    out, times = [], []
    for i, rec in enumerate(labels):
        frame = np.array(Image.open(clip / "frames" / rec["frame"]).convert("RGB"))
        t0 = time.perf_counter()
        dets = det.infer(frame)
        times.append((time.perf_counter() - t0) * 1000)
        out.append({
            "frame": rec["frame"],
            "detections": [{"xyxy": [round(v, 1) for v in d.xyxy],
                            "conf": round(d.confidence, 4),
                            "cls": d.cls_name} for d in dets],
        })
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(labels)}  ({np.median(times):.0f} ms/frame)", flush=True)

    suffix = mode if min_range is None else f"{mode}_from{int(min_range)}m"
    if tag:
        suffix = f"{suffix}_{tag}"
    res_path = clip / f"detections_{suffix}.jsonl"
    with open(res_path, "w") as fh:
        for r in out:
            fh.write(json.dumps(r) + "\n")
    timing = {"mode": mode, "device": device,
              "median_ms": round(float(np.median(times)), 1),
              "p90_ms": round(float(np.percentile(times, 90)), 1),
              "fps": round(1000.0 / float(np.median(times)), 1)}
    (clip / f"timing_{suffix}.json").write_text(json.dumps(timing, indent=2))
    print(f"\nwrote {res_path}")
    print(f"latency: median {timing['median_ms']} ms -> {timing['fps']} FPS")


# ---------------------------------------------------------------- stage 2

def analyze(clip: Path, mode: str, min_range: float | None = None,
            tag: str | None = None):
    suffix = mode if min_range is None else f"{mode}_from{int(min_range)}m"
    if tag:
        suffix = f"{suffix}_{tag}"
    labels = {r["frame"]: r for r in
              (json.loads(l) for l in open(clip / "labels.jsonl"))}
    det_file = clip / f"detections_{suffix}.jsonl"
    if not det_file.exists() and min_range is not None:
        det_file = clip / f"detections_{mode}.jsonl"   # score a full run over the subset
    dets = {r["frame"]: r["detections"] for r in
            (json.loads(l) for l in open(det_file))}
    if min_range is not None:
        dets = {f: d for f, d in dets.items()
                if labels[f].get("range_m") is not None
                and labels[f]["range_m"] >= min_range}
    meta = json.loads((clip / "meta.json").read_text())
    interval = meta.get("interval_s", 0.5)

    print(f"\n=== {clip.name} / {mode}"
          f"{f' / range >= {min_range:.0f}m' if min_range else ''} ===")
    print(f"{len(dets)} frames scored, {sum(1 for f in dets if labels[f]['visible'])} "
          f"with the drone in frame\n")

    any_birds = any(labels[f].get("birds") for f in dets)

    rows = []
    for thr in SWEEP:
        tp = fp = fn = fp_bird = 0
        by_bucket, by_range = {}, {}
        for frame, dlist in dets.items():
            gt = labels[frame]
            kept = [d for d in dlist
                    if d["conf"] >= thr and d.get("cls", "drone") == "drone"]
            birds = gt.get("birds") or []
            if birds:
                for d in kept:
                    on_bird = any(hits_object(d["xyxy"], b) for b in birds)
                    is_drone = gt["visible"] and is_hit(d["xyxy"], gt)
                    if on_bird and not is_drone:
                        fp_bird += 1
            if gt["visible"]:
                hits = [d for d in kept if is_hit(d["xyxy"], gt)]
                bucket = gt["bucket"]
                b = by_bucket.setdefault(bucket, [0, 0])
                rb = next(f"{lo}-{hi}m" for lo, hi in RANGE_BINS
                          if lo <= gt["range_m"] < hi)
                rr = by_range.setdefault(rb, [0, 0])
                b[1] += 1
                rr[1] += 1
                if hits:
                    tp += 1
                    b[0] += 1
                    rr[0] += 1
                else:
                    fn += 1
                fp += len(kept) - len(hits)
            else:
                fp += len(kept)
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        fp_per_min = fp / (len(dets) * interval / 60.0)
        rows.append((thr, recall, precision, fp_per_min, tp, fp, fn,
                     by_bucket, by_range, fp_bird))

    print(f"{'conf':>6} {'recall':>8} {'precis':>8} {'FP/min':>8} "
          f"{'TP':>5} {'FP':>5} {'FN':>5}" + (f" {'FP/bird':>8}" if any_birds else ""))
    for r in rows:
        thr, rc, pr, fpm, tp, fp, fn = r[0], r[1], r[2], r[3], r[4], r[5], r[6]
        extra = f" {r[9]:>8}" if any_birds else ""
        print(f"{thr:>6.2f} {rc:>8.3f} {pr:>8.3f} {fpm:>8.1f} {tp:>5} {fp:>5} {fn:>5}{extra}")

    # operating point: best recall among thresholds with precision >= 0.9
    ok = [r for r in rows if r[2] >= 0.9] or rows
    best = max(ok, key=lambda r: r[1])
    if any_birds:
        bird_frames = sum(1 for f in dets if labels[f].get("birds"))
        n_birds = sum(len(labels[f].get("birds") or []) for f in dets)
        print(f"\nbirds present in {bird_frames} frames "
              f"({n_birds} bird-sightings total); "
              f"{best[9]} drone-class detections landed on a bird at conf={best[0]:.2f}")
        ided = 0
        for frame, dlist in dets.items():
            for d in dlist:
                if d.get("cls") == "bird" and d["conf"] >= best[0] and any(
                        hits_object(d["xyxy"], b)
                        for b in (labels[frame].get("birds") or [])):
                    ided += 1
        print(f"(the model itself correctly identified birds {ided} times "
              f"at that threshold)")
    print(f"\nsuggested operating point: conf={best[0]:.2f} "
          f"(recall {best[1]:.3f}, precision {best[2]:.3f}, {best[3]:.1f} FP/min)")

    print(f"\nrecall by target size at conf={best[0]:.2f} "
          f"(<32px is the regime that decides the model):")
    for bucket in ("<16px", "16-32px", "32-96px", ">96px"):
        if bucket in best[7]:
            hit, tot = best[7][bucket]
            print(f"  {bucket:>8}: {hit:>4}/{tot:<4} = {hit / tot:.3f}")

    print(f"\nrecall by range at conf={best[0]:.2f}:")
    ordered = sorted(best[8].items(), key=lambda kv: float(kv[0].split("-")[0]))
    for rb, (hit, tot) in ordered:
        r = hit / tot
        bar = "#" * int(round(r * 30))
        print(f"  {rb:>10}: {hit:>4}/{tot:<4} = {r:.3f} {bar}")

    # Detection range: furthest bin (with enough samples) still above each level.
    for level in (0.9, 0.5):
        reach = None
        for rb, (hit, tot) in ordered:
            if tot >= 5 and hit / tot >= level:
                reach = rb
            elif tot >= 5:
                break
        print(f"  recall stays >= {level:.0%} out to the {reach} bin"
              if reach else f"  recall never reaches {level:.0%}")

    timing_file = clip / f"timing_{suffix}.json"
    if timing_file.exists():
        t = json.loads(timing_file.read_text())
        print(f"\nlatency: median {t['median_ms']} ms -> {t['fps']} FPS ({t['device']})")

    summary = {
        "clip": clip.name, "mode": mode, "min_range_m": min_range,
        "operating_conf": best[0], "recall": round(best[1], 4),
        "precision": round(best[2], 4), "fp_per_min": round(best[3], 2),
        "by_bucket": {k: {"hit": v[0], "total": v[1], "recall": round(v[0] / v[1], 4)}
                      for k, v in best[7].items()},
        "by_range": {k: {"hit": v[0], "total": v[1], "recall": round(v[0] / v[1], 4)}
                     for k, v in best[8].items()},
        "fp_on_birds": best[9] if any_birds else None,
        "sweep": [{"conf": r[0], "recall": round(r[1], 4),
                   "precision": round(r[2], 4), "fp_per_min": round(r[3], 2),
                   "fp_on_birds": r[9]}
                  for r in rows],
    }
    (clip / f"summary_{suffix}.json").write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {clip / f'summary_{suffix}.json'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["run", "analyze"])
    ap.add_argument("--clip", default="data/clips/baseline")
    ap.add_argument("--mode", default="full", choices=["full", "sahi"])
    ap.add_argument("--device", default=None, help="cuda|mps|cpu (default: auto)")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--min-range", type=float, default=None,
                    help="only score frames at or beyond this range, in metres")
    ap.add_argument("--weights", default=None,
                    help="run stage: override model weights (for old/new comparison)")
    ap.add_argument("--tag", default=None,
                    help="suffix for the detections/summary files, e.g. 'old'")
    args = ap.parse_args()

    clip = Path(args.clip)
    if args.stage == "run":
        run(clip, args.mode, args.device, args.limit, args.min_range,
            args.weights, args.tag)
    else:
        analyze(clip, args.mode, args.min_range, args.tag)


if __name__ == "__main__":
    main()
