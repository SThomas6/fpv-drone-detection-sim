#!/usr/bin/env python3
"""Track-level gating for a SINGLE RGB camera — no IR, no motion channel.

`camera/fuse_eval.py` gets its false-alarm discipline from a travel gate: a
sun-warmed rock or a blurred building corner never goes anywhere, while a real
target — even a hovering one — wanders. That gate needs only the RGB
detector's own track history, so it does not actually require the extra
sensors fusion has. This module applies it to the plain-RGB path, which is all
the real Anti-UAV footage has.

The problem it is aimed at, measured: on the real "medium" sequence the
detector reaches 0.956 recall but at 262 FP/min, and an OSD-mask audit already
ruled out the burned-in turret overlay as the cause — the false alarms are
genuine foliage and building blur. Appearance-only retraining was tried and
traded recall for precision rather than beating it (see docs/handoff.md).

    python camera/track_eval.py --clip data/clips/real_rgb_..._masked

Ego-motion matters here in a way it does not in the sim. The Anti-UAV clips
come from a PAN-TILT TRACKING TURRET, so the background sweeps across the
frame and static clutter is not static in image space. `--stabilize` measures
the frame-to-frame global shift by phase correlation and gates on the
displacement that REMAINS after removing it. With a locked-off camera the
global shift is ~0 and the two agree, so the compensated statistic is the
strictly safer one; both are reported side by side.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from camera.clutter_map import StaticClutterSuppressor  # noqa: E402
from camera.detector import Detection, merge_close  # noqa: E402
from camera.evaluate import hits_object, is_hit  # noqa: E402
from camera.tracking import CentroidTracker  # noqa: E402

TRAVEL_SWEEP = [0.0, 4.0, 8.0, 16.0, 32.0, 64.0, 100.0, 150.0, 250.0]


# ---------------------------------------------------------------- ego-motion

def _gray_small(path: Path, width: int = 480) -> np.ndarray:
    from PIL import Image
    im = Image.open(path).convert("L")
    h = max(1, int(round(im.height * width / im.width)))
    return np.asarray(im.resize((width, h), Image.BILINEAR), dtype=np.float64)


def _phase_shift(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """Global (dx, dy) taking image a to image b, by phase correlation.

    A Hann window kills the wrap-around edge response that otherwise pins the
    peak at (0, 0) on frames with strong borders (these clips have letterbox
    bars and a burned-in overlay).
    """
    h, w = a.shape
    win = np.outer(np.hanning(h), np.hanning(w))
    fa = np.fft.rfft2((a - a.mean()) * win)
    fb = np.fft.rfft2((b - b.mean()) * win)
    cross = fa.conj() * fb
    mag = np.abs(cross)
    mag[mag < 1e-12] = 1e-12
    corr = np.fft.irfft2(cross / mag, s=a.shape)
    peak = int(np.argmax(corr))
    py, px = divmod(peak, w)
    if py > h // 2:
        py -= h
    if px > w // 2:
        px -= w
    return float(px), float(py)


def ego_motion(clip: Path, frames: list[str], width: int = 480,
               cache: bool = True) -> dict[str, tuple[float, float]]:
    """Cumulative background shift per frame, in FULL-resolution pixels.

    Cached beside the clip: it depends only on the imagery, never on a model,
    so it survives every retrain.
    """
    cache_path = clip / f"egomotion_{width}.json"
    if cache and cache_path.exists():
        d = json.loads(cache_path.read_text())
        if d.get("frames") == frames:
            return {f: tuple(v) for f, v in d["cumulative"].items()}

    from PIL import Image
    full_w = Image.open(clip / "frames" / frames[0]).width
    scale = full_w / width
    cum = {frames[0]: (0.0, 0.0)}
    prev = _gray_small(clip / "frames" / frames[0], width)
    cx = cy = 0.0
    for i, f in enumerate(frames[1:], start=1):
        cur = _gray_small(clip / "frames" / f, width)
        dx, dy = _phase_shift(prev, cur)
        cx += dx * scale
        cy += dy * scale
        cum[f] = (cx, cy)
        prev = cur
        if i % 100 == 0:
            print(f"  ego-motion {i}/{len(frames)}", flush=True)
    if cache:
        cache_path.write_text(json.dumps(
            {"frames": frames, "width": width,
             "cumulative": {k: list(v) for k, v in cum.items()}}))
    return cum


# ---------------------------------------------------------------- replay

def _travel(points: list[tuple[float, float]]) -> float:
    """Extent of a track's positions: the diagonal of their bounding box.

    Same statistic fuse_eval.py gates on. Deliberately not path length —
    detector jitter on a static blob accumulates path length without the blob
    ever going anywhere, which is exactly the failure this gate must not have.
    """
    if len(points) < 2:
        return 0.0
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return math.hypot(max(xs) - min(xs), max(ys) - min(ys))


def replay(clip: Path, mode: str, tag: str | None, conf: float,
           stabilize: bool, gate_cap: float = 250.0,
           clutter: "StaticClutterSuppressor | None" = None,
           exempt_travel: float = 150.0):
    """Run cached detections through the tracker; one row per track-frame.

    With `clutter` set, detections are filtered by the static-clutter map
    BEFORE they reach the tracker — suppressed clutter must not spawn or feed
    tracks, or it keeps its alarm by another route. A detection lying on a
    track that has already demonstrated `exempt_travel` px of net motion is
    exempt, so a drone crossing an established clutter anchor is not muted.
    """
    suffix = f"{mode}_{tag}" if tag else mode
    det_file = clip / f"detections_{suffix}.jsonl"
    if not det_file.exists():
        raise SystemExit(f"{det_file} missing — run camera/evaluate.py run first")
    labels = {r["frame"]: r for r in
              (json.loads(l) for l in open(clip / "labels.jsonl"))}
    cached = {r["frame"]: r["detections"] for r in
              (json.loads(l) for l in open(det_file))}
    frames = [f for f in labels if f in cached]
    frames.sort(key=lambda f: labels[f]["t"])

    shift = ego_motion(clip, frames) if stabilize else {}

    tracker = CentroidTracker(class_consistent=True,
                              suppress_spawn_near_coasting=True,
                              max_size_gate_px=gate_cap)
    # Positions are recorded outside the tracker so the tracker itself stays
    # untouched: it must keep associating in raw image space, where the
    # measurements actually live. Only the travel STATISTIC is compensated.
    raw_hist: dict[int, list[tuple[float, float]]] = {}
    stab_hist: dict[int, list[tuple[float, float]]] = {}
    n_suppressed = 0

    def _exempt(d):
        """Is this detection already claimed by a demonstrably moving track?

        Deliberately narrow. Only tracks that were actually DETECTED on the
        previous frame count: a coasting track keeps its Kalman prediction
        flying across the frame at its last velocity, and letting those vouch
        for detections exempts essentially everything (measured — it took the
        suppressor from 188 suppressions to 0).
        """
        if exempt_travel <= 0:
            return False
        cx, cy = d.centre
        for tr in tracker.active:
            if tr.misses > 0:
                continue
            if _travel(raw_hist.get(tr.track_id, [])) < exempt_travel:
                continue
            px, py = tr.position
            if math.hypot(cx - px, cy - py) <= max(50.0, 0.75 * d.width):
                return True
        return False

    rows = []
    for f in frames:
        gt = labels[f]
        dets = merge_close([
            Detection(*d["xyxy"], confidence=d["conf"],
                      cls_name=d.get("cls", "drone"))
            for d in cached[f] if d["conf"] >= conf])
        if clutter is not None:
            dets, dropped = clutter.step(dets, gt["t"], exempt=_exempt)
            n_suppressed += len(dropped)
        sx, sy = shift.get(f, (0.0, 0.0))
        for tr in tracker.update(dets, timestamp=gt["t"]):
            x, y = tr.position
            rh = raw_hist.setdefault(tr.track_id, [])
            sh = stab_hist.setdefault(tr.track_id, [])
            if tr.misses == 0:
                rh.append((x, y))
                sh.append((x - sx, y - sy))
                del rh[:-90]
                del sh[:-90]
            votes = [h[5] for h in tr.history if len(h) > 5]
            rows.append({
                "frame": f,
                "track_id": tr.track_id,
                "is_drone": bool(gt["visible"] and is_hit(tr.box, gt)),
                "on_bird": any(hits_object(tr.box, b)
                               for b in (gt.get("birds") or [])),
                "hits": tr.hits,
                "misses": tr.misses,
                "width": tr.width,
                "travel": _travel(rh),
                "travel_stab": _travel(sh),
                "drone_vote": (sum(1 for v in votes if v == "drone") / len(votes)
                               if votes else 1.0),
            })
    if clutter is not None:
        print(f"static-clutter map suppressed {n_suppressed} detections, "
              f"{len(clutter.anchors)} live anchors at end")
    return labels, frames, rows


# ---------------------------------------------------------------- scoring

def score(labels, frames, rows, minutes, travel_key, min_travel,
          min_hits=1, vote_thr=0.0, allow_coast=True):
    """Frame recall and FP/min for one gate setting.

    Scored exactly like camera/evaluate.py so the numbers are comparable to
    the detector-alone scoreboard: recall is the share of drone-visible frames
    with at least one surviving declaration on the drone, and every other
    surviving declaration is a false alarm.
    """
    per_frame: dict[str, list] = {f: [] for f in frames}
    for r in rows:
        if r[travel_key] < min_travel or r["hits"] < min_hits:
            continue
        if r["drone_vote"] < vote_thr:
            continue
        if not allow_coast and r["misses"] > 0:
            continue
        per_frame[r["frame"]].append(r)
    vis = sum(1 for f in frames if labels[f]["visible"])
    tp = fp = 0
    for f in frames:
        kept = per_frame[f]
        hit = [r for r in kept if r["is_drone"]]
        if labels[f]["visible"] and hit:
            tp += 1
        fp += len(kept) - len(hit)
    return {"recall": tp / vis if vis else 0.0, "fp": fp,
            "fp_per_min": fp / minutes if minutes else 0.0, "tp": tp,
            "visible": vis}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", required=True)
    ap.add_argument("--mode", default="full", choices=["full", "sahi"])
    ap.add_argument("--tag", default=None)
    ap.add_argument("--conf", type=float, default=0.10)
    ap.add_argument("--stabilize", action="store_true",
                    help="gate on ego-motion-compensated travel (needed for "
                         "the pan-tilt turret footage)")
    ap.add_argument("--min-hits", type=int, default=1)
    ap.add_argument("--vote", type=float, default=0.0,
                    help="min share of a track's RGB frames called 'drone'")
    ap.add_argument("--no-coast", action="store_true",
                    help="a coasting track (no detection this frame) may not "
                         "raise an alarm")
    ap.add_argument("--clutter", action="store_true",
                    help="enable the online static-clutter location map")
    ap.add_argument("--clutter-radius", type=float, default=60.0)
    ap.add_argument("--clutter-window", type=float, default=8.0,
                    help="persistence window, seconds")
    ap.add_argument("--clutter-persist", type=int, default=8,
                    help="frames-in-window at which a place becomes clutter")
    ap.add_argument("--clutter-extent", type=float, default=40.0,
                    help="max px spread of a place's detections for it to "
                         "still count as static")
    ap.add_argument("--exempt-travel", type=float, default=150.0,
                    help="net track travel that exempts a detection from "
                         "clutter suppression")
    ap.add_argument("--gate-cap", type=float, default=250.0,
                    help="cap on the tracker's size term (see "
                         "CentroidTracker.max_size_gate_px)")
    ap.add_argument("--out", default=None, help="write results JSON here")
    args = ap.parse_args()

    clip = Path(args.clip)
    supp = (StaticClutterSuppressor(radius=args.clutter_radius,
                                    window_s=args.clutter_window,
                                    min_persist=args.clutter_persist,
                                    max_extent=args.clutter_extent)
            if args.clutter else None)
    labels, frames, rows = replay(clip, args.mode, args.tag, args.conf,
                                  args.stabilize, args.gate_cap, supp,
                                  args.exempt_travel)
    meta = json.loads((clip / "meta.json").read_text())
    minutes = len(frames) * meta.get("interval_s", 0.5) / 60.0
    vis = sum(1 for f in frames if labels[f]["visible"])

    print(f"\n=== track gating on {clip.name} / {args.mode}"
          f"{'/' + args.tag if args.tag else ''} @ conf {args.conf:.2f}, "
          f"gate cap {args.gate_cap:.0f}px"
          f"{', clutter map' if args.clutter else ''} ===")
    print(f"{len(frames)} frames, {vis} drone-visible, {minutes:.1f} min, "
          f"{len(rows)} track-frames")

    # Evidence for whether the gate can work at all: if drone tracks and
    # clutter tracks have the same travel distribution, no threshold helps.
    for key, name in (("travel", "raw travel"), ("travel_stab", "stabilized")):
        if key == "travel_stab" and not args.stabilize:
            continue
        for label, sel in (("drone", lambda r: r["is_drone"]),
                           ("clutter", lambda r: not r["is_drone"]
                            and not r["on_bird"])):
            v = sorted(r[key] for r in rows if sel(r))
            if not v:
                continue
            print(f"  {name} on {label:>7} track-frames (n={len(v):>5}): "
                  f"p10 {v[len(v) // 10]:>7.1f}  med {v[len(v) // 2]:>7.1f}  "
                  f"p90 {v[9 * len(v) // 10]:>7.1f}")

    keys = ["travel"] + (["travel_stab"] if args.stabilize else [])
    results = {}
    for key in keys:
        print(f"\n{'min ' + key:>16} {'recall':>8} {'FP':>6} {'FP/min':>9} "
              f"{'vs no gate':>11}")
        base = None
        for t in TRAVEL_SWEEP:
            s = score(labels, frames, rows, minutes, key, t,
                      args.min_hits, args.vote, not args.no_coast)
            if base is None:
                base = s
            drop = (1 - s["fp"] / base["fp"]) * 100 if base["fp"] else 0.0
            print(f"{t:>16.0f} {s['recall']:>8.3f} {s['fp']:>6} "
                  f"{s['fp_per_min']:>9.1f} {drop:>10.1f}%")
            results[f"{key}@{t:g}"] = s

    if args.out:
        Path(args.out).write_text(json.dumps(
            {"clip": clip.name, "mode": args.mode, "tag": args.tag,
             "conf": args.conf, "stabilize": args.stabilize,
             "min_hits": args.min_hits, "vote": args.vote,
             "gate_cap": args.gate_cap, "clutter": args.clutter,
             "clutter_radius": args.clutter_radius,
             "clutter_window": args.clutter_window,
             "clutter_persist": args.clutter_persist,
             "clutter_extent": args.clutter_extent,
             "frames": len(frames), "visible": vis, "minutes": round(minutes, 3),
             "results": results}, indent=2))
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
