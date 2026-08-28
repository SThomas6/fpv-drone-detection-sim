#!/usr/bin/env python3
"""Tell drones from birds using how they MOVE, not how they look.

Measured on this simulator: with birds in the air and no drone present, the
single-class detector raised ~662 false alarms per minute, and 99% of them
landed on a bird. Confidence cannot fix it — detections on birds average 0.36
against 0.57 for the drone, but the distributions overlap and birds reach 0.85,
higher than the drone ever scores. There is no threshold that separates them.

Appearance is exhausted anyway: past ~150 m the target is 2-3 px, which is
simply not enough pixels to tell a quadcopter from a gull. A second appearance
model would be guessing on the same pixels for half the frame rate.

Motion is still there, and it is different in kind:

  · a multirotor can HOVER and hold station; a bird essentially cannot
  · a multirotor flies straight legs between waypoints; birds wheel and soar
  · a flapping bird's projected width oscillates at a few Hz; a rigid airframe's
    does not

So this classifies a TRACK, not a frame, from features the tracker already
records. Cheap: a handful of numpy operations on a short history window.

    python camera/classify.py train --drone data/clips/baseline data/clips/range_sweep \\
                                    --bird data/clips/birds_only
    python camera/classify.py eval  --clip data/clips/birds_drone
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

MODEL_PATH = Path(__file__).resolve().parent / "motion_classifier.json"
MIN_HISTORY = 8          # samples needed before a verdict is offered

FEATURE_NAMES = [
    "straightness",      # net displacement / path length: 1 = straight line
    "hover_fraction",    # share of samples barely moving relative to own size
    "width_cv",          # projected-size variability — the flapping signature
    "turn_rate",         # mean absolute heading change per second (radians)
    "speed_cv",          # speed variability
    "speed_in_widths",   # speed in target-widths per second (scale-free)
]


def features_from_history(history) -> np.ndarray | None:
    """Motion features from a track history of (t, x, y, width, conf) samples."""
    if history is None or len(history) < MIN_HISTORY:
        return None
    h = np.asarray([(t, x, y, w) for (t, x, y, w, *_rest) in history], dtype=float)
    t, x, y, w = h[:, 0], h[:, 1], h[:, 2], h[:, 3]
    dt = np.diff(t)
    good = dt > 1e-6
    if good.sum() < MIN_HISTORY - 2:
        return None
    dx, dy = np.diff(x)[good], np.diff(y)[good]
    dt = dt[good]

    step = np.hypot(dx, dy)
    path_len = float(step.sum())
    net = float(math.hypot(x[-1] - x[0], y[-1] - y[0]))
    straightness = net / path_len if path_len > 1e-6 else 1.0

    speed = step / dt
    mean_w = float(np.mean(w)) or 1.0
    speed_widths = speed / max(mean_w, 1.0)
    hover_fraction = float(np.mean(speed_widths < 1.0))
    speed_cv = float(np.std(speed) / (np.mean(speed) + 1e-6))
    width_cv = float(np.std(w) / (np.mean(w) + 1e-6))

    heading = np.arctan2(dy, dx)
    dh = np.diff(heading)
    dh = (dh + np.pi) % (2 * np.pi) - np.pi          # wrap to [-pi, pi]
    turn_rate = float(np.mean(np.abs(dh)) / np.mean(dt)) if len(dh) else 0.0

    return np.array([straightness, hover_fraction, width_cv,
                     min(turn_rate, 50.0), min(speed_cv, 10.0),
                     min(float(np.mean(speed_widths)), 50.0)], dtype=float)


def drone_vote_fraction(history) -> float:
    """Share of a track's detections the detector called 'drone'.

    With a two-class detector a 4 px target flip-flops between classes frame to
    frame; no single frame is meaningful but the aggregate is. Histories from a
    single-class detector carry no class entry and count as all-drone, so this
    is 1.0 there and the vote gate becomes a no-op.
    """
    if not history:
        return 1.0
    votes = [h[5] if len(h) > 5 else "drone" for h in history]
    return sum(1 for v in votes if v == "drone") / len(votes)


class MotionClassifier:
    """Logistic regression on motion features. P(drone) given a track.

    Deliberately tiny and interpretable: six features, one weight each, fitted
    with plain gradient descent. No scipy or sklearn dependency, and the learned
    weights can be read and sanity-checked by a human.
    """

    def __init__(self, weights=None, bias=0.0, mean=None, std=None):
        self.w = None if weights is None else np.asarray(weights, dtype=float)
        self.b = float(bias)
        self.mean = None if mean is None else np.asarray(mean, dtype=float)
        self.std = None if std is None else np.asarray(std, dtype=float)

    # -------- training --------

    def fit(self, X, y, epochs: int = 4000, lr: float = 0.2, l2: float = 1e-3):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        self.mean = X.mean(axis=0)
        self.std = X.std(axis=0) + 1e-6
        Z = (X - self.mean) / self.std
        n, d = Z.shape
        self.w = np.zeros(d)
        self.b = 0.0
        # class weights: bird tracks vastly outnumber drone tracks
        pos = max(float(y.sum()), 1.0)
        neg = max(float(n - y.sum()), 1.0)
        wt = np.where(y > 0.5, n / (2 * pos), n / (2 * neg))
        for _ in range(epochs):
            p = 1.0 / (1.0 + np.exp(-(Z @ self.w + self.b)))
            g = (p - y) * wt
            self.w -= lr * ((Z.T @ g) / n + l2 * self.w)
            self.b -= lr * float(g.mean())
        return self

    # -------- inference --------

    def probability(self, feats) -> float:
        if self.w is None or feats is None:
            return 0.5
        z = (np.asarray(feats, dtype=float) - self.mean) / self.std
        return float(1.0 / (1.0 + np.exp(-(z @ self.w + self.b))))

    def classify_track(self, history, threshold: float = 0.5):
        """Returns (label, p_drone). label is 'drone', 'bird' or 'unknown'."""
        f = features_from_history(history)
        if f is None:
            return "unknown", 0.5
        p = self.probability(f)
        return ("drone" if p >= threshold else "bird"), p

    # -------- persistence --------

    def save(self, path=MODEL_PATH):
        Path(path).write_text(json.dumps({
            "features": FEATURE_NAMES, "weights": self.w.tolist(),
            "bias": self.b, "mean": self.mean.tolist(), "std": self.std.tolist(),
        }, indent=2))

    @classmethod
    def load(cls, path=MODEL_PATH):
        p = Path(path)
        if not p.exists():
            return cls()
        d = json.loads(p.read_text())
        return cls(d["weights"], d["bias"], d["mean"], d["std"])


# ---------------------------------------------------------------- data

def tracks_from_clip(clip: Path, want: str, conf: float = 0.10):
    """Replay a clip through detector+tracker, yielding (history, label).

    `want` is 'drone' or 'bird': tracks are labelled by which ground-truth
    object they follow, so a clip containing both contributes both.
    """
    from PIL import Image
    from camera.detector import DroneDetector
    from camera.evaluate import hits_object, is_hit
    from camera.tracking import CentroidTracker

    labels = [json.loads(l) for l in open(clip / "labels.jsonl")]
    det = DroneDetector(conf=conf)
    det.warmup()
    tracker = CentroidTracker()

    out = []
    for rec in labels:
        frame = np.array(Image.open(clip / "frames" / rec["frame"]).convert("RGB"))
        # Track EVERY class. A two-class detector flip-flops on a tiny target;
        # dropping the bird-labelled frames here would punch holes in the very
        # track whose motion is being learned.
        for tr in tracker.update(det.detect(frame), timestamp=rec["t"]):
            if len(tr.history) < MIN_HISTORY:
                continue
            is_drone = rec["visible"] and is_hit(tr.box, rec)
            on_bird = any(hits_object(tr.box, b) for b in (rec.get("birds") or []))
            if want == "drone" and is_drone:
                out.append(list(tr.history))
            elif want == "bird" and on_bird and not is_drone:
                out.append(list(tr.history))
    return out


def build_dataset(drone_clips, bird_clips):
    X, y = [], []
    for c in drone_clips:
        hs = tracks_from_clip(Path(c), "drone")
        n = 0
        for h in hs:
            f = features_from_history(h)
            if f is not None:
                X.append(f)
                y.append(1.0)
                n += 1
        print(f"  {c}: {n} drone track samples", flush=True)
    for c in bird_clips:
        hs = tracks_from_clip(Path(c), "bird")
        n = 0
        for h in hs:
            f = features_from_history(h)
            if f is not None:
                X.append(f)
                y.append(0.0)
                n += 1
        print(f"  {c}: {n} bird track samples", flush=True)
    return np.array(X), np.array(y)


# ---------------------------------------------------------------- commands

def cmd_train(args):
    print("building training set (this replays clips through the detector)...")
    X, y = build_dataset(args.drone, args.bird)
    if len(X) < 20:
        raise SystemExit("not enough track samples to train")
    print(f"\n{len(X)} samples: {int(y.sum())} drone, {int(len(y) - y.sum())} bird")

    clf = MotionClassifier().fit(X, y)
    p = np.array([clf.probability(f) for f in X])
    pred = (p >= 0.5).astype(float)
    acc = float((pred == y).mean())
    drone_recall = float(pred[y == 1].mean()) if (y == 1).any() else 0.0
    bird_reject = float((1 - pred[y == 0]).mean()) if (y == 0).any() else 0.0
    print(f"train accuracy {acc:.3f} | drone kept {drone_recall:.3f} | "
          f"birds rejected {bird_reject:.3f}")
    print("\nlearned weights (positive pushes toward 'drone'):")
    for name, wv in sorted(zip(FEATURE_NAMES, clf.w), key=lambda kv: -abs(kv[1])):
        print(f"  {name:>16}: {wv:+.3f}")
    clf.save()
    print(f"\nwrote {MODEL_PATH}")


def cmd_eval(args):
    """Score the operational policy: declare a target only once there is
    enough track history AND the motion says drone.

    A three-frame track is not evidence of anything, so 'undecided' means NOT
    DECLARED rather than 'assume drone'. For the drone that is a short delay to
    first alarm, which is measured; for birds it removes the constant churn of
    brief spurious tracks.
    """
    from PIL import Image
    from camera.detector import DroneDetector
    from camera.evaluate import hits_object, is_hit
    from camera.tracking import CentroidTracker

    clip = Path(args.clip)
    clf = MotionClassifier.load()
    if clf.w is None:
        raise SystemExit("no trained classifier - run `train` first")

    labels = [json.loads(l) for l in open(clip / "labels.jsonl")]
    det = DroneDetector(conf=args.conf)
    det.warmup()
    tracker = CentroidTracker()

    # cache per frame so the threshold sweep costs no extra inference
    observations = []
    for rec in labels:
        frame = np.array(Image.open(clip / "frames" / rec["frame"]).convert("RGB"))
        # All classes are tracked (see tracks_from_clip); the per-frame class
        # becomes an appearance VOTE aggregated over the track below.
        for tr in tracker.update(det.detect(frame), timestamp=rec["t"]):
            is_drone = bool(rec["visible"] and is_hit(tr.box, rec))
            on_bird = any(hits_object(tr.box, b) for b in (rec.get("birds") or []))
            if not is_drone and not on_bird:
                continue
            f = features_from_history(tr.history)
            observations.append({
                "t": rec["t"], "is_drone": is_drone,
                "p": clf.probability(f) if f is not None else None,
                "vote": drone_vote_fraction(tr.history),
            })

    interval = json.loads((clip / "meta.json").read_text()).get("interval_s", 0.4)
    minutes = len(labels) * interval / 60.0
    raw_drone = sum(1 for o in observations if o["is_drone"])
    raw_bird = sum(1 for o in observations if not o["is_drone"])

    print(f"\n=== motion classifier on {clip.name} ===")
    print(f"{len(labels)} frames ({minutes:.1f} min). Before classification: "
          f"{raw_drone} drone track-frames, {raw_bird} bird false alarms "
          f"({raw_bird / minutes:.0f}/min)\n")
    def policy_row(thr, vote_thr):
        kept = sum(1 for o in observations
                   if o["is_drone"] and o["p"] is not None and o["p"] >= thr
                   and o["vote"] >= vote_thr)
        alarms = sum(1 for o in observations
                     if not o["is_drone"] and o["p"] is not None and o["p"] >= thr
                     and o["vote"] >= vote_thr)
        return kept, alarms

    print(f"{'thresh':>7} {'drone kept':>11} {'bird alarms':>12} {'alarms/min':>11} "
          f"{'reduction':>10}")
    for thr in (0.30, 0.50, 0.70, 0.80, 0.90, 0.95):
        kept, alarms = policy_row(thr, 0.0)
        red = (1 - alarms / raw_bird) * 100 if raw_bird else 0.0
        print(f"{thr:>7.2f} {kept / max(raw_drone, 1) * 100:>10.1f}% {alarms:>12} "
              f"{alarms / minutes:>11.1f} {red:>9.1f}%")

    # Appearance votes only exist with a two-class detector; on a single-class
    # model every vote is 1.0 and this table repeats the one above.
    if any(o["vote"] < 1.0 for o in observations):
        print(f"\nfused with appearance votes "
              f"(motion >= thresh AND track drone-vote >= 0.5):")
        print(f"{'thresh':>7} {'drone kept':>11} {'bird alarms':>12} "
              f"{'alarms/min':>11} {'reduction':>10}")
        for thr in (0.30, 0.50, 0.70, 0.80, 0.90, 0.95):
            kept, alarms = policy_row(thr, 0.5)
            red = (1 - alarms / raw_bird) * 100 if raw_bird else 0.0
            print(f"{thr:>7.2f} {kept / max(raw_drone, 1) * 100:>10.1f}% {alarms:>12} "
                  f"{alarms / minutes:>11.1f} {red:>9.1f}%")
        v_kept = sum(1 for o in observations if o["is_drone"] and o["vote"] >= 0.5)
        v_alarms = sum(1 for o in observations
                       if not o["is_drone"] and o["vote"] >= 0.5)
        print(f"\n(votes alone, no motion: drone kept "
              f"{v_kept / max(raw_drone, 1) * 100:.1f}%, "
              f"{v_alarms / minutes:.1f} alarms/min)")

    # time from the drone first appearing to the first confirmed declaration
    thr = args.threshold
    first_seen = next((o["t"] for o in observations if o["is_drone"]), None)
    first_call = next((o["t"] for o in observations if o["is_drone"]
                       and o["p"] is not None and o["p"] >= thr), None)
    if first_seen is not None and first_call is not None:
        print(f"\ndelay from first sighting to first confirmed call "
              f"(at {thr:.2f}): {first_call - first_seen:.1f} s")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--drone", nargs="+", required=True)
    t.add_argument("--bird", nargs="+", required=True)
    t.set_defaults(func=cmd_train)
    e = sub.add_parser("eval")
    e.add_argument("--clip", required=True)
    e.add_argument("--conf", type=float, default=0.10)
    e.add_argument("--threshold", type=float, default=0.5)
    e.set_defaults(func=cmd_eval)
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
