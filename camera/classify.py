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


AUX_CLASSES = ("hotspot", "mover")

# ---------------------------------------------------------------- feature set v3
#
# Why a second feature set exists. Measured per-feature AUC for separating
# drone from bird track-windows (scripts/probe_track_features.py) showed the
# v1 set is not merely weak off-sky, it INVERTS. Its two largest weights are
# hover_fraction (+0.819) and straightness (+0.663); hover_fraction scores
# 0.961 on clear sky and 0.355 on canopy — the best feature in the domain it
# was fitted on, backwards in the domain it is applied to. That single fact
# explains the installed classifier's below-chance AUC 0.419 on canopy.
#
# The cause is the width normalisation: hover_fraction and speed_in_widths
# divide speed by apparent target width, and over terrain the birds are the
# BIGGER targets (width_mean AUC 0.188), so slow-looking birds masquerade as
# hovering drones. v3 therefore normalises by the track's OWN median step
# instead of by target size, which carries no cross-domain scale assumption.
#
# v3 admits only features whose AUC sits on the SAME side of 0.5 on all three
# held-out clips, plus the detector's own class votes — which measured as the
# strongest and most stable signal available (0.820 canopy, 0.978
# terrain+birds) and were previously used only as a hard gate, never given to
# the model to weigh.
#
# Deliberately EXCLUDED, with reasons, so nobody re-adds them by accident:
#   y_norm         AUC 0.915 on canopy and worthless elsewhere (0.54-0.57) —
#                  it is the drone's image height, i.e. memorised mission
#                  geometry. The most tempting feature here and the most wrong.
#   hover_fraction, speed_in_widths, vertical_ratio, support_rate — each
#                  strong in one domain and inverted or flat in another.
#   width_mean     strong (0.188/0.244) but it encodes "birds are nearer than
#                  the drone in these captures", a scene prior, not physics.
#   conf_mean, conf_cv — detector confidence, flips sign on sky.
FEATURE_NAMES_V3 = [
    # Class votes as EVIDENCE, not as a ratio. A ratio has to invent a value
    # for a track the detector never saw in RGB at all — and a neutral 0.5
    # there is not neutral in a linear model, it is a constant push. Measured:
    # with a 0.5 ratio the long-range sweep, where thermal carries the target
    # and RGB sees it in only 36.7% of frames, fell from 88.7% to 73.8%
    # coverage. Splitting the vote into two shares OF ALL SAMPLES makes an
    # aux-only track contribute exactly 0 to both, so it pushes neither way —
    # which is this project's standing convention that absence of an RGB
    # opinion is not evidence against (cf. `bird_vote is None` in bird-mute).
    "drone_evidence",    # RGB frames called 'drone'  / all samples
    "bird_evidence",     # RGB frames called 'bird'   / all samples
    "straightness",      # net displacement / path length, whole window
    "straight_short",    # same over the last 8 samples
    "turn_rate",         # mean |heading change| per second
    "speed_cv",          # speed variability
    "step_cv",           # step-length variability, width-free
    "dwell_fraction",    # steps below a quarter of this track's OWN median
    "width_cv",          # projected-size variability (birds flap: AUC < 0.5 everywhere)
    "width_trend",       # |slope of width| / mean width — a closing target
    "ir_frac",           # share of the track's samples the thermal channel fed
    "mv_frac",           # share the motion channel fed
]


def features_v3_from_history(history) -> np.ndarray | None:
    """Domain-stable track features. See FEATURE_NAMES_V3 for the rationale."""
    if history is None or len(history) < MIN_HISTORY:
        return None
    arr = np.asarray([(t, x, y, w, c) for (t, x, y, w, c, *_r) in history],
                     dtype=float)
    cls = [h[5] if len(h) > 5 else "drone" for h in history]
    t, x, y, w = arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3]
    dt = np.diff(t)
    good = dt > 1e-6
    if good.sum() < MIN_HISTORY - 4:
        return None
    dx, dy, dt = np.diff(x)[good], np.diff(y)[good], dt[good]

    step = np.hypot(dx, dy)
    path = float(step.sum())
    net = float(math.hypot(x[-1] - x[0], y[-1] - y[0]))
    speed = step / dt
    med_step = float(np.median(step)) or 1e-6

    heading = np.arctan2(dy, dx)
    dh = np.diff(heading)
    dh = (dh + np.pi) % (2 * np.pi) - np.pi

    k = min(8, len(x))
    p_short = float(np.hypot(np.diff(x[-k:]), np.diff(y[-k:])).sum())
    n_short = float(math.hypot(x[-1] - x[-k], y[-1] - y[-k]))

    mean_w = float(w.mean()) or 1.0
    span = float(t[-1] - t[0])
    n = len(cls)

    return np.array([
        sum(1 for c in cls if c == "drone") / n,
        sum(1 for c in cls if c == "bird") / n,
        net / path if path > 1e-6 else 1.0,
        n_short / p_short if p_short > 1e-6 else 1.0,
        min(float(np.mean(np.abs(dh)) / np.mean(dt)) if len(dh) else 0.0, 50.0),
        min(float(speed.std() / (speed.mean() + 1e-6)), 10.0),
        min(float(step.std() / (step.mean() + 1e-6)), 10.0),
        float(np.mean(step < 0.25 * med_step)),
        float(w.std() / (w.mean() + 1e-6)),
        min(float(abs(np.polyfit(t, w, 1)[0]) / mean_w) if span > 1e-6 else 0.0,
            10.0),
        sum(1 for c in cls if c == "hotspot") / len(cls),
        sum(1 for c in cls if c == "mover") / len(cls),
    ], dtype=float)


FEATURE_SETS = {
    "v1": (FEATURE_NAMES, features_from_history),
    "v3": (FEATURE_NAMES_V3, features_v3_from_history),
}


class MotionClassifier:
    """Logistic regression on motion features. P(drone) given a track.

    Deliberately tiny and interpretable: one weight per feature, fitted with
    plain gradient descent. No scipy or sklearn dependency, and the learned
    weights can be read and sanity-checked by a human.

    `feature_set` selects which extractor the model expects ('v1' — the
    original six sky-fitted motion features, or 'v3' — the domain-stable set
    above). It is stored in the JSON, so a saved model always knows how to
    featurise a track and old files without the key keep working as v1.
    """

    def __init__(self, weights=None, bias=0.0, mean=None, std=None,
                 feature_set: str = "v1"):
        self.w = None if weights is None else np.asarray(weights, dtype=float)
        self.b = float(bias)
        self.mean = None if mean is None else np.asarray(mean, dtype=float)
        self.std = None if std is None else np.asarray(std, dtype=float)
        self.feature_set = feature_set

    @property
    def extractor(self):
        return FEATURE_SETS[self.feature_set][1]

    @property
    def feature_names(self):
        return FEATURE_SETS[self.feature_set][0]

    def features(self, history):
        """Featurise a track history with THIS model's extractor.

        Always go through this rather than calling an extractor directly:
        it is what stops a v1 model being fed v3 vectors and silently
        producing nonsense.
        """
        return self.extractor(history)

    # -------- training --------

    def fit(self, X, y, epochs: int = 4000, lr: float = 0.2, l2: float = 1e-3):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        self.mean = X.mean(axis=0)
        # A feature that is CONSTANT in training must standardise to zero at
        # inference, not explode. With `std + 1e-6` a constant-zero feature
        # divides an inference-time value of 0.5 by 1e-6 — a 500000x input to
        # a linear model. It happens to be harmless today only because such a
        # feature also gets exactly zero gradient and keeps its zero initial
        # weight, so nothing multiplies the blow-up. That is luck, not design:
        # a NEARLY constant feature (tiny but non-zero variance) would get a
        # real weight and a huge amplification. This is live right now —
        # `ir_frac` is constant 0 across every training clip, because none of
        # them has a thermal stream, while the terrain_ir_* evaluation clips
        # feed it real values.
        raw_std = X.std(axis=0)
        self.std = np.where(raw_std < 1e-6, 1.0, raw_std)
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
        f = self.features(history)
        if f is None:
            return "unknown", 0.5
        p = self.probability(f)
        return ("drone" if p >= threshold else "bird"), p

    # -------- persistence --------

    def save(self, path=MODEL_PATH):
        Path(path).write_text(json.dumps({
            "feature_set": self.feature_set,
            "features": self.feature_names, "weights": self.w.tolist(),
            "bias": self.b, "mean": self.mean.tolist(), "std": self.std.tolist(),
        }, indent=2))

    @classmethod
    def load(cls, path=MODEL_PATH):
        p = Path(path)
        if not p.exists():
            return cls()
        d = json.loads(p.read_text())
        # Files written before the v3 feature set existed have no key and are
        # v1 by definition.
        return cls(d["weights"], d["bias"], d["mean"], d["std"],
                   d.get("feature_set", "v1"))


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
