#!/usr/bin/env python3
"""Centroid tracking for tiny targets.

Every tracker shipped with ultralytics (ByteTrack, BoT-SORT, OC-SORT...)
associates detections to tracks by IoU. That fails here: measured on the
station camera, the drone moves further than its own box width in 27% of
frames, so consecutive boxes do not overlap at all and IoU is exactly zero.
The result is an ID change roughly every 40 frames for a single drone.

This tracker associates on CENTRE DISTANCE against a constant-velocity Kalman
prediction instead, which is the right metric when targets are a few pixels
across and move several widths per frame. It also coasts through missed
detections, so a target that flickers keeps its identity.

Deliberately small: the station watches a handful of targets, not a crowd, so
greedy nearest-neighbour matching is optimal in practice and avoids a scipy
dependency.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class TrackState:
    """One tracked target."""
    track_id: int
    mean: np.ndarray            # [x, y, vx, vy] in pixels and pixels/second
    cov: np.ndarray
    confidence: float
    box: list[float]
    hits: int = 1
    misses: int = 0
    age: int = 0
    last_t: float = 0.0
    coasting: bool = False
    history: list = field(default_factory=list)
    # Measured slant range, from PCL. The image-plane filter has no range
    # state and cannot get one from a camera - two pixels and a bearing do
    # not constrain distance - so this is carried alongside rather than
    # fused into mean/cov. It is the one thing no passive optical channel
    # supplies, and what a hand-off to an effector actually needs.
    range_m: float | None = None
    range_sigma_m: float | None = None
    range_age: int = 0

    @property
    def position(self) -> tuple[float, float]:
        return (float(self.mean[0]), float(self.mean[1]))

    @property
    def velocity(self) -> tuple[float, float]:
        return (float(self.mean[2]), float(self.mean[3]))

    @property
    def width(self) -> float:
        return float(self.box[2] - self.box[0])

    def report(self) -> str:
        x, y = self.position
        vx, vy = self.velocity
        status = " (coasting)" if self.coasting else ""
        return (f"Target ID: {self.track_id}{status}\n"
                f"Position: ({x:.0f}, {y:.0f})\n"
                f"Velocity: ({vx:.1f}, {vy:.1f}) px/s\n"
                f"Confidence: {self.confidence:.2f}")


class CentroidTracker:
    """Constant-velocity Kalman tracking with distance-gated association.

    Args:
        max_age: frames a track may coast without a detection before deletion.
        min_hits: detections required before a track is reported as confirmed.
        base_gate_px: minimum association radius, in pixels.
            Raised 30 -> 60 for the 150 kph requirement: a
            crossing target at 500 m through the 6 deg lens
            moves 68 px/frame at 15 Hz, and at 30 the gate held
            only 40-50% above 15 px/frame. Widening it also CUT
            clutter tracks (canopy 13 -> 9/min, sky 11 -> 7/min)
            because an existing track absorbs a detection
            instead of a rival track spawning on it.
        accel_sigma: process noise, as pixels/second^2 of expected manoeuvre.
    """

    def __init__(self, max_age: int = 15, min_hits: int = 2,
                 base_gate_px: float = 60.0, accel_sigma: float = 250.0,
                 class_consistent: bool = False,
                 suppress_spawn_near_coasting: bool = False,
                 max_size_gate_px: float = 250.0):
        self.max_age = max_age
        self.min_hits = min_hits
        self.base_gate_px = base_gate_px
        self.accel_sigma = accel_sigma
        # The size term of the association gate is 4x the box width, which is
        # right for the 2-13 px targets this tracker was built for (a drone
        # that small crosses several of its own widths per frame) and absurd
        # once boxes get large: a 175 px blur blob on real footage buys a
        # 730 px radius, wider than a third of the frame, and unrelated
        # clutter detections chain into one wandering track. Capping the term
        # keeps association scale-appropriate at both ends. The 250 px default
        # is a measured no-op for every sim clip in this project (widest
        # cached detection is 61 px -> a 244 px term); real footage needs it
        # much tighter, so callers there pass their own value.
        self.max_size_gate_px = max_size_gate_px
        # Multi-stream fusion discipline (off by default — single-detector
        # behaviour is unchanged):
        # class_consistent: a large (>=8 px) detection whose class contradicts
        #   the track's established majority class may not join that track —
        #   this is what stops a nearby bird's boxes from poisoning the drone
        #   track's vote history. Below 8 px classes flip on real drones
        #   (measured), so small detections are exempt.
        # suppress_spawn_near_coasting: an unmatched detection inside a
        #   coasting track's gate does not spawn a duplicate track; the
        #   coasting track will re-acquire it instead.
        self.class_consistent = class_consistent
        self.suppress_spawn_near_coasting = suppress_spawn_near_coasting
        self._tracks: list[TrackState] = []
        self._next_id = 1
        self._last_t: Optional[float] = None

    @staticmethod
    def _majority_rgb_class(tr: "TrackState"):
        counts: dict[str, int] = {}
        for h in tr.history:
            if len(h) > 5 and h[5] not in ("hotspot", "mover"):
                counts[h[5]] = counts.get(h[5], 0) + 1
        if not counts or sum(counts.values()) < 4:
            return None
        top = max(counts, key=counts.get)
        return top if counts[top] > sum(counts.values()) / 2 else None

    # ---------------- Kalman internals ----------------

    @staticmethod
    def _predict_state(mean, cov, dt, accel_sigma):
        f = np.eye(4)
        f[0, 2] = dt
        f[1, 3] = dt
        # constant-acceleration process noise on a constant-velocity model
        g = np.array([[0.5 * dt * dt, 0.0], [0.0, 0.5 * dt * dt],
                      [dt, 0.0], [0.0, dt]])
        q = g @ (np.eye(2) * accel_sigma ** 2) @ g.T
        return f @ mean, f @ cov @ f.T + q

    @staticmethod
    def _update_bearing(mean, cov, meas_x, meas_var):
        """Kalman update from a BEARING-ONLY measurement.

        A microphone array measures direction, not position: it pins the
        image COLUMN and says nothing about the row. Feeding that in as a
        fake wide box would (a) drag the track's elevation toward the box
        centre and (b) claim a position certainty the sensor never had.
        The correct form is a 1-D observation, H = [1 0 0 0], so the filter
        sharpens x, leaves y alone, and - because the cross-covariance is
        respected - lets a confident azimuth also inform x-velocity.
        """
        h = np.zeros((1, 4))
        h[0, 0] = 1.0
        r = np.array([[meas_var]], dtype=float)
        s = h @ cov @ h.T + r
        k = cov @ h.T @ np.linalg.inv(s)
        y = np.array([meas_x], dtype=float) - h @ mean
        return mean + (k @ y).ravel(), (np.eye(4) - k @ h) @ cov

    def fuse_bearings(self, bearings, gate_px: float = 160.0):
        """Fold acoustic bearings into the live tracks.

        Call AFTER update() for the same frame: predict -> camera update ->
        bearing update is the correct sequential order. Each bearing is
        (column_px, sigma_px). A bearing is matched to the track whose
        predicted column is nearest within `gate_px`; unmatched bearings are
        NOT spawned as tracks (a direction alone cannot start a track - it
        has no elevation), they are returned so a caller can slew a camera.

        Returns (matched_track_ids, unmatched_bearings) - the unmatched list
        is the cueing signal: something is out there that nothing is tracking.
        """
        # Shortest link first, matching fuse_pcl and update(). Iterating
        # tracks in seniority order let the longest-lived track claim the
        # nearest bearing, so a stale clutter track could take the cue meant
        # for the real target - the same flaw that cost fuse_pcl 91 points of
        # ranged coverage before it was found there.
        pairs = sorted(
            ((abs(col - float(tr.mean[0])), ti, bi)
             for ti, tr in enumerate(self._tracks)
             for bi, (col, _sig) in enumerate(bearings)
             if abs(col - float(tr.mean[0])) <= gate_px),
            key=lambda x: x[0])
        matched, used, claimed = {}, set(), set()
        for _, ti, bi in pairs:
            if ti in claimed or bi in used:
                continue
            claimed.add(ti)
            used.add(bi)
            tr = self._tracks[ti]
            col, sig = bearings[bi]
            tr.mean, tr.cov = self._update_bearing(
                tr.mean, tr.cov, float(col), float(max(sig, 1.0)) ** 2)
            matched[tr.track_id] = float(sig)
        unmatched = [b for i, b in enumerate(bearings) if i not in used]
        return matched, unmatched

    def fuse_pcl(self, cues, gate_px: float = 160.0):
        """Fold passive-radar cues into the live tracks.

        A PCL dwell yields a bearing AND a slant range. The bearing goes
        through the same 1-D Kalman update as an acoustic bearing (H=[1 0 0
        0]: it sharpens image column and x-velocity and claims nothing about
        the row). The range is ATTACHED to the matched track instead, because
        the filter has no range dimension to fold it into.

        Cues are (column_px, sigma_px, range_m, range_sigma_m). Like
        bearings, an unmatched cue is never spawned as a track - PCL gives no
        elevation, so it cannot start one - but it is returned as a slew cue,
        and unlike acoustic it comes with a range, so the cue tells a zoom
        both where to look AND how far out to expect the target.
        """
        # Assign by DISTANCE, not by track seniority. Iterating tracks in
        # hits order let the longest-lived track claim the nearest cue first,
        # so a stale clutter track sitting within the gate stole the range
        # from the real target: measured, only 8.3% of tracked frames carried
        # a range where the cue was almost always available. Pair up shortest
        # link first instead, which is what update() already does.
        pairs = sorted(
            ((abs(c[0] - float(tr.mean[0])), ti, ci)
             for ti, tr in enumerate(self._tracks)
             for ci, c in enumerate(cues)
             if abs(c[0] - float(tr.mean[0])) <= gate_px),
            key=lambda x: x[0])
        matched, used, claimed = {}, set(), set()
        for _, ti, ci in pairs:
            if ti in claimed or ci in used:
                continue
            claimed.add(ti)
            used.add(ci)
            tr = self._tracks[ti]
            col, sig, rng_m, rng_sig = cues[ci]
            tr.mean, tr.cov = self._update_bearing(
                tr.mean, tr.cov, float(col), float(max(sig, 1.0)) ** 2)
            tr.range_m = float(rng_m)
            tr.range_sigma_m = float(rng_sig)
            tr.range_age = 0
            matched[tr.track_id] = float(rng_m)
        for tr in self._tracks:
            if tr.track_id not in matched and tr.range_m is not None:
                tr.range_age += 1        # stale: no PCL support this frame
        unmatched = [c for i, c in enumerate(cues) if i not in used]
        return matched, unmatched

    @staticmethod
    def _update_state(mean, cov, meas, meas_var):
        h = np.zeros((2, 4))
        h[0, 0] = h[1, 1] = 1.0
        r = np.eye(2) * meas_var
        s = h @ cov @ h.T + r
        k = cov @ h.T @ np.linalg.inv(s)
        y = meas - h @ mean
        return mean + k @ y, (np.eye(4) - k @ h) @ cov

    # ---------------- public API ----------------

    def update(self, detections, timestamp: float) -> list[TrackState]:
        """Advance the tracker one frame. Returns confirmed tracks."""
        dt = 1.0 / 15.0 if self._last_t is None else max(1e-3, timestamp - self._last_t)
        self._last_t = timestamp

        for tr in self._tracks:
            tr.mean, tr.cov = self._predict_state(tr.mean, tr.cov, dt, self.accel_sigma)
            tr.age += 1

        unmatched = list(range(len(detections)))
        pairs: list[tuple[TrackState, int]] = []

        # Greedy nearest-neighbour: with a handful of targets this matches the
        # optimal assignment, without pulling in scipy.
        for tr in sorted(self._tracks, key=lambda t: (-t.hits, t.misses)):
            best, best_d = None, None
            px, py = tr.mean[0], tr.mean[1]
            tr_cls = (self._majority_rgb_class(tr)
                      if self.class_consistent else None)
            for i in unmatched:
                d = detections[i]
                if (tr_cls is not None and d.width >= 8.0
                        and getattr(d, "cls_name", "drone")
                        not in (tr_cls, "hotspot", "mover")):
                    continue
                cx, cy = d.centre
                dist = float(np.hypot(cx - px, cy - py))
                gate = self._gate_for(tr, d, dt)
                if dist <= gate and (best_d is None or dist < best_d):
                    best, best_d = i, dist
            if best is not None:
                pairs.append((tr, best))
                unmatched.remove(best)

        matched_tracks = {id(tr) for tr, _ in pairs}
        for tr, di in pairs:
            det = detections[di]
            cx, cy = det.centre
            meas_var = max(1.5, (det.width * 0.5)) ** 2
            tr.mean, tr.cov = self._update_state(
                tr.mean, tr.cov, np.array([cx, cy], dtype=float), meas_var)
            tr.confidence = det.confidence
            tr.box = det.xyxy
            tr.hits += 1
            tr.misses = 0
            tr.coasting = False
            tr.last_t = timestamp
            # width and confidence are kept because motion-based classification
            # (camera/classify.py) needs the projected-size oscillation that a
            # flapping bird produces and a multirotor does not. The class name
            # is kept so a track can be judged by its per-frame appearance
            # votes: a two-class detector flip-flops on a 4 px target, and only
            # the aggregate over the track is meaningful.
            tr.history.append((timestamp, float(tr.mean[0]), float(tr.mean[1]),
                               float(det.width), float(det.confidence),
                               getattr(det, "cls_name", "drone")))
            del tr.history[:-90]

        for tr in self._tracks:
            if id(tr) not in matched_tracks:
                tr.misses += 1
                tr.coasting = True

        for i in unmatched:
            if self.suppress_spawn_near_coasting:
                d = detections[i]
                cx, cy = d.centre
                near_coasting = any(
                    tr.misses > 0 and float(np.hypot(
                        cx - tr.mean[0], cy - tr.mean[1]))
                    <= self._gate_for(tr, d, dt)
                    for tr in self._tracks if id(tr) not in matched_tracks)
                if near_coasting:
                    continue
            self._spawn(detections[i], timestamp)

        self._tracks = [t for t in self._tracks if t.misses <= self.max_age]
        confirmed = [t for t in self._tracks
                     if t.hits >= self.min_hits and t.misses <= self.max_age]
        # Best evidence first: freshly-updated tracks ahead of coasting ones, so
        # a consumer taking the first track gets the best-supported target.
        confirmed.sort(key=lambda t: (t.misses, -t.hits))
        return confirmed

    def _gate_for(self, tr: TrackState, det, dt: float) -> float:
        """Association radius: grows with target size, speed and time coasted."""
        speed = float(np.hypot(tr.mean[2], tr.mean[3]))
        size_term = min(4.0 * max(det.width, 2.0), self.max_size_gate_px)
        motion_term = speed * dt * 1.5
        coast_term = self.base_gate_px * tr.misses * 0.5
        return self.base_gate_px + size_term + motion_term + coast_term

    def _spawn(self, det, timestamp: float) -> None:
        cx, cy = det.centre
        mean = np.array([cx, cy, 0.0, 0.0], dtype=float)
        cov = np.diag([25.0, 25.0, 400.0, 400.0])
        tr = TrackState(track_id=self._next_id, mean=mean, cov=cov,
                        confidence=det.confidence, box=det.xyxy,
                        last_t=timestamp,
                        history=[(timestamp, cx, cy, det.width, det.confidence,
                                  getattr(det, "cls_name", "drone"))])
        self._next_id += 1
        self._tracks.append(tr)

    def reset(self) -> None:
        self._tracks.clear()
        self._next_id = 1
        self._last_t = None

    @property
    def active(self) -> list[TrackState]:
        return list(self._tracks)
