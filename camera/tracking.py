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
        accel_sigma: process noise, as pixels/second^2 of expected manoeuvre.
    """

    def __init__(self, max_age: int = 15, min_hits: int = 2,
                 base_gate_px: float = 30.0, accel_sigma: float = 250.0,
                 class_consistent: bool = False,
                 suppress_spawn_near_coasting: bool = False):
        self.max_age = max_age
        self.min_hits = min_hits
        self.base_gate_px = base_gate_px
        self.accel_sigma = accel_sigma
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
        size_term = 4.0 * max(det.width, 2.0)
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
