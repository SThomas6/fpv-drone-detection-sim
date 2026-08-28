#!/usr/bin/env python3
"""Static-clutter suppression by image location, not by track.

Measured on the real Anti-UAV "medium" sequence (the 262 FP/min case): 218
false positives fall into 56 spatial clusters, and 22 of those clusters — 83%
of all the false alarms — fire repeatedly at ONE fixed image location, often
for twenty consecutive frames. They are foliage and building blur, confirmed
by an OSD-mask audit not to be the burned-in turret overlay. The drone passes
within 60 px of only 2 of the 22.

The obvious fix is the travel gate `camera/fuse_eval.py` already uses, but
applied per TRACK it underperforms here: the tracker repeatedly loses and
re-spawns tracks on these blobs, so no single track ever accumulates the
"this thing never went anywhere" evidence, and a loose association gate
instead chains distant blobs into one track with large apparent travel. The
evidence lives at the LOCATION, which outlives any individual track — so that
is where it should be accumulated.

A place is clutter when BOTH hold over the recent window:

    it keeps firing        — at least `min_persist` frames, and
    it never goes anywhere — those detections span at most `max_extent` px

Both conditions are load-bearing. Persistence alone mutes the target: a drone
crossing a 60 px neighbourhood at the measured ~18 px/frame is inside it for
about seven consecutive frames, which is not far from any useful persistence
threshold. Extent is what actually separates a bush from a drone, and it is
the same "static things do not travel" idea as the track-level gate, just
computed somewhere that survives the tracker losing its lock. Measured: with
persistence alone the drone is suppressed and recall collapses to 0.46; with
the extent condition it is not.

Causality is the whole ballgame. The decision for frame t uses only frames
< t, so this could run live on a stream and its measured numbers are not a
peek at the future. Nothing is suppressed during warm-up, which is honest: a
detector cannot know a bush is a bush until the bush has sat there a while.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field


@dataclass
class _Anchor:
    """One place, and the recent detections that landed on it."""
    x: float
    y: float
    samples: deque = field(default_factory=deque)   # (t, x, y), ascending
    last_seen: float = 0.0

    def prune(self, now: float, window_s: float) -> None:
        while self.samples and now - self.samples[0][0] > window_s:
            self.samples.popleft()
        if self.samples:
            self.x = sum(s[1] for s in self.samples) / len(self.samples)
            self.y = sum(s[2] for s in self.samples) / len(self.samples)

    @property
    def frames(self) -> int:
        return len({s[0] for s in self.samples})

    @property
    def extent(self) -> float:
        """Diagonal of the bounding box of recent detections at this place."""
        if len(self.samples) < 2:
            return 0.0
        xs = [s[1] for s in self.samples]
        ys = [s[2] for s in self.samples]
        return math.hypot(max(xs) - min(xs), max(ys) - min(ys))


class StaticClutterSuppressor:
    """Online map of places that keep producing detections and never move.

    Args:
        radius: how close a detection must be to count as the same place.
            Grows with box size — a 175 px blob's centre wanders more than a
            6 px one's.
        window_s: how far back the evidence looks.
        min_persist: frames within the window at which a place may be called
            clutter.
        max_extent: how far those detections may spread and still count as
            one static thing. Above this the place is something that moves.
        forget_s: an anchor unseen for this long is dropped, so a pan or a
            change of scene clears the map by itself.
    """

    def __init__(self, radius: float = 60.0, window_s: float = 8.0,
                 min_persist: int = 8, max_extent: float = 40.0,
                 forget_s: float = 30.0):
        self.radius = radius
        self.window_s = window_s
        self.min_persist = min_persist
        self.max_extent = max_extent
        self.forget_s = forget_s
        self._anchors: list[_Anchor] = []

    def _radius_for(self, width: float) -> float:
        return max(self.radius, 0.5 * width)

    def _nearest(self, cx: float, cy: float, width: float):
        r = self._radius_for(width)
        best, best_d = None, None
        for a in self._anchors:
            d = (a.x - cx) ** 2 + (a.y - cy) ** 2
            if d <= r * r and (best_d is None or d < best_d):
                best, best_d = a, d
        return best

    def is_clutter(self, cx: float, cy: float, width: float = 0.0) -> bool:
        a = self._nearest(cx, cy, width)
        return (a is not None and a.frames >= self.min_persist
                and a.extent <= self.max_extent)

    def observe(self, cx: float, cy: float, t: float, width: float = 0.0) -> None:
        """Record a detection as evidence about this place."""
        a = self._nearest(cx, cy, width)
        if a is None:
            a = _Anchor(cx, cy)
            self._anchors.append(a)
        a.samples.append((t, cx, cy))
        a.last_seen = t

    def step(self, detections, t: float, exempt=None):
        """Decide, then learn. Returns (kept, suppressed).

        `exempt(det) -> bool` marks detections the caller has independently
        shown to belong to a moving target; those are never suppressed, and
        never counted as evidence either, so a drone crossing a bush neither
        gets muted nor teaches the map a lie about that bush.
        """
        for a in self._anchors:
            a.prune(t, self.window_s)
        self._anchors = [a for a in self._anchors
                         if a.samples and t - a.last_seen <= self.forget_s]

        exempted = [exempt(d) if exempt is not None else False
                    for d in detections]
        kept, suppressed = [], []
        for d, ex in zip(detections, exempted):
            cx, cy = d.centre
            if ex or not self.is_clutter(cx, cy, d.width):
                kept.append(d)
            else:
                suppressed.append(d)
        for d, ex in zip(detections, exempted):
            if ex:
                continue
            cx, cy = d.centre
            self.observe(cx, cy, t, d.width)
        return kept, suppressed

    @property
    def anchors(self) -> list[tuple[float, float, int, float]]:
        """(x, y, frames-in-window, extent) for each live anchor."""
        return [(a.x, a.y, a.frames, a.extent) for a in self._anchors]
