#!/usr/bin/env python3
"""Wind-driven vegetation suppression: long path, no net displacement.

The measured problem (data/clips/sway_canopy vs nosway_canopy): 40 swaying
crowns at 8 m/s produce ~39 movers per frame against 4.3 with no wind. The
shipped flood guard then blanks the frame, which is the right call
end-to-end but leaves the channel blind in wind.

Two things that do NOT work, both tried and measured:

  * a fixed threshold - the flood is real, so any threshold either blanks
    (drone 2%) or floods (11,000 alarms/min, fused coverage 57% -> 20%)
  * StaticClutterSuppressor's place-and-extent test - a crown oscillating
    +/-8-16 px at 44-92 m has the same EXTENT as a drone crossing the same
    neighbourhood, so the 60 px default mutes everything including the drone
    (0% found), r20/e14 drops it to 19%, and r12/e8 catches nothing at all

What actually separates them is not where the motion is or how far it
reaches, but its SHAPE IN TIME. A crown oscillates: it accumulates a long
path and returns to where it started. A drone translates: its net
displacement is most of its path. So the discriminant is the ratio

    straightness = |last - first| / (total path length)

which is ~0 for anything oscillating and ~1 for anything travelling, and is
scale-free - it does not care whether the wind is moving a crown 5 px or
25 px, which is exactly what a fixed extent threshold could not handle.

Evidence is accumulated per GRID CELL rather than per track, for the reason
clutter_map.py already documents: the tracker repeatedly loses and respawns
tracks on clutter, so no single track ever accumulates the evidence. Cells
are small (default 24 px) so a crown and a passing drone rarely share one.

The asymmetry that makes this work: a crown sits in its cell forever and
accumulates samples; a drone crosses a cell in a few frames and never
reaches `min_samples`, so it cannot be muted even in principle.

Causal throughout - the decision for frame t uses only frames < t.
"""

from __future__ import annotations

import math
from collections import defaultdict


class _Cell:
    __slots__ = ("samples",)

    def __init__(self):
        self.samples = []          # (t, x, y)

    def prune(self, now: float, window_s: float) -> None:
        cut = now - window_s
        while self.samples and self.samples[0][0] < cut:
            self.samples.pop(0)

    @property
    def n(self) -> int:
        return len(self.samples)

    def path_and_net(self):
        s = self.samples
        if len(s) < 2:
            return 0.0, 0.0
        path = 0.0
        for a, b in zip(s, s[1:]):
            path += math.hypot(b[1] - a[1], b[2] - a[2])
        net = math.hypot(s[-1][1] - s[0][1], s[-1][2] - s[0][2])
        return path, net


class VegetationSuppressor:
    """Mute movers whose motion accumulates path but no net displacement."""

    def __init__(self, cell_px: float = 24.0, window_s: float = 3.0,
                 min_samples: int = 6, min_path_px: float = 14.0,
                 max_straightness: float = 0.35, forget_s: float = 20.0):
        self.cell_px = cell_px
        self.window_s = window_s
        self.min_samples = min_samples
        self.min_path_px = min_path_px
        self.max_straightness = max_straightness
        self.forget_s = forget_s
        self._cells: dict[tuple[int, int], _Cell] = defaultdict(_Cell)
        self._last_seen: dict[tuple[int, int], float] = {}

    def _key(self, cx: float, cy: float):
        return (int(cx // self.cell_px), int(cy // self.cell_px))

    def is_vegetation(self, cx: float, cy: float) -> bool:
        c = self._cells.get(self._key(cx, cy))
        if c is None or c.n < self.min_samples:
            return False
        path, net = c.path_and_net()
        if path < self.min_path_px:
            return False          # barely moved at all: not our business
        return (net / path) <= self.max_straightness

    def step(self, blobs, t: float):
        """Decide, then learn. Returns (kept, suppressed)."""
        for k, c in list(self._cells.items()):
            c.prune(t, self.window_s)
            if not c.samples and t - self._last_seen.get(k, t) > self.forget_s:
                del self._cells[k]
                self._last_seen.pop(k, None)

        kept, dropped = [], []
        for b in blobs:
            cx, cy = b.centre
            (kept if not self.is_vegetation(cx, cy) else dropped).append(b)
        # learn from EVERY blob, including the ones just kept: a cell only
        # becomes vegetation by repeatedly failing to go anywhere, and a
        # drone's few passing samples raise its cell's straightness rather
        # than lowering it
        for b in blobs:
            cx, cy = b.centre
            k = self._key(cx, cy)
            self._cells[k].samples.append((t, cx, cy))
            self._last_seen[k] = t
        return kept, dropped

    @property
    def muted_cells(self) -> int:
        return sum(1 for k, c in self._cells.items()
                   if c.n >= self.min_samples
                   and self.is_vegetation((k[0] + 0.5) * self.cell_px,
                                          (k[1] + 0.5) * self.cell_px))
