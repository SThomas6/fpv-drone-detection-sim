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

Two conditions, and PERSISTENCE is the load-bearing one. A crown occupies
its cell for the whole clip; a drone crosses a 40 px cell at ~4 px/frame in
about ten frames. Requiring 25 samples inside a 10 s window means a
transiting target can never qualify no matter how its straightness scores,
while a crown qualifies within seconds. The first tuning (24 px cells, 3 s,
6 samples) was too permissive on both counts and removed only half the wind
flood; it also split a single crown's swing across two cells, so neither
cell saw the full oscillation.

Causal throughout - the decision for frame t uses only frames < t.
"""

from __future__ import annotations

import math
from collections import defaultdict

import numpy as np


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

    def periodicity(self, min_bins: int = 12):
        """How PERIODIC this cell's motion is, and at what frequency.

        Geometry could not separate wind from a drone: five geometric tests
        (extent, straightness, persistence) all failed, because a swaying
        crown and a crossing drone occupy the same pixel scales. But they do
        NOT occupy the same TIME scales. A crown is a pendulum - its
        displacement has a dominant frequency around 0.5-2 Hz and returns to
        the same place every cycle. A drone crossing a cell has no periodic
        component at all; its displacement is a ramp.

        Scored as spectral peakiness: the largest bin of the detrended
        displacement spectrum over the mean bin. A pure oscillation
        concentrates its energy in one bin and scores high; a ramp spreads
        across all of them and scores near 1.
        """
        s = self.samples
        if len(s) < 4:
            return 0.0, 0.0
        t = np.asarray([p[0] for p in s])
        dt = np.median(np.diff(t)) if len(t) > 1 else 0.2
        if dt <= 0 or t[-1] <= t[0]:
            return 0.0, 0.0
        # RESAMPLE onto a uniform grid first. A cell only gets a sample on
        # frames where it happened to fire, so the series is irregularly
        # spaced, and an FFT over sample INDEX then measures a time axis that
        # does not exist. Measured: without this the peakiness test rejected
        # every cell and suppressed nothing at all (11,018 false/min, the
        # same as no suppression).
        grid = np.arange(t[0], t[-1] + 1e-9, dt)
        if len(grid) < min_bins:
            return 0.0, 0.0
        x = np.interp(grid, t, [p[1] for p in s])
        y = np.interp(grid, t, [p[2] for p in s])
        best_peak, best_f = 0.0, 0.0
        for v in (x, y):
            v = v - v.mean()
            # remove the linear trend, or a drone's ramp leaks into bin 1 and
            # masquerades as a very low-frequency oscillation
            if len(v) > 2:
                k = np.polyfit(np.arange(len(v)), v, 1)
                v = v - np.polyval(k, np.arange(len(v)))
            if np.allclose(v, 0):
                continue
            sp = np.abs(np.fft.rfft(v * np.hanning(len(v))))[1:]
            if sp.size == 0:
                continue
            pk = float(sp.max() / (sp.mean() + 1e-12))
            if pk > best_peak:
                best_peak = pk
                best_f = float(np.fft.rfftfreq(len(v), dt)[1:][int(np.argmax(sp))])
        return best_peak, best_f

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

    def __init__(self, cell_px: float = 40.0, window_s: float = 10.0,
                 min_samples: int = 25, min_path_px: float = 20.0,
                 max_straightness: float = 0.55, forget_s: float = 30.0,
                 min_peakiness: float = 3.0, f_lo: float = 0.3,
                 f_hi: float = 3.0):
        self.cell_px = cell_px
        self.window_s = window_s
        self.min_samples = min_samples
        self.min_path_px = min_path_px
        self.max_straightness = max_straightness
        self.forget_s = forget_s
        self.min_peakiness = min_peakiness
        self.f_lo, self.f_hi = f_lo, f_hi
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
        if (net / path) > self.max_straightness:
            return False
        if self.min_peakiness > 0.0:
            peak, f = c.periodicity()
            if peak < self.min_peakiness:
                return False      # moves about, but not RHYTHMICALLY
            if not (self.f_lo <= f <= self.f_hi):
                return False      # periodic at the wrong rate to be a crown
        return True

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
