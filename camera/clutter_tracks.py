#!/usr/bin/env python3
"""Wind suppression by tracking the CLUTTER, not the place it sits in.

Six approaches failed before this one, and they failed for a single reason
that was eventually measured rather than guessed. Extent, straightness,
persistence and per-cell periodicity all ask "what is the motion AT THIS
PLACE?" - and under wind a place has no single answer. Measured on
data/clips/sway_canopy: over 60 frames the flood puts detections in 88
distinct 40 px cells, the median cell fires in only 13 of those frames, and
up to 7 different movers share one cell in a single frame. A cell contains a
transient jumble of different blobs, so there is no trajectory there to
analyse, and every test that assumed there was one measured noise.

The fix is to give the clutter a trajectory before asking about it. A cheap
nearest-neighbour tracker associates movers frame to frame, and each
resulting CLUTTER TRACK is then a single object with a real path. Now the
question has an answer:

    a swaying crown  oscillates about a fixed point - its path is long, its
                     net displacement is near zero, and its displacement has
                     a dominant frequency around 0.5-2 Hz
    a drone          translates - net displacement is most of its path, and
                     there is no periodic component at all

This is deliberately NOT the main CentroidTracker. That one is tuned to hold
a target through occlusion, with generous gates and coasting, which is
exactly wrong here: it would chain distant foliage blobs into one long track
with large apparent travel. This tracker is mean on purpose - a tight gate,
no coasting, and a track dies the moment it is unmatched.

Causal throughout: the decision for frame t uses only frames <= t.
"""

from __future__ import annotations

import math

import numpy as np


class _Track:
    __slots__ = ("x", "y", "pts", "last_t", "misses", "id")
    _next = 0

    def __init__(self, x, y, t):
        self.x, self.y = x, y
        self.pts = [(t, x, y)]
        self.last_t = t
        self.misses = 0
        self.id = _Track._next
        _Track._next += 1

    def update(self, x, y, t):
        self.x, self.y = x, y
        self.pts.append((t, x, y))
        self.last_t = t
        self.misses = 0

    def prune(self, window_s):
        cut = self.last_t - window_s
        while len(self.pts) > 2 and self.pts[0][0] < cut:
            self.pts.pop(0)

    def path_net(self):
        p = self.pts
        if len(p) < 2:
            return 0.0, 0.0
        path = sum(math.hypot(b[1] - a[1], b[2] - a[2])
                   for a, b in zip(p, p[1:]))
        net = math.hypot(p[-1][1] - p[0][1], p[-1][2] - p[0][2])
        return path, net

    def peakiness(self):
        """Spectral concentration of the detrended displacement.

        Resampled onto a uniform grid first: a track only gets a point on
        frames where its blob was detected, so the series is irregular and an
        FFT over sample index would measure a time axis that does not exist.
        """
        p = self.pts
        if len(p) < 8:
            return 0.0, 0.0
        t = np.asarray([q[0] for q in p])
        if t[-1] <= t[0]:
            return 0.0, 0.0
        dt = float(np.median(np.diff(t)))
        if dt <= 0:
            return 0.0, 0.0
        grid = np.arange(t[0], t[-1] + 1e-9, dt)
        if len(grid) < 8:
            return 0.0, 0.0
        best, bf = 0.0, 0.0
        for idx in (1, 2):
            v = np.interp(grid, t, [q[idx] for q in p])
            v = v - v.mean()
            k = np.polyfit(np.arange(len(v)), v, 1)
            v = v - np.polyval(k, np.arange(len(v)))   # a ramp is not a tone
            if np.allclose(v, 0):
                continue
            sp = np.abs(np.fft.rfft(v * np.hanning(len(v))))[1:]
            if sp.size == 0:
                continue
            pk = float(sp.max() / (sp.mean() + 1e-12))
            if pk > best:
                best = pk
                bf = float(np.fft.rfftfreq(len(v), dt)[1:][int(np.argmax(sp))])
        return best, bf


class ClutterTrackSuppressor:
    """Associate movers into tracks, then mute the ones that go nowhere."""

    def __init__(self, gate_px: float = 22.0, window_s: float = 8.0,
                 min_pts: int = 10, min_path_px: float = 18.0,
                 max_straightness: float = 0.45,
                 min_peakiness: float = 2.6,
                 f_lo: float = 0.25, f_hi: float = 4.0,
                 max_tracks: int = 400):
        self.gate = gate_px
        self.window_s = window_s
        self.min_pts = min_pts
        self.min_path_px = min_path_px
        self.max_straightness = max_straightness
        self.min_peakiness = min_peakiness
        self.f_lo, self.f_hi = f_lo, f_hi
        self.max_tracks = max_tracks
        self._tracks: list[_Track] = []

    def _is_vegetation(self, tr: _Track) -> bool:
        if len(tr.pts) < self.min_pts:
            return False
        path, net = tr.path_net()
        if path < self.min_path_px:
            return False               # barely moved: not our business
        if net / path > self.max_straightness:
            return False               # it is going somewhere
        pk, f = tr.peakiness()
        if pk < self.min_peakiness:
            return False               # wanders, but not rhythmically
        return self.f_lo <= f <= self.f_hi

    def step(self, blobs, t: float):
        """Associate, decide, learn. Returns (kept, suppressed)."""
        cents = [b.centre for b in blobs]
        used_tr, assign = set(), {}
        # shortest link first, same discipline as the main tracker
        pairs = sorted(
            ((math.hypot(c[0] - tr.x, c[1] - tr.y), bi, ti)
             for bi, c in enumerate(cents)
             for ti, tr in enumerate(self._tracks)
             if math.hypot(c[0] - tr.x, c[1] - tr.y) <= self.gate),
            key=lambda z: z[0])
        used_b = set()
        for _, bi, ti in pairs:
            if bi in used_b or ti in used_tr:
                continue
            used_b.add(bi)
            used_tr.add(ti)
            assign[bi] = ti
            self._tracks[ti].update(cents[bi][0], cents[bi][1], t)

        kept, dropped = [], []
        for bi, b in enumerate(blobs):
            ti = assign.get(bi)
            if ti is not None and self._is_vegetation(self._tracks[ti]):
                dropped.append(b)
            else:
                kept.append(b)

        # unmatched blobs start new tracks; unmatched tracks die immediately
        for bi, c in enumerate(cents):
            if bi not in used_b and len(self._tracks) < self.max_tracks:
                self._tracks.append(_Track(c[0], c[1], t))
        for ti, tr in enumerate(self._tracks):
            if ti not in used_tr:
                tr.misses += 1
        self._tracks = [tr for tr in self._tracks if tr.misses <= 2]
        for tr in self._tracks:
            tr.prune(self.window_s)
        return kept, dropped

    @property
    def n_tracks(self) -> int:
        return len(self._tracks)
