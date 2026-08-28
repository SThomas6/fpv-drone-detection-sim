#!/usr/bin/env python3
"""Line-of-sight test against the terrain world's trees and heightmap.

A drone behind an oak still projects a bounding box; labelling it 'visible'
would poison both training and evaluation. This module answers, from the
terrain manifest the world generator writes, whether the station camera can
actually see a world point.

Approximations (adequate at these pixel scales): a tree occludes within a
vertical cylinder of its canopy radius spanning the upper 65% of its height;
terrain occludes where the sightline dips below the heightfield.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from simulator.station_camera import CAM_POS

WORLDS = Path(__file__).resolve().parent / "worlds"


class OcclusionChecker:
    def __init__(self, manifest_path=None, heightmap_path=None):
        manifest_path = manifest_path or WORLDS / "terrain_manifest.json"
        heightmap_path = heightmap_path or WORLDS / "terrain_assets" / "heightmap.npy"
        m = json.loads(Path(manifest_path).read_text())
        self.trees = m["trees"]
        hm = m["heightmap"]
        self.h = np.load(heightmap_path)
        self.size = hm["size"]
        self.max = hm["max"]
        self.cx = hm["centre_x"]
        self.n = hm["n"]

    @classmethod
    def if_available(cls, manifest_path=None, heightmap_path=None):
        try:
            return cls(manifest_path, heightmap_path)
        except (FileNotFoundError, KeyError):
            return None

    def ground_z(self, x, y):
        gx = (x - (self.cx - self.size / 2)) / self.size * (self.n - 1)
        gy = (y + self.size / 2) / self.size * (self.n - 1)
        if not (0 <= gx <= self.n - 1 and 0 <= gy <= self.n - 1):
            return 0.0
        return float(self.h[int(round(gy)), int(round(gx))]) * self.max

    def occluded(self, target_pos, samples: int = 60) -> bool:
        """True when the sightline camera->target is blocked."""
        a = np.asarray(CAM_POS, dtype=float)
        b = np.asarray(target_pos, dtype=float)
        ts = np.linspace(0.02, 0.98, samples)
        pts = a[None, :] + ts[:, None] * (b - a)[None, :]
        for x, y, z in pts:
            if z < self.ground_z(x, y) - 0.3:
                return True
        for tr in self.trees:
            lo = tr["z"] + 0.35 * tr["h"]
            hi = tr["z"] + tr["h"]
            band = pts[(pts[:, 2] >= lo) & (pts[:, 2] <= hi)]
            if len(band) and np.any(
                    np.hypot(band[:, 0] - tr["x"], band[:, 1] - tr["y"])
                    <= tr["r"]):
                return True
        return False
