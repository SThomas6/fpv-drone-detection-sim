#!/usr/bin/env python3
"""Geometry of the detection station's camera.

Single source of truth for projecting world points into station-camera pixels,
shared by the clip recorder, the dataset recorder and the Phase 2 evaluation.

MUST stay in sync with simulator/worlds/detection_world.sdf. `check_intrinsics`
verifies the constants here against the camera_info topic at runtime.

Gazebo camera convention: the optical axis is +X of the sensor frame, +Y is
left, +Z is up.
"""

import math
import os

import numpy as np

# --- from detection_world.sdf: detection_station/camera_link + station_camera
CAM_POS = np.array([0.1, 0.0, 2.5])   # link pose, station model sits at origin
CAM_PITCH = -0.15                     # radians, nose-up about +Y
W, H = 1280, 720
# Overridable for narrow-FOV (zoom-lens) ablation worlds; must match the
# world's <horizontal_fov> or check_intrinsics will flag the mismatch.
HFOV = float(os.environ.get("STATION_HFOV", "1.047"))   # radians

# --- target_drone model extents (metres, half-sizes incl. propeller discs)
DRONE_HALF = np.array([0.17, 0.17, 0.04])

# --- bird wingspans (metres), keyed by the species in the model name.
# A gull at 100 m is ~13 px across; the drone at the same range is ~4 px, so
# birds are the LARGER target and a real test of discrimination.
BIRD_SPANS = {"sparrow": 0.25, "pigeon": 0.65, "gull": 1.20,
              "buzzard": 1.40, "crow": 0.95}


def bird_half_extents(name: str) -> np.ndarray:
    """Half-extents for a bird model, inferred from its species name."""
    span = next((s for key, s in BIRD_SPANS.items() if key in name), 0.6)
    return np.array([span * 0.45, span / 2.0, span * 0.06])

FX = (W / 2) / math.tan(HFOV / 2)     # ≈ 1108.5 px; square pixels, so fy == fx


def _rot_y(theta):
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def _quat_rot(q):
    """(w, x, y, z) -> 3x3 rotation matrix."""
    w, x, y, z = q
    n = math.sqrt(w * w + x * x + y * y + z * z)
    if n == 0:
        return np.eye(3)
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def world_to_camera(p_world):
    """World point -> camera frame (+X forward, +Y left, +Z up)."""
    return _rot_y(CAM_PITCH).T @ (np.asarray(p_world, dtype=float) - CAM_POS)


def project_point(p_world):
    """World point -> (u, v, range_m), or None if at/behind the image plane."""
    d = world_to_camera(p_world)
    if d[0] <= 0.5:
        return None
    return (W / 2 - FX * d[1] / d[0],
            H / 2 - FX * d[2] / d[0],
            float(np.linalg.norm(d)))


def ground_truth_bbox(pos, quat=(1.0, 0.0, 0.0, 0.0), half_extents=None):
    """Axis-aligned pixel bbox of an object.

    Projects the 8 corners of the object's oriented bounding box and takes
    their extent. Defaults to the target drone's size; pass `half_extents` for
    anything else (birds). Returns dict with xyxy, centre, range_m, px_width
    and `visible` (any part inside the image), or None if behind the camera.
    """
    centre = project_point(pos)
    if centre is None:
        return None
    half = DRONE_HALF if half_extents is None else np.asarray(half_extents, dtype=float)
    r = _quat_rot(quat)
    corners = []
    for sx in (-1, 1):
        for sy in (-1, 1):
            for sz in (-1, 1):
                offs = r @ (half * np.array([sx, sy, sz]))
                p = project_point(np.asarray(pos, dtype=float) + offs)
                if p is not None:
                    corners.append((p[0], p[1]))
    if not corners:
        return None
    xs, ys = [c[0] for c in corners], [c[1] for c in corners]
    x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
    return {
        "xyxy": [x1, y1, x2, y2],
        "centre": [centre[0], centre[1]],
        "range_m": centre[2],
        "px_width": x2 - x1,
        "px_height": y2 - y1,
        "visible": (x2 > 0 and x1 < W and y2 > 0 and y1 < H),
    }


def size_bucket(px_width):
    """Test-matrix bucket. <32 px is the regime that decides the model."""
    if px_width < 16:
        return "<16px"
    if px_width < 32:
        return "16-32px"
    if px_width < 96:
        return "32-96px"
    return ">96px"


def check_intrinsics(fx_reported, width, height, tol=1.0):
    """Compare this module's constants against a camera_info message."""
    problems = []
    if (width, height) != (W, H):
        problems.append(f"resolution {width}x{height} != module {W}x{H}")
    if abs(fx_reported - FX) > tol:
        problems.append(f"fx {fx_reported:.1f} != module {FX:.1f}")
    return problems
