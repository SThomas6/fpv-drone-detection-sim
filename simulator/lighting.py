#!/usr/bin/env python3
"""Change the world's lighting at runtime.

The Phase 2 results were all recorded under one fixed sun, which flatters the
detector. These presets vary sun elevation, azimuth and intensity so the same
flight can be evaluated in very different light — including the hard case of
shooting into a low sun.

    python simulator/lighting.py --preset dusk
    python simulator/lighting.py --list

Applies to the `sunUTC` directional light defined in detection_world.sdf.
"""

from __future__ import annotations

import argparse
import math
import sys

from gz.msgs10.boolean_pb2 import Boolean
from gz.msgs10.light_pb2 import Light
from gz.transport13 import Node

LIGHT_NAME = "sunUTC"

# elevation degrees above horizon, azimuth degrees (0 = sun behind the camera,
# looking east with it; 180 = sun ahead of the camera, backlighting targets),
# diffuse, specular, and a note.
PRESETS = {
    "midday":   (75.0, 20.0, 0.90, 0.27, "high sun, the Phase 2 baseline condition"),
    "morning":  (35.0, 30.0, 0.85, 0.25, "sun high-ish behind the station"),
    "backlit":  (20.0, 170.0, 0.95, 0.40, "low sun AHEAD of the camera — hardest case"),
    "dusk":     (7.0, 155.0, 0.45, 0.10, "very low, dim, warm light"),
    "overcast": (60.0, 90.0, 0.35, 0.02, "flat dim light, weak shadows"),
}


def direction_for(elev_deg: float, azim_deg: float):
    """Unit vector the light travels along (pointing down-range from the sun)."""
    e = math.radians(elev_deg)
    a = math.radians(azim_deg)
    return (math.cos(e) * math.cos(a), math.cos(e) * math.sin(a), -math.sin(e))


def apply(preset: str, world: str) -> bool:
    elev, azim, diffuse, specular, _ = PRESETS[preset]
    dx, dy, dz = direction_for(elev, azim)
    warm = preset in ("dusk", "backlit")

    msg = Light()
    msg.name = LIGHT_NAME
    msg.type = Light.DIRECTIONAL
    msg.direction.x, msg.direction.y, msg.direction.z = dx, dy, dz
    msg.diffuse.r = diffuse
    msg.diffuse.g = diffuse * (0.88 if warm else 1.0)
    msg.diffuse.b = diffuse * (0.72 if warm else 1.0)
    msg.diffuse.a = 1.0
    msg.specular.r = msg.specular.g = msg.specular.b = specular
    msg.specular.a = 1.0
    msg.range = 2000.0
    msg.attenuation_constant = 1.0
    msg.attenuation_linear = 0.0
    msg.attenuation_quadratic = 0.0
    msg.cast_shadows = True
    msg.intensity = 1.0

    node = Node()
    ok, rep = node.request(f"/world/{world}/light_config", msg, Light, Boolean, 3000)
    return bool(ok and rep.data)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", choices=sorted(PRESETS))
    ap.add_argument("--world", default="detection_world")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()

    if args.list or not args.preset:
        print("lighting presets:")
        for name, (e, a, d, s, note) in PRESETS.items():
            print(f"  {name:<9} elevation {e:>4.0f} deg, azimuth {a:>5.0f} deg, "
                  f"diffuse {d:.2f}  — {note}")
        return

    ok = apply(args.preset, args.world)
    print(f"[lighting] {args.preset}: {'applied' if ok else 'FAILED'}", flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
