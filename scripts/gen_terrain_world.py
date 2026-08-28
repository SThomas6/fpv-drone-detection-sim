#!/usr/bin/env python3
"""Generate the terrain world: field/forest environment for hard-background tests.

Produces simulator/worlds/detection_world_terrain.sdf plus its heightmap and
texture images. Everything is procedural and seeded, so the world is
reproducible from this script alone — no downloaded assets.

Layout (station at origin, camera looking +X):
  ·   0-170 m: flat meadow (grass texture, scattered bushes) so the near flight
      block keeps ground z≈0 and the existing label geometry stays valid.
  · 220-420 m: forested hillside (rising heightmap + conifer/broadleaf trees
      10-24 m tall). From the camera this puts CANOPY, not sky, behind a drone
      flying at 8-30 m altitude in the 30-160 m block — the hard case.
  · 450-800 m: bare rock rising to ~120 m — a mountain horizon band.

    python scripts/gen_terrain_world.py [--seed 0] [--trees 70]
"""

from __future__ import annotations

import argparse
import math
import random
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
WORLDS = ROOT / "simulator" / "worlds"
ASSETS = WORLDS / "terrain_assets"

# Heightmap footprint: 800x800 m centred 250 m east of the station.
HM_N = 257                  # gz heightmaps want 2^n + 1 samples per side
HM_SIZE = 800.0             # metres per side
HM_MAX = 120.0              # peak height, metres
HM_CENTRE_X = 250.0         # world x of heightmap centre


def value_noise(n, cells, rng, octaves=4):
    """Simple seeded multi-octave value noise in [0, 1]."""
    out = np.zeros((n, n))
    amp, total = 1.0, 0.0
    for o in range(octaves):
        c = cells * (2 ** o)
        grid = rng.random((c + 1, c + 1))
        idx = np.linspace(0, c, n)
        i0 = np.floor(idx).astype(int)
        f = idx - i0
        i1 = np.minimum(i0 + 1, c)
        fx = f[None, :]
        fy = f[:, None]
        g = (grid[np.ix_(i0, i0)] * (1 - fy) * (1 - fx)
             + grid[np.ix_(i0, i1)] * (1 - fy) * fx
             + grid[np.ix_(i1, i0)] * fy * (1 - fx)
             + grid[np.ix_(i1, i1)] * fy * fx)
        out += amp * g
        total += amp
        amp *= 0.5
    return out / total


def build_heightmap(seed):
    rng = np.random.default_rng(seed)
    noise = value_noise(HM_N, 4, rng)

    # world x for each column of the map (row axis = y, col axis = x)
    xs = np.linspace(HM_CENTRE_X - HM_SIZE / 2, HM_CENTRE_X + HM_SIZE / 2, HM_N)
    X = np.tile(xs, (HM_N, 1))

    # Height profile east of the station: flat, then hillside, then mountains.
    ramp = np.clip((X - 180.0) / (800.0 - 180.0), 0.0, 1.0) ** 1.6
    base = ramp * 1.0
    relief = noise * (0.06 + 0.94 * ramp)      # rougher as it rises
    h = np.clip(base * 0.55 + relief * 0.45, 0.0, 1.0)
    # hard-flatten the meadow so station and flight block sit at z≈0
    h[X < 170.0] *= 0.015
    return h


def ground_z(h, x, y):
    """Sample heightmap array at world (x, y) -> metres."""
    gx = (x - (HM_CENTRE_X - HM_SIZE / 2)) / HM_SIZE * (HM_N - 1)
    gy = (y + HM_SIZE / 2) / HM_SIZE * (HM_N - 1)
    gx = np.clip(gx, 0, HM_N - 1)
    gy = np.clip(gy, 0, HM_N - 1)
    return float(h[int(round(gy)), int(round(gx))]) * HM_MAX


def noise_texture(path, base_rgb, var, n=256, seed=0, blotch=True):
    rng = np.random.default_rng(seed)
    img = np.tile(np.array(base_rgb, dtype=float), (n, n, 1))
    grain = rng.normal(0, var, (n, n, 1))
    img += grain * np.array([1.0, 1.0, 0.8])
    if blotch:
        blot = value_noise(n, 6, rng)[..., None]
        img *= 0.82 + 0.36 * blot
    Image.fromarray(np.clip(img, 0, 255).astype(np.uint8)).save(path)


def tree_sdf(name, x, y, z, kind, h, r, rng):
    green = rng.uniform(0.18, 0.42)
    c = f"{green * 0.45:.3f} {green:.3f} {green * 0.4:.3f} 1"
    trunk_c = "0.28 0.2 0.12 1"
    trunk_h = h * 0.35
    parts = [f"""
      <visual name="trunk">
        <pose>0 0 {trunk_h / 2:.2f} 0 0 0</pose>
        <geometry><cylinder><radius>{r * 0.12:.2f}</radius><length>{trunk_h:.2f}</length></cylinder></geometry>
        <material><ambient>{trunk_c}</ambient><diffuse>{trunk_c}</diffuse></material>
      </visual>"""]
    if kind == "conifer":
        for i, (frac, rr) in enumerate([(0.45, 1.0), (0.65, 0.75), (0.85, 0.45)]):
            parts.append(f"""
      <visual name="canopy{i}">
        <pose>0 0 {h * frac:.2f} 0 0 0</pose>
        <geometry><cone><radius>{r * rr:.2f}</radius><length>{h * 0.42:.2f}</length></cone></geometry>
        <material><ambient>{c}</ambient><diffuse>{c}</diffuse></material>
      </visual>""")
    else:
        for i in range(3):
            dx, dy = rng.uniform(-r * 0.3, r * 0.3), rng.uniform(-r * 0.3, r * 0.3)
            parts.append(f"""
      <visual name="canopy{i}">
        <pose>{dx:.2f} {dy:.2f} {h * rng.uniform(0.55, 0.8):.2f} 0 0 0</pose>
        <geometry><sphere><radius>{r * rng.uniform(0.6, 0.95):.2f}</radius></sphere></geometry>
        <material><ambient>{c}</ambient><diffuse>{c}</diffuse></material>
      </visual>""")
    return f"""
    <model name="{name}">
      <static>true</static>
      <pose>{x:.2f} {y:.2f} {z:.2f} 0 0 {rng.uniform(0, 6.28):.2f}</pose>
      <link name="link">{''.join(parts)}
      </link>
    </model>"""


def bush_sdf(name, x, y, z, rng, temp_k=None):
    green = rng.uniform(0.15, 0.35)
    c = f"{green * 0.5:.3f} {green:.3f} {green * 0.35:.3f} 1"
    s = rng.uniform(0.3, 1.2)
    thermal = ("" if temp_k is None else f"""
          <plugin filename="gz-sim-thermal-system" name="gz::sim::systems::Thermal">
            <temperature>{temp_k:.1f}</temperature>
          </plugin>""")
    return f"""
    <model name="{name}">
      <static>true</static>
      <pose>{x:.2f} {y:.2f} {z + s * 0.35:.2f} 0 0 0</pose>
      <link name="link">
        <visual name="v">
          <geometry><sphere><radius>{s:.2f}</radius></sphere></geometry>
          <material><ambient>{c}</ambient><diffuse>{c}</diffuse></material>{thermal}
        </visual>
      </link>
    </model>"""


AMBIENT_K = 288.0

THERMAL_SENSOR = """
        <sensor name="station_thermal" type="thermal">
          <topic>/detection_station/thermal/image</topic>
          <update_rate>15</update_rate>
          <always_on>true</always_on>
          <visualize>false</visualize>
          <camera>
            <horizontal_fov>0.4189</horizontal_fov>
            <image>
              <width>640</width>
              <height>512</height>
              <format>L16</format>
            </image>
            <clip>
              <near>0.1</near>
              <far>1000</far>
            </clip>
          </camera>
          <plugin filename="gz-sim-thermal-sensor-system"
                  name="gz::sim::systems::ThermalSensor">
            <min_temp>253.15</min_temp>
            <max_temp>673.15</max_temp>
            <resolution>0.01</resolution>
          </plugin>
        </sensor>"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--trees", type=int, default=70)
    ap.add_argument("--bushes", type=int, default=55)
    ap.add_argument("--thermal", action="store_true",
                    help="add the station thermal camera (Boson-class 640x512 "
                         "24deg L16), ambient atmosphere temperature, and "
                         "sun-warmed clutter patches; writes *_ir.sdf")
    args = ap.parse_args()

    ASSETS.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    h = build_heightmap(args.seed)
    Image.fromarray((h * 65535).astype(np.uint16)).save(ASSETS / "heightmap.png")
    noise_texture(ASSETS / "grass.png", (86, 118, 60), 14, seed=args.seed)
    noise_texture(ASSETS / "rock.png", (117, 110, 100), 18, seed=args.seed + 1)
    flat = np.zeros((8, 8, 3), dtype=np.uint8)
    flat[..., 2] = 255
    flat[..., 0] = flat[..., 1] = 128
    Image.fromarray(flat).save(ASSETS / "flat_normal.png")

    models = []
    manifest = {"heightmap": {"size": HM_SIZE, "max": HM_MAX,
                              "centre_x": HM_CENTRE_X, "n": HM_N},
                "trees": []}
    # forested hillside: canopy behind the near flight block
    for i in range(args.trees):
        x = rng.uniform(220.0, 420.0)
        y = rng.uniform(-170.0, 170.0)
        z = ground_z(h, x, y)
        kind = "conifer" if rng.random() < 0.6 else "broadleaf"
        th = rng.uniform(10.0, 24.0)
        r = th * rng.uniform(0.16, 0.24)
        models.append(tree_sdf(f"tree_{i}", x, y, z, kind, th, r, rng))
        manifest["trees"].append({"x": x, "y": y, "z": z, "h": th, "r": r})
    # scattered near trees for edge clutter (off the direct flight corridor)
    for i in range(14):
        x = rng.uniform(35.0, 170.0)
        y = rng.choice([-1, 1]) * rng.uniform(18.0, 60.0)
        th = rng.uniform(6.0, 15.0)
        r = th * rng.uniform(0.18, 0.26)
        models.append(tree_sdf(f"neartree_{i}", x, y, 0.0,
                               "broadleaf" if rng.random() < 0.6 else "conifer",
                               th, r, rng))
        manifest["trees"].append({"x": x, "y": y, "z": 0.0, "h": th, "r": r})
    for i in range(args.bushes):
        x = rng.uniform(25.0, 215.0)
        y = rng.uniform(-70.0, 70.0)
        # In the thermal world, a quarter of the bushes are sun-warmed clutter
        # (rock-like ambient+10-20K) so hot-spot detection has honest
        # distractors instead of a uniformly cold background.
        temp = (AMBIENT_K + rng.uniform(10, 20)
                if args.thermal and rng.random() < 0.25 else None)
        models.append(bush_sdf(f"bush_{i}", x, y, ground_z(h, x, y), rng,
                               temp_k=temp))

    src = (WORLDS / "detection_world.sdf").read_text()
    # rename world; drop the flat ground plane (heightmap replaces it)
    out = src.replace('world name="detection_world"', 'world name="detection_world_terrain"')
    if args.thermal:
        out = out.replace('<atmosphere type="adiabatic"/>',
                          '<atmosphere type="adiabatic">'
                          f'<temperature>{AMBIENT_K:.0f}</temperature>'
                          '</atmosphere>')
        out = out.replace("</sensor>", "</sensor>" + THERMAL_SENSOR, 1)
    gp0 = out.index('<model name="ground_plane">')
    gp1 = out.index('</model>', gp0) + len('</model>')
    heightmap = f"""<model name="terrain">
      <static>true</static>
      <link name="link">
        <visual name="visual">
          <geometry>
            <heightmap>
              <uri>terrain_assets/heightmap.png</uri>
              <size>{HM_SIZE:.0f} {HM_SIZE:.0f} {HM_MAX:.0f}</size>
              <pos>{HM_CENTRE_X:.0f} 0 0</pos>
              <texture>
                <diffuse>terrain_assets/grass.png</diffuse>
                <normal>terrain_assets/flat_normal.png</normal>
                <size>18</size>
              </texture>
              <texture>
                <diffuse>terrain_assets/rock.png</diffuse>
                <normal>terrain_assets/flat_normal.png</normal>
                <size>30</size>
              </texture>
              <blend><min_height>28</min_height><fade_dist>18</fade_dist></blend>
            </heightmap>
          </geometry>
        </visual>
      </link>
    </model>"""
    out = out[:gp0] + heightmap + out[gp1:]
    out = out.replace("</world>", "".join(models) + "\n  </world>")
    dest = WORLDS / ("detection_world_terrain_ir.sdf" if args.thermal
                     else "detection_world_terrain.sdf")
    dest.write_text(out)
    import json
    np.save(ASSETS / "heightmap.npy", h)   # exact array for occlusion tests
    (WORLDS / "terrain_manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"wrote {dest} ({args.trees + 14} trees, {args.bushes} bushes"
          f"{', thermal camera ON' if args.thermal else ''}) "
          f"+ assets in {ASSETS} + terrain_manifest.json")


if __name__ == "__main__":
    main()
