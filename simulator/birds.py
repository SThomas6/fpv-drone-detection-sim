#!/usr/bin/env python3
"""Spawn and fly birds through the detection station's field of view.

Birds are the classic false-positive source for anti-drone systems, and at the
pixel scales that matter here they are genuinely confusable: a gull with a
1.2 m wingspan at 100 m is ~13 px across, while the 0.34 m drone at the same
range is only ~4 px. So the birds are, if anything, easier to see than the
target — exactly the pressure the detector should be tested under.

    python simulator/birds.py --count 6            # spawn and fly until stopped

Models are generated as SDF strings rather than files so wingspan, colour and
body proportions can be varied per bird.

Flight: birds soar on drifting circular paths with occasional glides, and roll
periodically. Rolling changes the projected width of the wings, which stands in
for wing-flap — real skeletal animation is not worth it when the target is a
handful of pixels across, but the periodic width change is exactly the cue a
motion-based classifier could later exploit.
"""

from __future__ import annotations

import argparse
import math
import random
import sys
import threading
import time

from gz.msgs10.boolean_pb2 import Boolean
from gz.msgs10.entity_factory_pb2 import EntityFactory
from gz.msgs10.pose_pb2 import Pose
from gz.transport13 import Node

UPDATE_HZ = 30.0

# Species-ish presets: (name, wingspan m, body length m, grey level)
SPECIES = [
    ("sparrow", 0.25, 0.14, 0.25),
    ("pigeon", 0.65, 0.32, 0.35),
    ("gull", 1.20, 0.55, 0.75),
    ("buzzard", 1.40, 0.60, 0.20),
    ("crow", 0.95, 0.45, 0.10),
]


def bird_sdf(name: str, span: float, body: float, grey: float) -> str:
    """A bird as a body capsule plus two swept wing plates."""
    c = f"{grey} {grey * 0.9} {grey * 0.8} 1"
    half = span / 2.0
    return f"""<?xml version="1.0" ?>
<sdf version="1.9">
  <model name="{name}">
    <static>false</static>
    <link name="body">
      <gravity>false</gravity>
      <inertial>
        <mass>0.5</mass>
        <inertia><ixx>0.01</ixx><iyy>0.01</iyy><izz>0.01</izz>
                 <ixy>0</ixy><ixz>0</ixz><iyz>0</iyz></inertia>
      </inertial>
      <visual name="torso">
        <geometry><box><size>{body} {body * 0.28} {body * 0.28}</size></box></geometry>
        <material><ambient>{c}</ambient><diffuse>{c}</diffuse></material>
      </visual>
      <visual name="wing_l">
        <pose>{-body * 0.05} {half / 2} 0 0 0 0.25</pose>
        <geometry><box><size>{body * 0.45} {half} {0.008}</size></box></geometry>
        <material><ambient>{c}</ambient><diffuse>{c}</diffuse></material>
      </visual>
      <visual name="wing_r">
        <pose>{-body * 0.05} {-half / 2} 0 0 0 -0.25</pose>
        <geometry><box><size>{body * 0.45} {half} {0.008}</size></box></geometry>
        <material><ambient>{c}</ambient><diffuse>{c}</diffuse></material>
      </visual>
      <visual name="tail">
        <pose>{-body * 0.55} 0 0 0 0 0</pose>
        <geometry><box><size>{body * 0.3} {body * 0.22} {0.006}</size></box></geometry>
        <material><ambient>{c}</ambient><diffuse>{c}</diffuse></material>
      </visual>
    </link>
  </model>
</sdf>"""


class Bird:
    """A soaring bird on a drifting circular path."""

    def __init__(self, idx: int, rng: random.Random):
        species, span, body, grey = rng.choice(SPECIES)
        self.name = f"bird_{idx}_{species}"
        self.span, self.body, self.grey = span, body, grey
        # Circle centre placed across the camera's view volume (camera looks +X)
        self.cx = rng.uniform(35.0, 200.0)
        self.cy = rng.uniform(-45.0, 45.0)
        self.alt = rng.uniform(8.0, 55.0)
        self.radius = rng.uniform(8.0, 30.0)
        self.speed = rng.uniform(6.0, 14.0)          # metres/second
        self.phase = rng.uniform(0, 2 * math.pi)
        self.drift = rng.uniform(-1.2, 1.2)          # metres/second along +X
        self.climb = rng.uniform(-0.5, 0.5)
        self.flap_hz = rng.uniform(2.5, 7.0)
        self.roll_amp = rng.uniform(0.25, 0.7)       # radians

    def pose_at(self, t: float):
        omega = self.speed / max(self.radius, 1e-3)
        a = self.phase + omega * t
        x = self.cx + self.drift * t + self.radius * math.cos(a)
        y = self.cy + self.radius * math.sin(a)
        z = max(4.0, self.alt + self.climb * t + 2.0 * math.sin(0.2 * t))
        yaw = a + math.pi / 2                         # face along the path
        roll = self.roll_amp * math.sin(2 * math.pi * self.flap_hz * t)
        pitch = 0.1 * math.sin(0.7 * t)
        return (x, y, z), (roll, pitch, yaw)


def euler_to_quat(roll, pitch, yaw):
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    return (cr * cp * cy + sr * sp * sy, sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy, cr * cp * sy - sr * sp * cy)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--world", default="detection_world")
    ap.add_argument("--count", type=int, default=6)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--seconds", type=float, default=0.0, help="0 = until stopped")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    node = Node()
    birds = [Bird(i, rng) for i in range(args.count)]

    create_svc = f"/world/{args.world}/create"
    for b in birds:
        req = EntityFactory()
        req.sdf = bird_sdf(b.name, b.span, b.body, b.grey)
        (p, _), _ = (b.pose_at(0.0), None)
        req.pose.position.x, req.pose.position.y, req.pose.position.z = p
        req.pose.orientation.w = 1.0
        ok, rep = node.request(create_svc, req, EntityFactory, Boolean, 5000)
        print(f"[birds] spawn {b.name} (span {b.span} m): "
              f"{'ok' if ok and rep.data else 'FAILED'}", flush=True)

    set_svc = f"/world/{args.world}/set_pose"
    print(f"[birds] flying {len(birds)} birds", flush=True)
    t0 = time.time()
    try:
        while True:
            t = time.time() - t0
            if args.seconds and t > args.seconds:
                break
            for b in birds:
                (x, y, z), (roll, pitch, yaw) = b.pose_at(t)
                req = Pose()
                req.name = b.name
                req.position.x, req.position.y, req.position.z = x, y, z
                qw, qx, qy, qz = euler_to_quat(roll, pitch, yaw)
                req.orientation.w, req.orientation.x = qw, qx
                req.orientation.y, req.orientation.z = qy, qz
                node.request(set_svc, req, Pose, Boolean, 100)
            time.sleep(1.0 / UPDATE_HZ)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
