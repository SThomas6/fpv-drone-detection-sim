#!/usr/bin/env python3
"""Fly a controlled test profile for Phase 2 evaluation.

Unlike the free-running mission in fly_mission.py, each profile puts the drone
at known, repeatable positions so the evaluation can build a detection-range
curve and cover every target-size bucket.

    python scripts/fly_test_profile.py --profile range_sweep

Geometry: the station is at the world origin and the camera looks along +X.
PX4's home is 50 m east of it, so world_x (the range) = 50 + px4_east, and
world_y (crossing axis) = px4_north.
"""

import argparse
import asyncio
import math
import sys

from mavsdk import System
from mavsdk.offboard import OffboardError, PositionNedYaw

HOME_RANGE = 50.0        # metres east of the station where PX4 home sits
CAM_HEIGHT = 2.5
CAM_PITCH = 0.15         # radians, nose-up

# Ranges chosen to span every size bucket: at 1108 px focal length a 0.34 m
# drone is ~94 px at 4 m, ~38 px at 10 m, ~9 px at 40 m, ~1.5 px at 250 m.
RANGE_STEPS = [8, 12, 20, 30, 40, 60, 80, 100, 130, 160, 200, 250]


def altitude_for(range_m: float) -> float:
    """Keep the target near the vertical centre of the camera's view."""
    return max(4.0, CAM_HEIGHT + math.tan(CAM_PITCH) * range_m)


async def connect() -> System:
    drone = System()
    print("[profile] connecting to PX4 ...", flush=True)
    await drone.connect(system_address="udpin://0.0.0.0:14540")
    async for state in drone.core.connection_state():
        if state.is_connected:
            break

    async def _ready():
        async for h in drone.telemetry.health():
            if h.is_global_position_ok and h.is_home_position_ok and h.is_armable:
                return
    await asyncio.wait_for(_ready(), timeout=120)
    return drone


async def position(drone: System):
    async for pv in drone.telemetry.position_velocity_ned():
        return pv.position


async def goto(drone: System, north, east, down, yaw=0.0,
               tol=3.0, timeout=90.0, settle=6.0, label=""):
    """Fly to a setpoint, wait until it is reached, then hold still."""
    await drone.offboard.set_position_ned(PositionNedYaw(north, east, down, yaw))
    loop = asyncio.get_event_loop()
    t0 = loop.time()
    while loop.time() - t0 < timeout:
        p = await position(drone)
        if math.dist((p.north_m, p.east_m, p.down_m), (north, east, down)) < tol:
            break
        await asyncio.sleep(0.5)
    rng = HOME_RANGE + east
    print(f"[profile] {label} at range {rng:.0f} m, alt {-down:.0f} m "
          f"— holding {settle:.0f}s", flush=True)
    await asyncio.sleep(settle)


async def takeoff(drone: System):
    print("[profile] arming + takeoff", flush=True)
    await drone.action.set_takeoff_altitude(15.0)
    await drone.action.arm()
    await drone.action.takeoff()
    await asyncio.sleep(12)
    await drone.offboard.set_position_ned(PositionNedYaw(0.0, 0.0, -15.0, 0.0))
    try:
        await drone.offboard.start()
    except OffboardError as e:
        print(f"[profile] offboard start failed: {e._result.result}", flush=True)
        await drone.action.land()
        raise SystemExit(1)


async def profile_range_sweep(drone: System):
    """Step straight out along the camera axis, holding at each range."""
    for rng in RANGE_STEPS:
        await goto(drone, north=0.0, east=rng - HOME_RANGE,
                   down=-altitude_for(rng), settle=7.0, label="range step")


async def profile_crossing(drone: System):
    """Traverse the field of view laterally at a few fixed ranges."""
    for rng in (40.0, 80.0, 120.0):
        alt = altitude_for(rng)
        half = 0.45 * rng * math.tan(1.047 / 2)   # stay inside the horizontal FOV
        for north in (-half, half, -half):
            await goto(drone, north=north, east=rng - HOME_RANGE, down=-alt,
                       tol=4.0, settle=2.0, label="crossing")


async def profile_low_altitude(drone: System):
    """Fly low so the drone appears against terrain rather than sky."""
    for rng in (30.0, 50.0, 80.0):
        for alt in (3.0, 6.0):
            await goto(drone, north=0.0, east=rng - HOME_RANGE, down=-alt,
                       settle=6.0, label="low (terrain background)")


async def profile_no_drone(drone: System):
    """Park the drone far outside the field of view (false-positive baseline)."""
    print("[profile] moving target out of view for a clean-sky baseline", flush=True)
    await goto(drone, north=-160.0, east=-40.0, down=-8.0, tol=6.0,
               settle=90.0, label="out of view")


PROFILES = {
    "range_sweep": profile_range_sweep,
    "crossing": profile_crossing,
    "low_altitude": profile_low_altitude,
    "no_drone": profile_no_drone,
}


async def run(name: str, land: bool):
    drone = await connect()
    await takeoff(drone)
    try:
        await PROFILES[name](drone)
        print(f"[profile] {name} complete", flush=True)
    finally:
        if land:
            try:
                await drone.offboard.stop()
            except OffboardError:
                pass
            await drone.action.land()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", required=True, choices=sorted(PROFILES))
    ap.add_argument("--no-land", action="store_true")
    args = ap.parse_args()
    try:
        asyncio.run(run(args.profile, land=not args.no_land))
    except KeyboardInterrupt:
        sys.exit(0)


if __name__ == "__main__":
    main()
