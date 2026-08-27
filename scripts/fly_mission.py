#!/usr/bin/env python3
"""Autonomous flight pattern for the target drone (Phase 1).

Arms, takes off, then flies a repeating offboard pattern inside the
detection station camera's field of view (the camera at the origin looks
east; the drone's home is ~50 m east of the station). Land with Ctrl-C or
scripts/stop_sim.sh.

Waypoints are NED offsets from home: north +-18 m, east -20..+30 m
(i.e. 30-80 m from the station), altitude 5-30 m.
"""

import argparse
import asyncio
import sys

from mavsdk import System
from mavsdk.offboard import OffboardError, PositionNedYaw

# (north_m, east_m, down_m, yaw_deg) offsets from home
PATTERN = [
    (0.0, 0.0, -15.0, 0.0),
    (15.0, 20.0, -25.0, 45.0),
    (-15.0, 25.0, -10.0, 180.0),
    (10.0, -15.0, -30.0, 270.0),
    (-18.0, -10.0, -18.0, 90.0),
    (0.0, 10.0, -8.0, 0.0),
]
LEG_SECONDS = 12.0


async def wait_until_ready(drone: System, timeout_s: float = 120.0) -> None:
    async def _ready():
        async for health in drone.telemetry.health():
            if (health.is_global_position_ok and health.is_home_position_ok
                    and health.is_armable):
                return
    await asyncio.wait_for(_ready(), timeout=timeout_s)


async def run(loop_forever: bool) -> int:
    drone = System()
    print("[mission] connecting to PX4 on udpin://0.0.0.0:14540 ...", flush=True)
    await drone.connect(system_address="udpin://0.0.0.0:14540")
    async for state in drone.core.connection_state():
        if state.is_connected:
            break
    print("[mission] connected; waiting until armable...", flush=True)
    await wait_until_ready(drone)

    print("[mission] arming + takeoff", flush=True)
    await drone.action.set_takeoff_altitude(15.0)
    await drone.action.arm()
    await drone.action.takeoff()
    await asyncio.sleep(12)

    await drone.offboard.set_position_ned(PositionNedYaw(0.0, 0.0, -15.0, 0.0))
    try:
        await drone.offboard.start()
    except OffboardError as err:
        print(f"[mission] offboard start failed: {err._result.result}", flush=True)
        await drone.action.land()
        return 1

    print("[mission] flying pattern", flush=True)
    lap = 0
    try:
        while True:
            lap += 1
            for (n, e, d, yaw) in PATTERN:
                print(f"[mission] lap {lap} -> N{n:+.0f} E{e:+.0f} D{d:+.0f} yaw{yaw:.0f}", flush=True)
                await drone.offboard.set_position_ned(PositionNedYaw(n, e, d, yaw))
                await asyncio.sleep(LEG_SECONDS)
            if not loop_forever:
                break
    finally:
        print("[mission] stopping offboard, landing", flush=True)
        try:
            await drone.offboard.stop()
        except OffboardError:
            pass
        await drone.action.land()
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--loop", action="store_true", help="repeat pattern until killed")
    args = ap.parse_args()
    try:
        sys.exit(asyncio.run(run(args.loop)))
    except KeyboardInterrupt:
        sys.exit(0)


if __name__ == "__main__":
    main()
