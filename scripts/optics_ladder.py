#!/usr/bin/env python3
"""How many camera zoom levels are worth having? Compute it, don't guess.

The tempting answer is "add three or four fixed cameras at different zooms".
This works out what each rung would actually buy, using thresholds MEASURED
in this project rather than assumed:

  * ~2.5 px against SKY - the tiled detector held 92% at 150-200 m where the
    target is 2.5 px (scripts/system_range_table.py).
  * ~8 px against CLUTTER - the telephoto point detector held 88-95% down to
    8.7 px against magnified rock and collapsed to 0% by 5.9 px
    (scripts/point_target_detect.py on range_tele).

That gap is the important part: the pixel threshold is not a property of the
lens, it is a property of the BACKGROUND. Against sky a target one tenth the
size is still detectable. So "more zoom" and "less clutter" buy the same
thing, and only one of them costs money.

Three costs are charged against each rung:
  RANGE    how far it detects (linear in focal length)
  SEARCH   a narrow lens sees less sky, so covering the wide camera's own
           field needs (60/fov)^2 pointings - the reason a long lens cannot
           be a search sensor, only a confirm sensor
  COMPUTE  a detector pass per camera per frame, measured at 26 ms native
           and 185 ms tiled on this box's RTX 3060 Ti

    python scripts/optics_ladder.py
"""

from __future__ import annotations

import argparse
import math

DRONE_M = 0.34          # airframe width used everywhere in this project
W_PX = 1280
NATIVE_MS = 26.0        # measured, camera/evaluate.py --mode full
TILED_MS = 185.0        # measured, --mode sahi


def fx_of(fov_deg):
    return (W_PX / 2) / math.tan(math.radians(fov_deg) / 2)


def range_at(fov_deg, px):
    """Range at which the airframe subtends `px` pixels."""
    return DRONE_M * fx_of(fov_deg) / px


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fovs", type=float, nargs="+",
                    default=[60, 40, 24, 12, 6, 3])
    ap.add_argument("--rate", type=float, default=15.0,
                    help="frame rate each camera would run at")
    args = ap.parse_args()

    print(f"\ntarget {DRONE_M} m, {W_PX} px sensor, thresholds measured in "
          f"this project\n")
    hdr = (f"{'FOV':>6} {'focal px':>9} {'range @2.5px':>13} "
           f"{'range @8px':>11} {'sky share':>10} {'pointings':>10} "
           f"{'GPU @15Hz':>10}")
    print(hdr)
    print("-" * len(hdr))
    for f in args.fovs:
        share = (f / 60.0) ** 2
        pts = max(1.0, 1.0 / share)
        gpu = NATIVE_MS * args.rate / 1000.0
        print(f"{f:>5.0f}d {fx_of(f):>9.0f} {range_at(f, 2.5):>11.0f} m "
              f"{range_at(f, 8.0):>9.0f} m {share:>9.1%} {pts:>10.0f} "
              f"{gpu:>9.0%}")

    print("\nsky share = fraction of the 60 deg camera's field this lens sees "
          "at one pointing")
    print("pointings = how many looks it needs to cover that same field")
    print(f"GPU       = one native detector pass per camera at {args.rate:.0f} "
          f"Hz, as a fraction of one GPU\n")

    # what a fixed staring stack costs versus what it buys
    for stack in ([60], [60, 24], [60, 24, 6], [60, 40, 24, 12, 6]):
        gpu = len(stack) * NATIVE_MS * args.rate / 1000.0
        best = max(range_at(f, 8.0) for f in stack)
        print(f"stack {str(stack):<22} {len(stack)} cameras  "
              f"GPU {gpu:>4.0%} of one card   clutter-limited reach "
              f"{best:>5.0f} m")

    print("\nThe same reach, bought by cueing instead of by staring:")
    gpu = (NATIVE_MS * args.rate + NATIVE_MS * 5) / 1000.0
    print(f"  wide 60 deg at {args.rate:.0f} Hz + ONE steerable 6 deg at 5 Hz"
          f"   GPU {gpu:.0%}   reach {range_at(6, 8.0):.0f} m")
    print("  ...because a 6 deg lens needs 100 pointings to search the wide "
          "field,\n  so a STARING long lens covers 1% of the sky and a "
          "STEERED one covers all of it.")


if __name__ == "__main__":
    main()
