#!/usr/bin/env python3
"""Microphone-array acoustic sensor model - Phase 3.

The passive early-warning layer the user's architecture puts BEFORE the
cameras: a small array of microphones listens for the harmonic buzz of
multirotor propellers and reports a BEARING (and rough elevation) to a
suspected drone, so the wide camera, zoom, IR and (last) radar know roughly
where to look before the target is even a resolvable speck. Completely
passive - it only listens.

Modelled the way ir_detector / radar_sim are: a sensor over simulator
ground truth with published-order performance and honest error, never an
oracle. The sim has no audio, so the model synthesises the detection
DECISION and the bearing measurement from geometry + literature figures.

What the physics gives us, and its limits (all reflected below):
  - A multirotor radiates a strong harmonic comb (blade-pass frequency and
    overtones, ~100-300 Hz fundamentals) that a bird does not - so unlike
    every camera stage, ACOUSTIC HAS NO BIRD PROBLEM. Its false alarms are
    other engines (not modelled here; no such sources in-sim).
  - Range is short and weather-dependent: reliable to ~150-300 m for a small
    quad in low wind, degrading fast beyond (inverse-square + air
    absorption). This is WHY it is early-warning cueing, not the detector.
  - A compact array gives good BEARING (few degrees) but poor RANGE, so the
    output is a direction to slew to, not a track. Bearing sigma grows as
    SNR falls with range.
  - Wind/traffic raise the noise floor; a --wind knob scales the effective
    range and the false-cue rate (the literature's main degradation).

Output: detections_acoustic.jsonl, one record per frame, each detection a
bearing cue expressed IN CAMERA PIXELS (azimuth -> image column via fx, a
wide elevation band) with cls 'acoustic', conf = modelled SNR. Consumers
treat it as a low-precision wide cue, never a box.

    python scripts/acoustic_sim.py run --clip data/clips/terrain_ir_sweep
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# small-quad acoustics in STILL, QUIET air (see docstring for provenance)
RANGE_FULL_M = 180.0     # Pd ~ high inside this
RANGE_MAX_M = 320.0      # essentially inaudible beyond
BEARING_SIGMA_DEG = 3.0  # at good SNR; scales up with range
ELEV_BAND_PX = 220.0     # elevation is poorly constrained by a ground array
FALSE_CUE_PER_MIN = 0.3  # non-drone acoustic cues in quiet conditions

# ---------------------------------------------------------------- environment
#
# The honest reason this table exists: the first version of this model had a
# single `wind` scalar and NOTHING else, so "is the microphone tested against
# wind, rain, leaves, general noise?" had to be answered "no". Each row is a
# distinct noise environment with its own effect, because they degrade the
# array differently and a single knob cannot represent that:
#
#   range_mult  - how far the drone stays audible (noise floor eats SNR)
#   bearing_mult- extra bearing error (turbulence at the mics, wet diaphragms,
#                 and decorrelated wind noise all smear the cross-correlation)
#   false_mult  - extra spurious cues per minute
#   band_note   - WHY, in spectral terms; wind/foliage energy is mostly low
#                 frequency and partly separable from the 100-300 Hz blade
#                 comb, rain is broadband and is not.
ENVIRONMENTS = {
    "still":       (1.00, 1.0, 1.0, "reference: light air, rural night"),
    "breeze":      (0.80, 1.2, 1.8, "5-15 km/h; low-freq wind noise at the "
                                    "mics, partly filtered below the comb"),
    "windy":       (0.50, 1.8, 4.0, "25-40 km/h; turbulence dominates, "
                                    "windshields saturate"),
    "leaves":      (0.72, 1.4, 3.0, "wind THROUGH foliage: broadband hiss "
                                    "plus incoherent point sources that mimic "
                                    "a moving emitter - the worst bearing "
                                    "case for its noise level"),
    "light_rain":  (0.65, 1.5, 2.5, "broadband splash noise overlapping the "
                                    "blade comb; drips on the housing"),
    "heavy_rain":  (0.35, 2.4, 5.0, "broadband floor swamps the comb; also "
                                    "the drone itself flies less"),
    "traffic":     (0.60, 1.3, 6.0, "engine harmonics are the ONE false "
                                    "source that genuinely looks like a "
                                    "multirotor to a harmonic detector"),
    "urban":       (0.55, 1.6, 7.0, "traffic + HVAC + reflections off "
                                    "buildings (multipath bearing error)"),
}


def run(clip: Path, seed: int, wind: float, env: str = "still"):
    labels = [json.loads(l) for l in open(clip / "labels.jsonl")]
    meta = json.loads((clip / "meta.json").read_text())
    fx = meta.get("fx", 1108.77)
    W = meta.get("width", 1280)
    H = meta.get("height", 720)
    interval = meta.get("interval_s", 0.5)
    rng = random.Random(seed)

    rmult, bmult, fmult, note = ENVIRONMENTS[env]
    # `wind` remains as a continuous extra on top of the named environment
    range_full = RANGE_FULL_M * rmult / (1.0 + 0.6 * wind)
    range_max = RANGE_MAX_M * rmult / (1.0 + 0.6 * wind)
    false_rate = FALSE_CUE_PER_MIN * fmult * (1.0 + 2.0 * wind)

    def pd_at(r):
        if r <= range_full:
            return 0.97
        if r >= range_max:
            return 0.0
        return 0.97 * (range_max - r) / (range_max - range_full)

    out = []
    for rec in labels:
        dets = []
        r = rec.get("range_m")
        if rec.get("visible") and rec.get("bbox") and r is not None:
            if rng.random() < pd_at(r):
                # bearing sigma grows as SNR falls with range
                snr = max(0.0, min(1.0, (range_max - r) / range_max))
                sig_deg = (BEARING_SIGMA_DEG * bmult
                           * (1.0 + 2.5 * (1.0 - snr)))
                # true azimuth from the drone's image column
                cx = (rec["bbox"][0] + rec["bbox"][2]) / 2
                az_true = math.atan2(cx - W / 2, fx)
                az = az_true + math.radians(rng.gauss(0.0, sig_deg))
                col = W / 2 + fx * math.tan(az)
                cy = H * rng.uniform(0.3, 0.6)   # weak elevation prior
                if 0 <= col < W:
                    dets.append({
                        "xyxy": [round(col - ELEV_BAND_PX / 2, 1),
                                 round(cy - ELEV_BAND_PX / 2, 1),
                                 round(col + ELEV_BAND_PX / 2, 1),
                                 round(cy + ELEV_BAND_PX / 2, 1)],
                        "conf": round(0.4 + 0.5 * snr, 3),
                        "cls": "acoustic",
                        "bearing_deg": round(math.degrees(az), 2),
                        "bearing_sigma_deg": round(sig_deg, 2)})
        # false cues (wind gusts, distant machinery) - NOT birds
        if rng.random() < false_rate * interval / 60.0:
            col = rng.uniform(0, W)
            dets.append({
                "xyxy": [round(col - ELEV_BAND_PX / 2, 1),
                         round(H * 0.45 - ELEV_BAND_PX / 2, 1),
                         round(col + ELEV_BAND_PX / 2, 1),
                         round(H * 0.45 + ELEV_BAND_PX / 2, 1)],
                "conf": round(0.4 + 0.2 * rng.random(), 3),
                "cls": "acoustic",
                "bearing_deg": round(math.degrees(
                    math.atan2(col - W / 2, fx)), 2),
                "bearing_sigma_deg": 6.0})
        out.append({"frame": rec["frame"], "detections": dets})

    dst = clip / "detections_acoustic.jsonl"
    with open(dst, "w") as fh:
        for r_ in out:
            fh.write(json.dumps(r_) + "\n")
    n = sum(len(r_["detections"]) for r_ in out)
    hit = sum(1 for r_, rec in zip(out, labels)
              if rec.get("visible") and any(
                  d["cls"] == "acoustic" for d in r_["detections"]))
    vis = sum(1 for rec in labels if rec.get("visible"))
    false_n = n - hit
    print(f"wrote {dst} [{env}: {note}] ({len(out)} frames, {n} cues; "
          f"drone heard in {hit}/{vis} = {hit / max(vis, 1):.1%} of visible "
          f"frames; ~{false_n} non-drone cues; wind={wind})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["run"])
    ap.add_argument("--clip", required=True)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--env", default="still", choices=sorted(ENVIRONMENTS),
                    help="named noise environment (see ENVIRONMENTS)")
    ap.add_argument("--wind", type=float, default=0.2,
                    help="0 = still, 1 = strong; scales range down, "
                         "false-cue rate up")
    args = ap.parse_args()
    run(Path(args.clip), args.seed, args.wind, args.env)


if __name__ == "__main__":
    main()
