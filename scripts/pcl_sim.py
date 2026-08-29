#!/usr/bin/env python3
"""Passive coherent location (passive radar) sensor model — Phase 5.

Listens to a transmitter it does not own (DVB-T, DAB, LTE, 5G), and looks for
the echo that transmitter's signal makes off the drone. It emits nothing, so
it cannot be located by an RF direction finder, and it does not care that the
threat is fibre-optic — it bounces signals off the airframe rather than
listening for the airframe's own radio.

Modelled the way camera/ir_detector.py and scripts/radar_sim.py are: a sensor
model over simulator ground truth, with published figures and honest failure
modes, never an oracle. What it measures per dwell is a BISTATIC range sum
and a bistatic Doppler — not a position — so a single receiver constrains the
target to an ellipsoid with the transmitter and receiver at the foci. Bearing
comes separately, and coarsely, from a small receive array.

Three failure modes are modelled because they dominate real performance and
none of them appear in the radar equation:

  1. THE DIRECT SIGNAL, not thermal noise, sets sensitivity. The echo off a
     0.01 m^2 micro-UAV at 1.5 km is ~95 dB BELOW the direct path arriving
     straight into the receiver, and cancellation (ECA/CLEAN) buys a finite
     70-90 dB. What is left, plus its multipath sidelobes smeared across
     range-Doppler cells, is the floor.
     A from-scratch budget of this is NOT trustworthy and this model does not
     pretend otherwise: written out naively it predicted detection to 5 km
     with 34 dB to spare, which contradicts every published micro-UAV field
     result. The residual-clutter term that actually dominates is not
     analytically tractable. So the ABSOLUTE range scale is CALIBRATED to
     published performance (--anchor-km, default 1.5 km on a 0.01 m^2 target
     with 80 dB cancellation) and the physics is used only for RELATIVE
     scaling — how detection moves with RCS, illuminator power, cancellation
     and geometry. Same discipline as radar_sim.py taking its Pd from the
     literature rather than inventing it.
  2. THE ZERO-DOPPLER RIDGE. A target whose bistatic Doppler is near zero
     sits in the same bin as the direct signal and all static clutter, and is
     unrecoverable. That happens whenever the target moves along the bistatic
     ellipse rather than across it — a real, geometry-dependent blind set,
     not a small correction.
  3. BASELINE GEOMETRY. Near the transmitter-receiver baseline the bistatic
     geometry degenerates: range resolution collapses and the direct signal
     is strongest. Detection there is scored zero.

Range resolution is c/(2B), so the illuminator choice matters as much as its
power: FM's 150 kHz gives 1 km resolution, DVB-T's 8 MHz gives 19 m, 5G's
100 MHz gives 1.5 m. Doppler resolution is 1/T for integration time T, and T
is bounded by how long the target stays in one range-Doppler cell.

Output: detections_pcl.jsonl, one record per frame. Each detection carries
the bearing in CAMERA PIXELS (so the tracker can fuse it exactly like the
acoustic bearing) plus the range PCL actually measures, which no passive
optical channel provides.

    python scripts/pcl_sim.py run --clip data/clips/range_fast_sky
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

C = 3.0e8
K_BOLTZ = 1.380649e-23
T0 = 290.0

# name: (MHz, bandwidth Hz, ERP W, note)
ILLUMINATORS = {
    "fm":     (98.0, 150e3, 50e3, "FM radio: huge range, 1 km range cells, "
                                  "needs a 12 m array"),
    "dab":    (220.0, 1.5e6, 10e3, "DAB: 100 m cells, 5.5 m array"),
    "dvbt":   (600.0, 8e6, 50e3, "DVB-T: the workhorse - 19 m cells, 2 m "
                                 "array, high power"),
    "lte800": (800.0, 20e6, 200.0, "LTE 800: low power per cell but the "
                                   "tower is usually close; 7.5 m cells"),
    "lte1800": (1800.0, 20e6, 100.0, "LTE 1800: 0.67 m array, 7.5 m cells"),
    "nr3500": (3500.0, 100e6, 50.0, "5G n78: 0.34 m array, 1.5 m cells, "
                                    "shortest reach"),
}


class PCL:
    def __init__(self, illum="dvbt", tx_range_km=20.0, tx_az_deg=140.0,
                 tx_el_deg=1.0, suppression_db=80.0, rx_gain_dbi=10.0,
                 noise_fig_db=3.0, integration_s=0.5, rcs=0.01,
                 min_doppler_hz=2.0, baseline_null_deg=12.0,
                 anchor_km=1.5):
        mhz, bw, erp, _ = ILLUMINATORS[illum]
        self.f = mhz * 1e6
        self.lam = C / self.f
        self.bw, self.erp = bw, erp
        self.suppression = 10 ** (suppression_db / 10.0)
        self.gr = 10 ** (rx_gain_dbi / 10.0)
        self.nf = 10 ** (noise_fig_db / 10.0)
        self.T = integration_s
        self.rcs = rcs
        self.min_doppler = min_doppler_hz
        self.baseline_null = math.radians(baseline_null_deg)
        r = tx_range_km * 1000.0
        el = math.radians(tx_el_deg)
        az = math.radians(tx_az_deg)
        self.tx = np.array([r * math.cos(el) * math.cos(az),
                            r * math.cos(el) * math.sin(az),
                            r * math.sin(el)])
        self.range_res = C / (2 * self.bw)
        self.dopp_res = 1.0 / self.T
        # Calibrate the absolute scale: force SNR = 10 dB (the detection
        # threshold) at the anchor range for THIS configuration, so the model
        # reproduces published micro-UAV performance instead of the radar
        # equation's fantasy. Everything else then scales physically.
        self.cal_db = 0.0
        a = np.array([anchor_km * 1000 * 0.7, anchor_km * 1000 * 0.7, 150.0])
        rt, rr, _, _ = self.bistatic(a, np.zeros(3))
        self.cal_db = 10.0 - self.echo_to_residual_db(rt, rr)

    # ---------------------------------------------------------------- physics
    def bistatic(self, p, rx):
        """Extra path the echo travels, versus the direct signal."""
        rt = float(np.linalg.norm(p - self.tx))
        rr = float(np.linalg.norm(p - rx))
        base = float(np.linalg.norm(self.tx - rx))
        return rt, rr, rt + rr - base, base

    def echo_to_residual_db(self, rt, rr):
        """Echo power against the RESIDUAL direct signal after cancellation.

        Thermal noise is included but is almost never the binding term; the
        leftover direct path after finite cancellation usually is.
        """
        pr = (self.erp * self.gr * self.lam ** 2 * self.rcs) / \
             ((4 * math.pi) ** 3 * rt ** 2 * rr ** 2)
        base = float(np.linalg.norm(self.tx))
        direct = (self.erp * self.gr * self.lam ** 2) / \
                 ((4 * math.pi) ** 2 * max(base, 1.0) ** 2)
        residual = direct / self.suppression
        thermal = K_BOLTZ * T0 * self.bw * self.nf
        gain = self.T * self.bw            # coherent processing gain
        return (10 * math.log10(pr * gain / (residual + thermal))
                + self.cal_db)

    def detect(self, p, v, rx, rng):
        """One dwell. Returns a dict or None, with the reason it failed."""
        rt, rr, bi_range, base = self.bistatic(p, rx)
        # bistatic Doppler: rate of change of the two-leg path
        ut = (p - self.tx) / max(rt, 1e-6)
        ur = (p - rx) / max(rr, 1e-6)
        fd = -float(np.dot(v, ut + ur)) / self.lam
        if abs(fd) < max(self.min_doppler, self.dopp_res):
            return None, "zero-Doppler ridge"
        # baseline geometry: near the Tx-Rx line the ellipse degenerates
        to_tx = self.tx / max(float(np.linalg.norm(self.tx)), 1e-6)
        to_p = (p - rx) / max(rr, 1e-6)
        if math.acos(float(np.clip(np.dot(to_tx, to_p), -1, 1))) < self.baseline_null:
            return None, "baseline null"
        snr = self.echo_to_residual_db(rt, rr)
        if snr < 10.0:
            return None, "below detection floor"
        # measurement noise: range from resolution, bearing from array size
        sig_r = self.range_res / max(1.0, math.sqrt(10 ** (snr / 10.0)) / 3.0)
        sig_az = math.radians(3.0) / max(1.0, (10 ** (snr / 20.0)) / 10.0)
        return {"bistatic_range_m": bi_range + rng.gauss(0, sig_r),
                "doppler_hz": fd,
                "range_m": rr + rng.gauss(0, sig_r),
                "snr_db": snr,
                "sigma_range_m": sig_r,
                "sigma_az_deg": math.degrees(max(sig_az, math.radians(1.5)))}, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["run", "budget"])
    ap.add_argument("--clip", default=None)
    ap.add_argument("--illuminator", default="dvbt", choices=sorted(ILLUMINATORS))
    ap.add_argument("--tx-range-km", type=float, default=20.0)
    ap.add_argument("--tx-az-deg", type=float, default=140.0,
                    help="transmitter bearing. 0 would put it straight down "
                         "the camera's boresight and inside the baseline null")
    ap.add_argument("--suppression-db", type=float, default=80.0,
                    help="direct-signal cancellation actually achieved. This "
                         "single number, not transmit power, sets range")
    ap.add_argument("--rcs", type=float, default=0.01,
                    help="micro-UAV radar cross-section, m^2")
    ap.add_argument("--integration-s", type=float, default=0.5)
    ap.add_argument("--anchor-km", type=float, default=1.5,
                    help="published detection range for a 0.01 m^2 micro-UAV "
                         "at 80 dB cancellation; the absolute scale is pinned "
                         "here because the residual-clutter term that really "
                         "sets it is not analytically tractable")
    ap.add_argument("--seed", type=int, default=11)
    args = ap.parse_args()

    pcl = PCL(args.illuminator, args.tx_range_km, args.tx_az_deg,
              suppression_db=args.suppression_db, rcs=args.rcs,
              integration_s=args.integration_s, anchor_km=args.anchor_km)
    mhz, bw, erp, note = ILLUMINATORS[args.illuminator]
    print(f"\nilluminator {args.illuminator}: {mhz:.0f} MHz, {bw/1e6:.1f} MHz "
          f"BW, {erp/1e3:.0f} kW ERP\n  {note}")
    print(f"  wavelength {pcl.lam:.2f} m -> half-wave element {pcl.lam/2:.2f} m,"
          f" 8-element array {4*pcl.lam:.2f} m")
    print(f"  range resolution {pcl.range_res:.0f} m, Doppler resolution "
          f"{pcl.dopp_res:.1f} Hz ({args.integration_s:.2f} s dwell)")

    if args.stage == "budget":
        print(f"\n{'target range':>13} {'echo/residual':>14} {'detect?':>8}")
        for r in (300, 500, 800, 1200, 2000, 3000, 5000):
            p = np.array([r * 0.7, r * 0.7, 150.0])
            rt, rr, _, _ = pcl.bistatic(p, np.zeros(3))
            snr = pcl.echo_to_residual_db(rt, rr)
            print(f"{r:>11} m {snr:>13.1f} dB {'yes' if snr >= 10 else 'no':>8}")
        print("\nSuppression sensitivity (target at 1200 m):")
        for sup in (60, 70, 80, 90, 100):
            q = PCL(args.illuminator, args.tx_range_km, args.tx_az_deg,
                    suppression_db=sup, rcs=args.rcs,
                    integration_s=args.integration_s,
                    anchor_km=args.anchor_km)
            q.cal_db = pcl.cal_db          # hold the calibration fixed
            p = np.array([1200 * 0.7, 1200 * 0.7, 150.0])
            rt, rr, _, _ = q.bistatic(p, np.zeros(3))
            print(f"  {sup:>3} dB cancellation -> "
                  f"{q.echo_to_residual_db(rt, rr):>6.1f} dB echo/residual")
        return

    clip = Path(args.clip)
    meta = json.loads((clip / "meta.json").read_text())
    fx = float(meta.get("fx", 1108.77))
    W = int(meta.get("width", 1280))
    recs = [json.loads(l) for l in open(clip / "labels.jsonl")]
    rx = np.array([0.1, 0.0, 2.5])
    rng = random.Random(args.seed)

    out, reasons, seen, tot = [], {}, 0, 0
    prev_p, prev_t = None, None
    for rec in recs:
        p = np.array(rec["pos"], dtype=float)
        t = rec["t"]
        v = ((p - prev_p) / max(t - prev_t, 1e-6)) if prev_p is not None \
            else np.zeros(3)
        prev_p, prev_t = p, t
        dets = []
        if rec.get("visible") and np.linalg.norm(v) > 0:
            tot += 1
            d, why = pcl.detect(p, v, rx, rng)
            if d is None:
                reasons[why] = reasons.get(why, 0) + 1
            else:
                seen += 1
                rel = p - rx
                az = math.atan2(rel[1], rel[0])
                col = W / 2 - fx * math.tan(az)
                half = max(4.0, fx * math.tan(math.radians(d["sigma_az_deg"])))
                dets.append({"xyxy": [col - half, 0.0, col + half,
                                      float(meta.get("height", 720))],
                             "conf": round(min(1.0, d["snr_db"] / 30.0), 4),
                             "cls": "pcl",
                             "range_m": round(d["range_m"], 1),
                             "bistatic_range_m": round(d["bistatic_range_m"], 1),
                             "doppler_hz": round(d["doppler_hz"], 1),
                             "snr_db": round(d["snr_db"], 1),
                             "bearing_sigma_deg": round(d["sigma_az_deg"], 2)})
        out.append({"frame": rec["frame"], "detections": dets})

    dst = clip / "detections_pcl.jsonl"
    with open(dst, "w") as fh:
        for r in out:
            fh.write(json.dumps(r) + "\n")
    print(f"\n{clip.name}: {seen}/{tot} moving-target dwells detected "
          f"= {seen/max(tot,1):.1%}")
    for why, n in sorted(reasons.items(), key=lambda kv: -kv[1]):
        print(f"  missed - {why:<22} {n:>6} ({n/max(tot,1):.1%})")
    print(f"wrote {dst}")


if __name__ == "__main__":
    main()
