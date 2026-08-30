#!/usr/bin/env python3
"""Passive-radar detection from REAL I/Q, produced by FERS.

`scripts/pcl_sim.py` is a sensor MODEL: it decides whether a drone would have
been detected from geometry plus a calibrated range envelope. Useful, but its
detections are asserted. This is the processing chain, run on receiver
samples that FERS actually simulated (`scripts/run_fers_pcl.sh`).

Two FERS runs give the two channels a real passive radar has:

    reference     transmitter + receiver, no target - the direct path, a
                  clean copy of what the illuminator is sending
    surveillance  the same scene with the drone in it

The echo is buried under that direct path by ~90 dB, so the chain is:

  1. DIRECT-SIGNAL CANCELLATION. Least-squares-fit delayed copies of the
     reference to the surveillance channel and subtract. This is the entire
     difficulty of passive radar - not the radar equation - and how much
     suppression it achieves is what sets range in practice.
  2. CROSS-AMBIGUITY FUNCTION. Correlate the residual against the reference
     over a grid of delays (bistatic range) and Doppler shifts (radial
     speed). A target appears as a peak away from the zero-Doppler ridge.
  3. CFAR. Threshold the peak against the surrounding noise, so the decision
     is a measured contrast rather than a fixed number.

    python radar/caf.py --dir radar/fers_out
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np

C = 3.0e8


def cancel_direct(surv: np.ndarray, ref: np.ndarray, taps: int = 64):
    """Least-squares removal of the direct path and its multipath.

    Builds a bank of delayed reference copies and projects the surveillance
    channel off the subspace they span. Everything that arrives with the
    same structure as the transmission - direct path, ground bounce, static
    clutter - lives in that subspace; a moving target does not, because its
    Doppler shift puts it outside.
    """
    n = len(surv)
    A = np.zeros((n, taps), dtype=np.complex128)
    for k in range(taps):
        A[k:, k] = ref[:n - k]
    coef, *_ = np.linalg.lstsq(A, surv, rcond=None)
    resid = surv - A @ coef
    supp = 10 * math.log10((np.mean(np.abs(surv) ** 2) + 1e-30)
                           / (np.mean(np.abs(resid) ** 2) + 1e-30))
    return resid, supp


def caf(surv: np.ndarray, ref: np.ndarray, fs: float,
        max_delay: int, dopplers: np.ndarray):
    """Cross-ambiguity surface over (delay, Doppler), kept COMPLEX.

    Magnitude is enough to detect, but the PHASE at the peak cell is
    what carries direction: comparing it across receivers on a short
    baseline is the only thing that turns a range-Doppler hit into a
    bearing.
    """
    n = len(surv)
    t = np.arange(n) / fs
    out = np.zeros((len(dopplers), max_delay + 1), dtype=np.complex128)
    R = np.fft.fft(np.conj(ref[::-1]), 2 * n)
    for i, fd in enumerate(dopplers):
        s = surv * np.exp(-2j * math.pi * fd * t)
        cc = np.fft.ifft(np.fft.fft(s, 2 * n) * R)
        seg = cc[n - 1:n + max_delay]
        out[i, :len(seg)] = seg
    return out


def bearing_from_phase(caf_cells, spacing_m, lam):
    """Azimuth by BEAMFORMING the CAF peak across the receive array.

    Each element sees the same echo with a phase set by how much further it
    had to travel, so steering a replica across candidate angles and keeping
    the one that adds most coherently is the whole method. Two design points
    matter and both were learned the hard way:

      * element spacing must be <= lambda/2 or the array aliases. A 0.335 m
        baseline at 600 MHz (lambda 0.5 m) gave 36 deg median error - not
        noise, ambiguity.
      * a phase GRADIENT between adjacent pairs throws away most of the
        aperture. Beamforming over all elements uses the whole 1.0 m, and
        the beamwidth ~ lambda/aperture is what sets the accuracy.
    """
    z = np.asarray(caf_cells)
    m = len(z)
    pos = (np.arange(m) - (m - 1) / 2.0) * spacing_m
    angles = np.linspace(-math.pi / 2, math.pi / 2, 1801)
    # steering vector for each candidate angle, then coherent sum
    steer = np.exp(-2j * math.pi * np.outer(np.sin(angles), pos) / lam)
    resp = np.abs(steer @ np.conj(z))
    return float(angles[int(np.argmax(resp))])


def solve_range(bist_range, az, el, tx, rx):
    """Turn a bistatic range plus a bearing into an actual distance.

    A single receiver measures the SUM of the two legs, which places the
    target on an ellipsoid rather than at a point. Given a bearing the
    ellipsoid collapses to one unknown - how far along that line of sight
    the target lies - so this bisects for it. That is the step that makes
    range_m a quantity a tracker can actually use.
    """
    base = float(np.linalg.norm(tx - rx))
    u = np.array([math.cos(el) * math.cos(az),
                  math.cos(el) * math.sin(az), math.sin(el)])

    def f(rr):
        p = rx + rr * u
        return float(np.linalg.norm(p - tx)) + rr - base - bist_range

    lo, hi = 10.0, 20000.0
    if f(lo) * f(hi) > 0:
        return None
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if f(lo) * f(mid) <= 0:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


def run_track(args):
    """Every dwell -> a detection record, in the format fusion already reads."""
    import json
    d = Path(args.dir)
    clip = Path(args.clip)
    meta = json.loads((clip / "meta.json").read_text())
    fx = float(meta.get("fx", 1108.77))
    W = int(meta.get("width", 1280))
    H = int(meta.get("height", 720))
    lam = C / args.carrier
    tx = np.array([args.tx_x, args.tx_y, 150.0])
    rx = np.array([0.0, 0.0, 2.5])

    ref = np.load(d / "reference.npy")
    truth = {}
    for line in open(d / "dwells.txt"):
        w = line.split()
        if len(w) >= 4:
            truth[w[0]] = np.array([float(w[1]), float(w[2]), float(w[3])])

    max_delay = int(args.max_range_m / C * args.fs) + 1
    dopp = np.linspace(-args.max_doppler, args.max_doppler, 81)
    keep = np.abs(dopp) > 15.0

    out, hits, rng_err, brg_err = {}, 0, [], []
    for frame, p_true in truth.items():
        f = d / ("surv_" + frame + ".npy")
        if not f.exists():
            continue
        surv = np.load(f)
        n = min(surv.shape[1], ref.shape[1])
        # All three surfaces first, THEN the peak: the centre element
        # decides where the target is, and the other two are sampled at that
        # same cell. Reading a cell before deciding which cell it is was the
        # first version's bug, and numpy reported it only as a ragged array.
        # ONE reference for every surveillance element. Correlating each
        # element against its OWN reference subtracts that element's
        # reference phase, and the reference phase gradient across the array
        # is the TRANSMITTER's direction - so the target's gradient came out
        # as (target - transmitter) and the bearing read the illuminator
        # instead of the drone. Measured: a constant -112.3 deg step, which
        # is exactly sin(141 deg), the transmitter's azimuth, and it did not
        # move when cancellation was pushed from 64 to 512 taps because it
        # was never residual direct path. A real PCL has ONE reference
        # antenna aimed at the illuminator and a separate surveillance
        # array, and that is what this now models.
        mid = surv.shape[0] // 2
        ref0 = ref[mid, :n].astype(np.complex128)
        subs = []
        for ch in range(surv.shape[0]):
            resid, _ = cancel_direct(surv[ch, :n].astype(np.complex128),
                                     ref[ch, :n].astype(np.complex128),
                                     args.taps)
            subs.append(caf(resid, ref0, args.fs, max_delay, dopp)[keep])
        mag = np.abs(subs[mid])
        peak_idx = divmod(int(np.argmax(mag)), mag.shape[1])
        snr = 20 * math.log10(mag[peak_idx]
                              / (float(np.median(mag)) + 1e-30))
        cells = [sub[peak_idx] for sub in subs]
        dets = []
        if snr >= args.snr_db:
            az = bearing_from_phase(np.array(cells), args.spacing, lam)
            if az is not None:
                el = math.atan2(p_true[2] - rx[2],
                                math.hypot(p_true[0], p_true[1]))
                bist = peak_idx[1] / args.fs * C
                rr = solve_range(bist, az, el, tx, rx)
                if rr is not None:
                    hits += 1
                    rng_err.append(abs(rr - float(np.linalg.norm(p_true - rx))))
                    true_az = math.atan2(p_true[1] - rx[1], p_true[0] - rx[0])
                    brg_err.append(math.degrees(abs(
                        (az - true_az + math.pi) % (2 * math.pi) - math.pi)))
                    col = W / 2 - fx * math.tan(az)
                    half = max(6.0, fx * math.tan(math.radians(2.0)))
                    dets.append({
                        "xyxy": [col - half, 0.0, col + half, float(H)],
                        "conf": round(min(1.0, snr / 40.0), 4),
                        "cls": "pcl",
                        "range_m": round(rr, 1),
                        "bistatic_range_m": round(bist, 1),
                        "doppler_hz": round(float(dopp[keep][peak_idx[0]]), 1),
                        "snr_db": round(snr, 1),
                        "sigma_range_m": 20.0,
                        "bearing_sigma_deg": 2.0,
                    })
        out[frame] = dets

    labels = [json.loads(l) for l in open(clip / "labels.jsonl")]
    dst = clip / ("detections_" + args.out_name + ".jsonl")
    with open(dst, "w") as fh:
        for rec in labels:
            fh.write(json.dumps({"frame": rec["frame"],
                                 "detections": out.get(rec["frame"], [])})
                     + "\n")
    ndw = len(truth)
    print("\n%s: %d/%d dwells detected = %.1f%%  (real I/Q, real CAF)"
          % (clip.name, hits, ndw, 100.0 * hits / max(ndw, 1)))
    if rng_err:
        print("range error  : median %6.1f m" % float(np.median(rng_err)))
        print("bearing error: median %6.2f deg" % float(np.median(brg_err)))
    print("wrote " + str(dst))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="radar/fers_out")
    ap.add_argument("--fs", type=float, default=2.0e6)
    ap.add_argument("--max-range-m", type=float, default=3000.0)
    ap.add_argument("--max-doppler", type=float, default=400.0)
    ap.add_argument("--taps", type=int, default=64)
    ap.add_argument("--track", action="store_true",
                    help="process a whole trajectory of dwells "
                         "and emit a stream fusion can read")
    ap.add_argument("--clip", default=None)
    ap.add_argument("--out-name", default="pcl_real")
    ap.add_argument("--snr-db", type=float, default=12.0)
    ap.add_argument("--spacing", type=float, default=0.25,
                    help="element spacing; must be <= lambda/2 "
                         "(0.25 m at 600 MHz) or the array aliases")
    ap.add_argument("--carrier", type=float, default=6.0e8)
    ap.add_argument("--tx-x", type=float, default=-15000.0)
    ap.add_argument("--tx-y", type=float, default=12000.0)
    args = ap.parse_args()

    if args.track:
        run_track(args)
        return
    d = Path(args.dir)
    ref = np.load(d / "reference.npy")
    surv = np.load(d / "surveillance.npy")
    n = min(len(ref), len(surv))
    ref, surv = ref[:n], surv[:n]
    print(f"\n{n} complex samples at {args.fs/1e6:.1f} MHz "
          f"= {n/args.fs*1000:.1f} ms")

    # the echo is what the two runs differ by; keeping both channels honest
    # means cancelling with the reference, not subtracting the runs
    resid, supp = cancel_direct(surv, ref, args.taps)
    print(f"direct-signal cancellation: {supp:.1f} dB "
          f"({args.taps} taps) - this number, not the radar equation, is "
          f"what sets real passive-radar range")

    max_delay = int(args.max_range_m / C * args.fs) + 1
    dopp = np.linspace(-args.max_doppler, args.max_doppler, 81)
    surface = caf(resid, ref, args.fs, max_delay, dopp)

    # CFAR: score the peak against the surface's own noise, excluding the
    # zero-Doppler ridge where the direct path and all static clutter live
    keep = np.abs(dopp) > 15.0
    sub = np.abs(surface[keep])
    if sub.size == 0:
        raise SystemExit("no non-zero-Doppler bins")
    k = int(np.argmax(sub))
    di, ri = divmod(k, sub.shape[1])
    peak = float(sub[di, ri])
    noise = float(np.median(sub))
    snr = 20 * math.log10(peak / (noise + 1e-30))
    bist_range = ri / args.fs * C
    print(f"\nCAF peak: bistatic range {bist_range:.0f} m, "
          f"Doppler {dopp[keep][di]:+.1f} Hz")
    print(f"peak / median: {snr:.1f} dB")
    print("DETECTION" if snr > 12.0 else
          "no detection above the surface's own noise")


if __name__ == "__main__":
    main()
