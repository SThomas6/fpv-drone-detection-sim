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
    """Cross-ambiguity surface over (delay, Doppler)."""
    n = len(surv)
    t = np.arange(n) / fs
    out = np.zeros((len(dopplers), max_delay + 1))
    R = np.fft.fft(np.conj(ref[::-1]), 2 * n)
    for i, fd in enumerate(dopplers):
        s = surv * np.exp(-2j * math.pi * fd * t)
        cc = np.fft.ifft(np.fft.fft(s, 2 * n) * R)
        seg = np.abs(cc[n - 1:n + max_delay])
        out[i, :len(seg)] = seg
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="radar/fers_out")
    ap.add_argument("--fs", type=float, default=2.0e6)
    ap.add_argument("--max-range-m", type=float, default=3000.0)
    ap.add_argument("--max-doppler", type=float, default=400.0)
    ap.add_argument("--taps", type=int, default=64)
    args = ap.parse_args()

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
    sub = surface[keep]
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
