#!/usr/bin/env python3
"""Track-before-detect: pull SUB-PIXEL targets out of noise by integrating.

The measured wall: past ~550 m the drone is under one pixel in every passive
sensor (wide 0.71 px, thermal 0.96 px at 550 m), and a per-frame detector
cannot threshold what is buried in noise. Every frame is judged alone and
each judgement is a coin-flip, so the information is thrown away 15 times a
second.

Track-before-detect inverts the order: do NOT threshold per frame. Keep the
continuous residual, and integrate it along candidate constant-velocity
TRAJECTORIES over many frames. A real target contributes its (tiny) energy
to the same trajectory every frame, so energy grows ~N while zero-mean noise
grows ~sqrt(N): integrating 20 frames buys ~sqrt(20) ~ 4.5x in SNR, which is
worth roughly a factor of 2 in range for an inverse-square signal. This is
the standard technique for dim point targets in IRST and space surveillance
(velocity-filter banks / 3D matched filters).

Implementation is the velocity-filter bank, deliberately the simple form:
  1. residual  = frame - temporal median  (no threshold - keep the values)
  2. for each candidate velocity (vx, vy) in a bank, shift each residual
     back by -v*dt and sum: a target moving at that velocity stacks
     coherently, everything else smears
  3. the peak over the bank is the detection statistic; its argmax gives
     position AND velocity at once

Cost is a handful of array shifts per velocity - no learning, no new sensor.

    python scripts/tbd_detect.py --clip data/clips/range_1km --min-range 550
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def residual_stack(frames, med_window: int = 9):
    """Background-suppressed residuals: what is NOT the static scene."""
    out = []
    buf = []
    for im in frames:
        g = im.astype(np.float32)
        if g.ndim == 3:
            g = g.mean(axis=2)
        buf.append(g)
        if len(buf) > med_window:
            buf.pop(0)
        bg = np.median(np.stack(buf), axis=0)
        out.append(g - bg)
    return out


def velocity_bank(max_px_per_frame: float, step: float):
    v = []
    n = int(max_px_per_frame / step)
    for i in range(-n, n + 1):
        for j in range(-n, n + 1):
            v.append((i * step, j * step))
    return v


def integrate(residuals, vx, vy):
    """Shift-and-add residuals along one constant-velocity hypothesis."""
    acc = np.zeros_like(residuals[-1])
    n = len(residuals)
    for k, r in enumerate(residuals):
        dt = (n - 1) - k               # frames back from the newest
        sx, sy = int(round(-vx * dt)), int(round(-vy * dt))
        acc += np.roll(np.roll(r, sy, axis=0), sx, axis=1)
    return acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", required=True)
    ap.add_argument("--band", default="ir", choices=["ir", "rgb"],
                    help="which imagery to integrate (thermal has the best "
                         "pixels-on-target at long range)")
    ap.add_argument("--frames", type=int, default=16,
                    help="integration length N (SNR gain ~ sqrt(N))")
    ap.add_argument("--max-v", type=float, default=6.0,
                    help="max target speed to search, px/frame")
    ap.add_argument("--v-step", type=float, default=1.0)
    ap.add_argument("--min-range", type=float, default=0.0,
                    help="only score frames beyond this range (the regime "
                         "single-frame detection has already lost)")
    ap.add_argument("--tol-px", type=float, default=12.0)
    ap.add_argument("--limit", type=int, default=260)
    args = ap.parse_args()

    clip = Path(args.clip)
    recs = [json.loads(l) for l in open(clip / "labels.jsonl")]
    meta = json.loads((clip / "meta.json").read_text())
    sub = "frames_ir" if args.band == "ir" else "frames"

    def to_band(gx, gy):
        """Ground truth (RGB pixels) -> the band's own pixel frame.

        The thermal camera is 640x512 with its own focal length, so a plain
        focal-ratio scale is wrong: the mapping is about the PRINCIPAL POINT,
        exactly the inverse of camera/ir_detector.ir_to_rgb. Getting this
        wrong put the ground truth ~570 px off and scored a working detector
        at 0.000.
        """
        if args.band != "ir":
            return gx, gy
        th = meta.get("thermal") or {}
        f_ir = th.get("fx", 1505.4)
        f_rgb = meta.get("fx", 1108.77)
        iw, ih = th.get("width", 640), th.get("height", 512)
        rw, rh = meta.get("width", 1280), meta.get("height", 720)
        s_ = f_ir / f_rgb
        return (iw / 2 + (gx - rw / 2) * s_,
                ih / 2 + (gy - rh / 2) * s_)

    idx = [i for i, r in enumerate(recs)
           if r.get("visible") and (r.get("range_m") or 0) >= args.min_range]
    idx = idx[:args.limit]
    if not idx:
        raise SystemExit("no frames beyond --min-range")
    print(f"{clip.name}: {len(idx)} frames beyond {args.min_range:.0f} m, "
          f"band={args.band}, N={args.frames}, "
          f"velocity bank={len(velocity_bank(args.max_v, args.v_step))}")

    bank = velocity_bank(args.max_v, args.v_step)
    hits = tot = 0
    margins = []
    for pos in range(args.frames, len(idx)):
        window = idx[pos - args.frames:pos]
        if window[-1] - window[0] != args.frames - 1:
            continue                      # non-contiguous, skip
        imgs = [np.array(Image.open(clip / sub / recs[i]["frame"]))
                for i in window]
        res = residual_stack(imgs)
        best_val, best_xy = -1e18, None
        for vx, vy in bank:
            acc = integrate(res, vx, vy)
            k = int(np.argmax(acc))
            y, x = divmod(k, acc.shape[1])
            if acc[y, x] > best_val:
                best_val, best_xy = float(acc[y, x]), (x, y)
        rec = recs[window[-1]]
        gx, gy = to_band((rec["bbox"][0] + rec["bbox"][2]) / 2,
                         (rec["bbox"][1] + rec["bbox"][3]) / 2)
        d = float(np.hypot(best_xy[0] - gx, best_xy[1] - gy))
        tot += 1
        hits += d <= args.tol_px
        margins.append(d)
        if tot % 25 == 0:
            print(f"  {tot} windows, running hit-rate "
                  f"{hits / tot:.3f}", flush=True)
    ms = sorted(margins)
    print(f"\nTBD peak lands on the drone in {hits}/{tot} = "
          f"{hits / max(tot,1):.3f} of integration windows")
    if ms:
        print(f"miss distance px: p50 {ms[len(ms)//2]:.1f}, "
              f"p90 {ms[int(len(ms)*0.9)]:.1f}")


if __name__ == "__main__":
    main()
