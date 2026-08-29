#!/usr/bin/env python3
"""Thermal hot-spot detection for the station's IR camera (Phase: EO/IR fusion).

Deliberately CLASSICAL: Gazebo's thermal camera emits a noiseless per-visual
temperature map, which any deep model would trivially memorise — a threshold
detector is the honest baseline there, and research on fielded counter-UAS
systems says the thermal channel's value is detection persistence for fusion,
not appearance classification (birds are warm too).

Pipeline per frame:
  L16 PNG (Kelvin = value * 0.01)
    -> degradation shim: optics blur (MTF) + NETD noise, because raw sim
       thermal is unrealistically clean — numbers without the shim are fake
    -> threshold at ambient + delta_t
    -> connected components -> centroid, area, peak contrast
    -> map IR pixels into RGB-frame coordinates (same-pose cameras: exact
       linear mapping by focal ratio about the principal point)

    python camera/ir_detector.py run --clip data/clips/terrain_ir_birds
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

KELVIN_PER_COUNT = 0.01
IR_W, IR_H = 640, 512
IR_FX = 320 / np.tan(0.4189 / 2)          # ≈ 1505 px
RGB_W, RGB_H, RGB_FX = 1280, 720, 1108.5
AMBIENT_K = 288.0

# Degradation shim defaults: Boson-class NETD is 20-30 mK; we degrade harder
# (50 mK) plus optics blur so sim numbers stay conservative.
NETD_K = 0.05
BLUR_SIGMA_PX = 0.8


def _gaussian_blur(img: np.ndarray, sigma: float) -> np.ndarray:
    """Separable Gaussian without scipy; edge-normalised (zero-padding would
    drag border temperatures toward 0 K)."""
    r = max(1, int(3 * sigma))
    x = np.arange(-r, r + 1, dtype=float)
    k = np.exp(-0.5 * (x / sigma) ** 2)
    k /= k.sum()

    def conv(m):
        return np.convolve(m, k, mode="same") / np.convolve(
            np.ones_like(m), k, mode="same")
    out = np.apply_along_axis(conv, 0, img)
    return np.apply_along_axis(conv, 1, out)


def degrade(kelvin: np.ndarray, rng: np.random.Generator,
            netd=NETD_K, blur=BLUR_SIGMA_PX) -> np.ndarray:
    out = _gaussian_blur(kelvin, blur)
    return out + rng.normal(0.0, netd, out.shape)


def _blobs(mask: np.ndarray):
    """Connected components (4-neighbour) without scipy; yields pixel lists."""
    seen = np.zeros_like(mask, dtype=bool)
    ys, xs = np.nonzero(mask)
    for y0, x0 in zip(ys, xs):
        if seen[y0, x0]:
            continue
        stack, blob = [(y0, x0)], []
        seen[y0, x0] = True
        while stack:
            y, x = stack.pop()
            blob.append((y, x))
            for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                ny, nx = y + dy, x + dx
                if (0 <= ny < mask.shape[0] and 0 <= nx < mask.shape[1]
                        and mask[ny, nx] and not seen[ny, nx]):
                    seen[ny, nx] = True
                    stack.append((ny, nx))
        yield blob


def ir_to_rgb(u_ir: float, v_ir: float) -> tuple[float, float]:
    """Same-pose pinhole cameras: exact linear mapping by focal ratio."""
    s = RGB_FX / IR_FX
    return (RGB_W / 2 + (u_ir - IR_W / 2) * s,
            RGB_H / 2 + (v_ir - IR_H / 2) * s)


def detect(kelvin: np.ndarray, ambient=AMBIENT_K, delta_t=2.5,
           peak_delta_t=3.0, max_area=400):
    """Hot spots above ambient+delta_t whose PEAK clears ambient+peak_delta_t.

    The peak gate matters: sun-shaded ambient surfaces render ~1.4 K warm and,
    once optics blur and NETD noise are applied, speckle across a lower
    threshold as thousands of one-pixel blobs (measured: 1,181/frame at
    delta_t=1.5 with no gate). A real target's core survives the gate; noise
    speckle does not. Returns dicts in RGB pixel space.
    """
    mask = kelvin > (ambient + delta_t)
    dets = []
    for blob in _blobs(mask):
        if len(blob) > max_area:      # sun-warmed ground patch, not a target
            continue
        arr = np.array(blob)
        temps = kelvin[arr[:, 0], arr[:, 1]]
        if float(temps.max()) < ambient + peak_delta_t:
            continue
        cy, cx = arr[:, 0].mean(), arr[:, 1].mean()
        u, v = ir_to_rgb(cx, cy)
        h_ir = np.ptp(arr[:, 0]) + 1
        w_ir = np.ptp(arr[:, 1]) + 1
        s = RGB_FX / IR_FX
        contrast = float(temps.max() - ambient)
        dets.append({
            "xyxy": [u - w_ir * s / 2, v - h_ir * s / 2,
                     u + w_ir * s / 2, v + h_ir * s / 2],
            "conf": round(min(1.0, contrast / 10.0), 4),
            "cls": "hotspot",
            "area_ir_px": len(blob),
            "peak_contrast_k": round(contrast, 2),
        })
    return dets


def _adopt_clip_intrinsics(clip: Path) -> None:
    """Take the RGB/IR focal lengths from the CLIP, not from constants.

    ir_to_rgb re-projects a thermal blob into the visible camera's pixel
    frame, and that mapping is pure focal-length ratio. The module defaults
    describe the 60 deg station lens; on a telephoto clip the true RGB fx is
    12212 px, so keeping 1108.5 would place every hot spot ~11x too close to
    the principal point - silently, with no error, and every fused number
    downstream would be wrong. Each clip records its own intrinsics in
    meta.json, so read them.
    """
    global RGB_W, RGB_H, RGB_FX, IR_FX
    meta_path = clip / "meta.json"
    if not meta_path.exists():
        return
    meta = json.loads(meta_path.read_text())
    RGB_W = int(meta.get("width", RGB_W))
    RGB_H = int(meta.get("height", RGB_H))
    RGB_FX = float(meta.get("fx", RGB_FX))
    th = meta.get("thermal") or {}
    IR_FX = float(th.get("fx", IR_FX))


def run(clip: Path, delta_t: float, seed=0):
    ir_dir = clip / "frames_ir"
    if not ir_dir.exists():
        raise SystemExit(f"{ir_dir} missing — capture with --thermal")
    _adopt_clip_intrinsics(clip)
    print(f"intrinsics from {clip.name}/meta.json: "
          f"RGB fx={RGB_FX:.1f}, IR fx={IR_FX:.1f}", flush=True)
    rng = np.random.default_rng(seed)
    out_path = clip / "detections_ir.jsonl"
    n = 0
    with open(out_path, "w") as fh:
        for f in sorted(ir_dir.glob("*.png")):
            kelvin = np.asarray(Image.open(f), dtype=np.float64) * KELVIN_PER_COUNT
            dets = detect(degrade(kelvin, rng), delta_t=delta_t)
            fh.write(json.dumps({"frame": f.name, "detections": dets}) + "\n")
            n += 1
            if n % 100 == 0:
                print(f"  {n} frames", flush=True)
    print(f"wrote {out_path} ({n} frames)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["run"])
    ap.add_argument("--clip", required=True)
    ap.add_argument("--delta-t", type=float, default=2.5,
                    help="detection threshold above ambient, Kelvin")
    args = ap.parse_args()
    run(Path(args.clip), args.delta_t)


if __name__ == "__main__":
    main()
