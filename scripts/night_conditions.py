#!/usr/bin/env python3
"""Does the system work at night? Build night imagery and measure per sensor.

No clip in this project was ever captured after dark - the darkest lighting
preset is `dusk` - so "does it work at night?" had no measured answer.

Night is not "the same picture, darker". A real low-light camera at high gain
gives: a collapsed signal (photon-starved), a raised black level (sensor and
sky glow), heavy read/shot noise that grows as the square root of the
attenuated signal, and near-total loss of colour. All four are applied here.
The `moonlit` and `starlight` presets differ by orders of magnitude of
available light.

Critically, THE THERMAL BAND IS NOT DARKENED, and that is physics, not a
convenience: an LWIR camera senses EMITTED heat, so a drone's motors are
just as bright at midnight as at noon. If anything the night background is
colder and the contrast improves - modelling thermal as unchanged is the
conservative choice.

    python scripts/night_conditions.py --clip data/clips/terrain_ir_canopy
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from camera.detector import DroneDetector  # noqa: E402
from camera.evaluate import is_hit  # noqa: E402

# gain: fraction of daylight signal that survives
# black: raised black level in DN (sky glow + sensor pedestal)
# read_e: read-noise sigma in ELECTRONS
# sat:   remaining colour saturation
PRESETS = {
    "day":        (1.00, 0.0, 3.0, 1.00, "control"),
    "twilight":   (0.22, 6.0, 4.0, 0.55, "civil twilight, camera still easy"),
    "moonlit":    (0.055, 10.0, 5.0, 0.15, "full moon, clear: high gain, "
                                           "visible noise, near-monochrome"),
    "starlight":  (0.018, 14.0, 6.0, 0.05, "no moon: photon-starved, heavy "
                                            "noise - the hard case"),
}


FULL_WELL_E = 20000.0    # electrons at daylight full scale


def make_night(img: np.ndarray, preset: str, rng) -> np.ndarray:
    """Photon-limited night imaging, in ELECTRONS.

    An earlier version added sqrt(DN) shot noise regardless of light level,
    which corrupted even the daylight control (recall 0.98 -> 0.28) - the
    model, not the camera, was the problem. Noise has to be generated where
    it physically arises: photons are counted, the count is Poisson, and a
    high-gain camera then amplifies signal AND noise together back into
    display range. That makes daylight essentially clean and starlight
    genuinely ugly, which is the whole point of the comparison.
    """
    gain, black, read_e, sat, _ = PRESETS[preset]
    x = img.astype(np.float32)
    if sat < 1.0:
        grey = x.mean(axis=2, keepdims=True)
        x = grey + (x - grey) * sat
    # photons actually collected at this light level
    e = FULL_WELL_E * gain * np.clip(x, 0, 255) / 255.0
    e = e + rng.normal(0.0, 1.0, e.shape) * np.sqrt(np.maximum(e, 1e-6))
    e = e + rng.normal(0.0, read_e, e.shape)
    # amplify back to display range - noise comes up with the signal
    out = 255.0 * e / max(FULL_WELL_E * gain, 1e-9) + black
    return np.clip(out, 0, 255).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", required=True)
    ap.add_argument("--conf", type=float, default=0.10)
    ap.add_argument("--limit", type=int, default=150)
    ap.add_argument("--presets", nargs="+", default=list(PRESETS))
    ap.add_argument("--save-example", default=None)
    ap.add_argument("--write-clip", default=None,
                    help="also write a full night CLIP here (visible "
                         "darkened, thermal copied unchanged)")
    ap.add_argument("--write-preset", default="moonlit")
    args = ap.parse_args()

    clip = Path(args.clip)
    recs = [json.loads(l) for l in open(clip / "labels.jsonl")]
    if args.write_clip:
        dst = Path(args.write_clip)
        (dst / "frames").mkdir(parents=True, exist_ok=True)
        shutil.copy(clip / "labels.jsonl", dst / "labels.jsonl")
        meta = json.loads((clip / "meta.json").read_text())
        meta["note"] = f"{clip.name} at NIGHT ({args.write_preset}); " \
                       "thermal unchanged (emitted band)"
        (dst / "meta.json").write_text(json.dumps(meta, indent=2))
        rng = np.random.default_rng(9)
        for i, rec in enumerate(recs):
            im = np.array(Image.open(clip / "frames" / rec["frame"]
                                     ).convert("RGB"))
            Image.fromarray(make_night(im, args.write_preset, rng)).save(
                dst / "frames" / rec["frame"])
            if (i + 1) % 200 == 0:
                print(f"  {i+1}/{len(recs)}", flush=True)
        ir_src = clip / "frames_ir"
        if ir_src.exists():
            (dst / "frames_ir").mkdir(exist_ok=True)
            for f in ir_src.glob("*.png"):
                shutil.copy(f, dst / "frames_ir" / f.name)
        print(f"wrote night clip {dst}")
        return

    vis = [r for r in recs if r.get("visible")][:args.limit]
    det = DroneDetector(conf=0.03)
    det.warmup()
    print(f"\n{clip.name}: {len(vis)} drone-visible frames, VISIBLE camera")
    print(f"{'light':>11} {'recall':>8} {'med conf':>9}  note")
    for name in args.presets:
        rng = np.random.default_rng(9)
        seen, confs = 0, []
        for rec in vis:
            im = np.array(Image.open(clip / "frames" / rec["frame"]
                                     ).convert("RGB"))
            im = make_night(im, name, rng)
            if args.save_example and rec is vis[len(vis) // 2]:
                Image.fromarray(im).save(f"{args.save_example}_{name}.png")
            hits = [d for d in det.detect(im)
                    if d.confidence >= args.conf and is_hit(d.xyxy, rec)]
            if hits:
                seen += 1
                confs.append(max(h.confidence for h in hits))
        print(f"{name:>11} {seen / max(len(vis), 1):>8.3f} "
              f"{(np.median(confs) if confs else 0):>9.3f}  {PRESETS[name][4]}",
              flush=True)


if __name__ == "__main__":
    main()
