#!/usr/bin/env python3
"""One image: what the system actually sees at each range, and who saw it.

The range table says the wide camera dies around 150 m and the telephoto
carries 300-550 m. This shows the pixels behind those numbers so the claim
can be checked by eye rather than taken on trust.

Every tile is the SAME crop size in source pixels, magnified identically, so
the target visibly shrinks across the row - that shrinking is the entire
physical story. Boxes: green = ground truth, and one coloured box per sensor
that fired. A tile with only a green box is an honest miss.

    python scripts/render_contact_sheet.py --out sheet.png
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from camera.evaluate import is_hit  # noqa: E402

C_GT = (80, 255, 80)
SENSORS = (("full", (60, 60, 255), 0.10, "RGB"),
           ("sahi", (60, 160, 255), 0.10, "tiled"),
           ("ir", (0, 200, 255), 0.20, "thermal"),
           ("motion", (255, 255, 0), 0.10, "motion"),
           ("point", (255, 0, 255), 0.0, "point"))

CROP = 44          # source pixels per tile, identical everywhere
TILE = 232
CAP = 74


def load(clip: Path, name: str):
    p = clip / f"detections_{name}.jsonl"
    if not p.exists():
        return {}
    return {r["frame"]: r["detections"]
            for r in (json.loads(l) for l in open(p))}


def put(img, text, xy, scale=0.44, colour=(235, 235, 235), thick=1):
    cv2.putText(img, text, xy, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0),
                thick + 2, cv2.LINE_AA)
    cv2.putText(img, text, xy, cv2.FONT_HERSHEY_SIMPLEX, scale, colour,
                thick, cv2.LINE_AA)


def build_row(clip: Path, targets, label):
    recs = [json.loads(l) for l in open(clip / "labels.jsonl")
            if json.loads(l).get("visible")]
    streams = {n: load(clip, n) for n, _, _, _ in SENSORS}
    tiles = []
    for want in targets:
        rec = min(recs, key=lambda r: abs((r.get("range_m") or 1e9) - want))
        frame = cv2.imread(str(clip / "frames" / rec["frame"]))
        if frame is None:
            continue
        fh, fw = frame.shape[:2]
        gx = (rec["bbox"][0] + rec["bbox"][2]) / 2
        gy = (rec["bbox"][1] + rec["bbox"][3]) / 2
        x0 = int(np.clip(gx - CROP / 2, 0, fw - CROP))
        y0 = int(np.clip(gy - CROP / 2, 0, fh - CROP))
        tile = cv2.resize(frame[y0:y0 + CROP, x0:x0 + CROP], (TILE, TILE),
                          interpolation=cv2.INTER_NEAREST)
        z = TILE / CROP

        def box(b, colour, pad):
            a1, b1 = int((b[0] - x0) * z) - pad, int((b[1] - y0) * z) - pad
            a2, b2 = int((b[2] - x0) * z) + pad, int((b[3] - y0) * z) + pad
            cv2.rectangle(tile, (a1, b1), (a2, b2), colour, 2)

        hits = []
        for name, colour, thr, tag in SENSORS:
            for d in streams[name].get(rec["frame"], []):
                if d["conf"] >= thr and is_hit(d["xyxy"], rec):
                    box(d["xyxy"], colour, 9)
                    if tag not in [h[0] for h in hits]:
                        hits.append((tag, colour))
                    break
        box(rec["bbox"], C_GT, 4)

        cap = np.full((CAP, TILE, 3), 24, np.uint8)
        rng = rec.get("range_m") or 0
        gw = rec["bbox"][2] - rec["bbox"][0]
        put(cap, f"{rng:.0f} m", (6, 22), 0.62, (120, 220, 255), 2)
        put(cap, f"{gw:.1f} px", (110, 22), 0.46, (170, 170, 170), 1)
        if hits:
            x = 6
            for tag, colour in hits[:3]:
                put(cap, tag, (x, 46), 0.44, colour, 1)
                x += 12 + 9 * len(tag)
            if len(hits) > 3:
                put(cap, f"+{len(hits)-3}", (x, 46), 0.44, (200, 200, 200), 1)
            put(cap, "DETECTED", (6, 66), 0.44, (120, 255, 120), 1)
        else:
            put(cap, "missed", (6, 46), 0.46, (90, 90, 235), 1)
        tiles.append(np.vstack([tile, cap]))
    if not tiles:
        return None
    strip = np.hstack([np.pad(t, ((0, 0), (3, 3), (0, 0)),
                              constant_values=24) for t in tiles])
    head = np.full((30, strip.shape[1], 3), 24, np.uint8)
    put(head, label, (8, 21), 0.56, (240, 240, 240), 1)
    return np.vstack([head, strip])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", default="data/clips/range_1km_hr")
    ap.add_argument("--tele-clip", default="data/clips/range_tele")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rows = []
    r = build_row(Path(args.clip),
                  [30, 60, 100, 150, 200, 300, 450, 700],
                  "WIDE 60 deg CAMERA + thermal + motion  (all sensors on the "
                  "same frame; identical magnification across the row)")
    if r is not None:
        rows.append(r)
    r = build_row(Path(args.tele_clip),
                  [320, 400, 500, 600, 750, 900, 1100, 1350],
                  "6 deg TELEPHOTO + point-target detector  (same crop size, "
                  "same magnification - note the target is still visible)")
    if r is not None:
        rows.append(r)
    sheet = np.vstack([np.pad(x, ((6, 6), (6, 6), (0, 0)), constant_values=24)
                       for x in rows])
    legend = np.full((34, sheet.shape[1], 3), 24, np.uint8)
    put(legend, "green = ground truth    red = RGB    blue = tiled    "
                "orange = thermal    cyan = motion    magenta = point-target",
        (10, 22), 0.46, (185, 185, 185), 1)
    cv2.imwrite(args.out, np.vstack([sheet, legend]))
    print(f"wrote {args.out} {sheet.shape[1]}x{sheet.shape[0] + 34}")


if __name__ == "__main__":
    main()
