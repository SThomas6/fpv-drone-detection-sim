#!/usr/bin/env python3
"""Render detections against ground truth for eyeballing.

At these target sizes a full frame tells you nothing, so this also writes
zoomed crops around the true position and around each detection.

    python camera/visualize.py --clip data/clips/baseline --mode full --n 6
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

GT_COLOUR = (40, 220, 40)
DET_COLOUR = (255, 40, 40)


def crop(img: Image.Image, cx: float, cy: float, half: int, zoom: int):
    x1, y1 = int(cx - half), int(cy - half)
    box = img.crop((x1, y1, x1 + 2 * half, y1 + 2 * half))
    return box.resize((2 * half * zoom, 2 * half * zoom), Image.NEAREST), (x1, y1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", default="data/clips/baseline")
    ap.add_argument("--mode", default="full")
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--only", choices=["all", "fp", "miss", "hit"], default="all")
    ap.add_argument("--conf", type=float, default=0.15)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    clip = Path(args.clip)
    out = Path(args.out or clip / f"viz_{args.mode}")
    out.mkdir(parents=True, exist_ok=True)

    labels = {r["frame"]: r for r in
              (json.loads(l) for l in open(clip / "labels.jsonl"))}
    dets = {r["frame"]: r["detections"] for r in
            (json.loads(l) for l in open(clip / f"detections_{args.mode}.jsonl"))}

    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from camera.evaluate import is_hit

    chosen = []
    for frame, dlist in dets.items():
        gt = labels[frame]
        kept = [d for d in dlist if d["conf"] >= args.conf]
        hits = [d for d in kept if gt["visible"] and is_hit(d["xyxy"], gt)]
        fps = [d for d in kept if d not in hits]
        if args.only == "fp" and not fps:
            continue
        if args.only == "miss" and (not gt["visible"] or hits):
            continue
        if args.only == "hit" and not hits:
            continue
        chosen.append((frame, gt, kept, hits, fps))
        if len(chosen) >= args.n:
            break

    print(f"rendering {len(chosen)} frames -> {out}/")
    for frame, gt, kept, hits, fps in chosen:
        img = Image.open(clip / "frames" / frame).convert("RGB")
        d = ImageDraw.Draw(img)
        if gt["visible"]:
            x1, y1, x2, y2 = gt["bbox"]
            d.rectangle([x1 - 3, y1 - 3, x2 + 3, y2 + 3], outline=GT_COLOUR, width=2)
        for det in kept:
            d.rectangle(det["xyxy"], outline=DET_COLOUR, width=2)
        img.save(out / f"{Path(frame).stem}_full.png")

        panels = []
        if gt["visible"]:
            cx, cy = (gt["bbox"][0] + gt["bbox"][2]) / 2, (gt["bbox"][1] + gt["bbox"][3]) / 2
            c, _ = crop(img, cx, cy, 40, 6)
            panels.append(("truth", c))
        for i, det in enumerate(kept[:3]):
            cx = (det["xyxy"][0] + det["xyxy"][2]) / 2
            cy = (det["xyxy"][1] + det["xyxy"][3]) / 2
            c, _ = crop(img, cx, cy, 40, 6)
            panels.append((f"det{i}_{det['conf']:.2f}", c))
        if panels:
            w = sum(p[1].width for p in panels) + 10 * (len(panels) - 1)
            h = max(p[1].height for p in panels)
            sheet = Image.new("RGB", (w, h), (20, 20, 20))
            x = 0
            for name, p in panels:
                sheet.paste(p, (x, 0))
                ImageDraw.Draw(sheet).text((x + 4, 4), name, fill=(255, 255, 0))
                x += p.width + 10
            sheet.save(out / f"{Path(frame).stem}_crops.png")

        gtd = (f"gt px_w {gt['px_width']:.1f} @ {gt['range_m']}m"
               if gt["visible"] else "no drone in frame")
        print(f"  {frame}: {gtd}; {len(hits)} hit, {len(fps)} other "
              f"(confs {[round(x['conf'], 2) for x in kept]})")


if __name__ == "__main__":
    main()
