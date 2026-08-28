#!/usr/bin/env python3
"""Build a two-class (drone, bird) YOLO dataset from recorded clips.

The simulator gives exact ground truth for both classes, so labels are free.
Frames are cut into 640x640 tiles AT NATIVE SCALE — no resizing — so object
pixel sizes during training match what full-frame inference at imgsz (736,1280)
sees. Each anchor object gets one jittered tile; every other object whose box
falls inside the tile is labelled too, and empty background tiles are added so
the model keeps seeing sky/terrain with nothing in it.

    python camera/build_finetune_dataset.py \\
        --train data/clips/birds_only data/clips/birds_drone ... \\
        --val   data/clips/backlit_birds \\
        --out   data/finetune

Split by CLIP, never by frame: frames inside one clip are temporally
correlated, and a within-clip split would leak nearly-identical images into
validation.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from PIL import Image

TILE = 640
MIN_BOX_PX = 2.5          # tinier than this is label noise, skip
# --tile 320 --upscale 2 builds tiles that mimic SAHI's inference scale:
# detect_sliced cuts 320 px slices and infers them at image_size 640, so the
# model sees everything at 2x. Measured (2026-08-28): at that scale the
# deployed model names canopy birds 'bird' only 79.5% of the time against
# 93.2% at native scale - the model never trained on upscaled imagery.
# Normalised YOLO labels are scale-invariant, so only the image is resized.
MAX_BOX_PX = 500          # a bird flying right past the lens; not useful
CLS = {"drone": 0, "bird": 1}


def objects_in_frame(rec) -> list[tuple[int, list[float]]]:
    out = []
    if rec.get("visible") and rec.get("bbox"):
        w = rec["bbox"][2] - rec["bbox"][0]
        if MIN_BOX_PX <= w <= MAX_BOX_PX:
            out.append((CLS["drone"], rec["bbox"]))
    for b in rec.get("birds") or []:
        w = b["bbox"][2] - b["bbox"][0]
        if MIN_BOX_PX <= w <= MAX_BOX_PX:
            out.append((CLS["bird"], b["bbox"]))
    return out


def tile_origin_for(bbox, img_w, img_h, rng, tile=TILE):
    """Top-left of a tile-sized crop containing bbox, with random jitter."""
    cx = (bbox[0] + bbox[2]) / 2
    cy = (bbox[1] + bbox[3]) / 2
    # object lands uniformly within the central 80% of the tile
    jx = rng.uniform(0.1, 0.9) * tile
    jy = rng.uniform(0.1, 0.9) * tile
    x0 = int(min(max(cx - jx, 0), img_w - tile))
    y0 = int(min(max(cy - jy, 0), img_h - tile))
    return x0, y0


def labels_for_tile(objs, x0, y0, tile=TILE):
    """YOLO label lines for every object visible inside the tile."""
    lines = []
    for cls, (bx1, by1, bx2, by2) in objs:
        ix1, iy1 = max(bx1, x0), max(by1, y0)
        ix2, iy2 = min(bx2, x0 + tile), min(by2, y0 + tile)
        if ix2 - ix1 < 2.0 or iy2 - iy1 < 1.0:
            continue
        cx = ((ix1 + ix2) / 2 - x0) / tile
        cy = ((iy1 + iy2) / 2 - y0) / tile
        w = (ix2 - ix1) / tile
        h = (iy2 - iy1) / tile
        lines.append(f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    return lines


def tile_is_empty(objs, x0, y0, tile=TILE) -> bool:
    for _, (bx1, by1, bx2, by2) in objs:
        if bx2 > x0 and bx1 < x0 + tile and by2 > y0 and by1 < y0 + tile:
            return False
    return True


def collect_anchors(clips):
    """(clip_path, rec, cls, bbox) for every usable object instance."""
    anchors = []
    for clip in clips:
        clip = Path(clip)
        for rec in (json.loads(l) for l in open(clip / "labels.jsonl")):
            for cls, bbox in objects_in_frame(rec):
                anchors.append((clip, rec, cls, bbox))
    return anchors


def build_split(clips, out_dir, split, caps, bg_per_frames, rng,
                tile=TILE, upscale=1):
    img_dir = out_dir / "images" / split
    lbl_dir = out_dir / "labels" / split
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    anchors = collect_anchors(clips)
    by_cls = {0: [a for a in anchors if a[2] == 0],
              1: [a for a in anchors if a[2] == 1]}
    chosen = []
    for cls, cap in caps.items():
        rng.shuffle(by_cls[cls])
        chosen += by_cls[cls][:cap]
    rng.shuffle(chosen)

    counts = {"drone_tiles": 0, "bird_tiles": 0, "bg_tiles": 0,
              "drone_boxes": 0, "bird_boxes": 0}
    frame_cache: dict[str, Image.Image] = {}

    def load_frame(clip, rec):
        key = f"{clip}/{rec['frame']}"
        if key not in frame_cache:
            frame_cache.clear()          # keep exactly one frame in memory
            frame_cache[key] = Image.open(clip / "frames" / rec["frame"]).convert("RGB")
        return frame_cache[key]

    n = 0
    for clip, rec, cls, bbox in chosen:
        img = load_frame(clip, rec)
        objs = objects_in_frame(rec)
        x0, y0 = tile_origin_for(bbox, img.width, img.height, rng, tile)
        lines = labels_for_tile(objs, x0, y0, tile)
        if not lines:
            continue
        name = f"{split}_{n:06d}"
        crop = img.crop((x0, y0, x0 + tile, y0 + tile))
        if upscale != 1:
            crop = crop.resize((tile * upscale, tile * upscale),
                               Image.BILINEAR)
        crop.save(img_dir / f"{name}.png")
        (lbl_dir / f"{name}.txt").write_text("\n".join(lines) + "\n")
        counts["drone_tiles" if cls == 0 else "bird_tiles"] += 1
        counts["drone_boxes"] += sum(1 for l in lines if l.startswith("0 "))
        counts["bird_boxes"] += sum(1 for l in lines if l.startswith("1 "))
        n += 1

        # occasional background tile from the same frame
        if n % bg_per_frames == 0:
            for _ in range(12):
                bx0 = rng.randint(0, img.width - tile)
                by0 = rng.randint(0, img.height - tile)
                if tile_is_empty(objs, bx0, by0, tile):
                    bname = f"{split}_bg_{n:06d}"
                    bcrop = img.crop((bx0, by0, bx0 + tile, by0 + tile))
                    if upscale != 1:
                        bcrop = bcrop.resize((tile * upscale, tile * upscale),
                                             Image.BILINEAR)
                    bcrop.save(img_dir / f"{bname}.png")
                    (lbl_dir / f"{bname}.txt").write_text("")
                    counts["bg_tiles"] += 1
                    break
    return counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", nargs="+", required=True)
    ap.add_argument("--val", nargs="+", required=True)
    ap.add_argument("--out", default="data/finetune")
    ap.add_argument("--train-per-class", type=int, default=1100)
    ap.add_argument("--val-per-class", type=int, default=220)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tile", type=int, default=TILE)
    ap.add_argument("--upscale", type=int, default=1,
                    help="resize each tile by this factor after cropping "
                         "(2 with --tile 320 mimics the SAHI pass's scale)")
    args = ap.parse_args()

    overlap = set(map(str, args.train)) & set(map(str, args.val))
    if overlap:
        raise SystemExit(f"clip(s) in both train and val: {overlap}")

    out = Path(args.out)
    rng = random.Random(args.seed)
    caps_t = {0: args.train_per_class, 1: args.train_per_class}
    caps_v = {0: args.val_per_class, 1: args.val_per_class}

    print("building train split...")
    tc = build_split([Path(c) for c in args.train], out, "train", caps_t, 5,
                     rng, args.tile, args.upscale)
    print(f"  {tc}")
    print("building val split...")
    vc = build_split([Path(c) for c in args.val], out, "val", caps_v, 5,
                     rng, args.tile, args.upscale)
    print(f"  {vc}")

    yaml = out / "dataset.yaml"
    # "path: ." keeps the dataset portable; finetune.py rewrites it to an
    # absolute path at run time so ultralytics never guesses a datasets_dir.
    yaml.write_text(
        "path: .\n"
        "train: images/train\n"
        "val: images/val\n"
        "names:\n  0: drone\n  1: bird\n")
    (out / "provenance.json").write_text(json.dumps({
        "train_clips": [str(c) for c in args.train],
        "val_clips": [str(c) for c in args.val],
        "tile": TILE, "seed": args.seed,
        "train_counts": tc, "val_counts": vc,
    }, indent=2))
    print(f"\nwrote {yaml}")


if __name__ == "__main__":
    main()
