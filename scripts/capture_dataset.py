#!/usr/bin/env python3
"""Record a labelled dataset from the detection station's camera.

Saves CLEAN frames (nothing drawn on them) plus exact ground-truth labels
derived from the simulator's own pose data — so Phase 2 evaluation has free,
perfect labels to score detections against.

    conda activate dronesim
    python scripts/capture_dataset.py --seconds 120 --out data/clips/baseline

Output:
    <out>/frames/000000.png ...
    <out>/labels.jsonl      one record per frame
    <out>/meta.json         intrinsics, resolution, capture settings
"""

import argparse
import json
import os
import sys
import threading
import time
from pathlib import Path

import numpy as np
from PIL import Image as PILImage

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from simulator.station_camera import (  # noqa: E402
    FX, H, W, bird_half_extents, check_intrinsics, ground_truth_bbox, size_bucket,
)

from gz.msgs10.camera_info_pb2 import CameraInfo  # noqa: E402
from gz.msgs10.image_pb2 import Image  # noqa: E402
from gz.msgs10.pose_v_pb2 import Pose_V  # noqa: E402
from gz.transport13 import Node  # noqa: E402

CAM_TOPIC = "/detection_station/camera/image"
INFO_TOPIC = "/detection_station/camera/camera_info"

_latest = {}
_lock = threading.Lock()


def img_cb(msg: Image):
    with _lock:
        _latest["img"] = np.frombuffer(msg.data, dtype=np.uint8).reshape(
            msg.height, msg.width, 3).copy()
        _latest["img_t"] = time.time()
        _latest["img_seq"] = _latest.get("img_seq", 0) + 1


def pose_cb(msg: Pose_V):
    birds = {}
    for p in msg.pose:
        if p.name == "target_drone" or p.name.startswith("x500"):
            with _lock:
                _latest["pos"] = (p.position.x, p.position.y, p.position.z)
                _latest["quat"] = (p.orientation.w, p.orientation.x,
                                   p.orientation.y, p.orientation.z)
        elif p.name.startswith("bird_"):
            birds[p.name] = ((p.position.x, p.position.y, p.position.z),
                             (p.orientation.w, p.orientation.x,
                              p.orientation.y, p.orientation.z))
    if birds:
        with _lock:
            _latest["birds"] = birds


def info_cb(msg: CameraInfo):
    with _lock:
        _latest["info"] = (msg.intrinsics.k[0], msg.width, msg.height)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=120.0)
    ap.add_argument("--interval", type=float, default=0.5,
                    help="seconds between saved frames")
    ap.add_argument("--out", default="data/clips/baseline")
    ap.add_argument("--note", default="", help="free-text note stored in meta.json")
    ap.add_argument("--world", default="detection_world")
    ap.add_argument("--no-target", action="store_true",
                    help="record a target-free clip (background/FP mining); "
                         "all frames are labelled visible=False")
    ap.add_argument("--thermal", action="store_true",
                    help="also record the station thermal camera "
                         "(L16 Kelvin/0.01 frames into frames_ir/)")
    ap.add_argument("--verify-below-v", type=float, default=460.0,
                    help="only self-check labels above this pixel row; on "
                         "terrain worlds pass ~250 so the dark-object check "
                         "runs only against clear sky")
    ap.add_argument("--occlusion-manifest", default=None,
                    help="terrain manifest used for occlusion-aware labels; "
                         "needed for an isolated generated training world")
    ap.add_argument("--occlusion-heightmap", default=None,
                    help="heightmap .npy paired with --occlusion-manifest")
    args = ap.parse_args()

    # Terrain worlds carry an occlusion manifest; a drone behind a tree or a
    # ridge must not be labelled visible.
    occ = None
    if "terrain" in args.world:
        from simulator.occlusion import OcclusionChecker
        occ = OcclusionChecker.if_available(args.occlusion_manifest,
                                             args.occlusion_heightmap)
        print(f"occlusion checking: {'ON' if occ else 'manifest missing!'}",
              flush=True)

    node = Node()
    node.subscribe(Image, CAM_TOPIC, img_cb)
    node.subscribe(Pose_V, f"/world/{args.world}/dynamic_pose/info", pose_cb)
    node.subscribe(CameraInfo, INFO_TOPIC, info_cb)
    if args.thermal:
        def ir_cb(msg: Image):
            with _lock:
                _latest["ir"] = np.frombuffer(msg.data, dtype=np.uint16).reshape(
                    msg.height, msg.width).copy()
        node.subscribe(Image, "/detection_station/thermal/image", ir_cb)

    deadline = time.time() + 20
    while time.time() < deadline:
        with _lock:
            if "img" in _latest and ("pos" in _latest or args.no_target):
                break
        time.sleep(0.2)
    else:
        raise SystemExit("timed out waiting for camera/pose — is the sim running?")

    with _lock:
        info = _latest.get("info")
    if info:
        problems = check_intrinsics(*info)
        if problems:
            print("WARNING: station_camera.py disagrees with camera_info: "
                  + "; ".join(problems), flush=True)
        else:
            print(f"intrinsics verified against camera_info (fx={info[0]:.1f})", flush=True)
    else:
        print("WARNING: no camera_info received; using module constants", flush=True)

    out = Path(args.out)
    (out / "frames").mkdir(parents=True, exist_ok=True)
    labels_path = out / "labels.jsonl"

    n = int(args.seconds / args.interval)
    t0 = time.time()
    kept, buckets, visible_n = 0, {}, 0
    verify_err, verify_checked, stale_skipped = [], 0, 0
    last_seq = -1
    with open(labels_path, "w") as fh:
        for i in range(n):
            target_t = t0 + i * args.interval
            time.sleep(max(0.0, target_t - time.time()))
            with _lock:
                frame = _latest.get("img")
                pos, quat = _latest.get("pos"), _latest.get("quat", (1, 0, 0, 0))
                birds = dict(_latest.get("birds", {}))
                img_age = time.time() - _latest.get("img_t", 0)
                seq = _latest.get("img_seq", 0)
            if frame is None or (pos is None and not args.no_target):
                continue
            if args.no_target:
                pos, quat = None, None
            # A frozen camera stream silently pairs old pixels with fresh poses
            # and produces a dataset whose labels are all wrong.
            if seq == last_seq or img_age > 2.0:
                stale_skipped += 1
                continue
            last_seq = seq

            name = f"{i:06d}.png"
            PILImage.fromarray(frame).save(out / "frames" / name)
            if args.thermal:
                with _lock:
                    ir = _latest.get("ir")
                if ir is not None:
                    (out / "frames_ir").mkdir(exist_ok=True)
                    PILImage.fromarray(ir, mode="I;16").save(
                        out / "frames_ir" / name)
            gt = ground_truth_bbox(pos, quat) if pos is not None else None
            rec = {
                "frame": name,
                "t": round(time.time() - t0, 3),
                "pos": [round(v, 3) for v in pos] if pos is not None else None,
                "quat": [round(v, 4) for v in quat] if quat is not None else None,
            }
            hidden = occ is not None and gt and gt["visible"] and occ.occluded(pos)
            if hidden:
                rec.update({"bbox": None, "visible": False, "occluded": True,
                            "range_m": round(gt["range_m"], 1)})
            elif gt and gt["visible"]:
                rec.update({
                    "bbox": [round(v, 1) for v in gt["xyxy"]],
                    "centre": [round(v, 1) for v in gt["centre"]],
                    "range_m": round(gt["range_m"], 1),
                    "px_width": round(gt["px_width"], 2),
                    "bucket": size_bucket(gt["px_width"]),
                    "visible": True,
                })
                buckets[rec["bucket"]] = buckets.get(rec["bucket"], 0) + 1
                visible_n += 1
                # Self-check: against sky the drone is by far the darkest thing,
                # so its true pixel position can be found without a model. If
                # that disagrees with the projected label, the data is bad.
                cx_gt, cy_gt = gt["centre"]
                if cy_gt < args.verify_below_v and 20 < cx_gt < W - 20:
                    lum = frame[:500].astype(np.int32).sum(axis=2)
                    med = float(np.median(lum))
                    # Search a window around the expected position rather than
                    # the whole sky: with birds present the darkest pixel in
                    # frame is often a bird.
                    r = 50
                    y0, y1 = max(0, int(cy_gt) - r), min(lum.shape[0], int(cy_gt) + r)
                    x0, x1 = max(0, int(cx_gt) - r), min(lum.shape[1], int(cx_gt) + r)
                    win = lum[y0:y1, x0:x1]
                    if win.size:
                        yy, xx = np.unravel_index(int(np.argmin(win)), win.shape)
                        if win[yy, xx] < med - 60:      # a dark object is there
                            err = float(np.hypot(x0 + xx - cx_gt, y0 + yy - cy_gt))
                            verify_err.append(err)
                            rec["verify_px_err"] = round(err, 1)
                        else:
                            verify_err.append(999.0)    # nothing rendered there
                    verify_checked += 1
            else:
                rec.update({"bbox": None, "visible": False,
                            "range_m": round(gt["range_m"], 1) if gt else None})

            bird_recs = []
            for bname, (bpos, bquat) in birds.items():
                bgt = ground_truth_bbox(bpos, bquat, bird_half_extents(bname))
                if bgt and bgt["visible"] and (occ is None or not occ.occluded(bpos)):
                    bird_recs.append({
                        "name": bname,
                        "bbox": [round(v, 1) for v in bgt["xyxy"]],
                        "centre": [round(v, 1) for v in bgt["centre"]],
                        "range_m": round(bgt["range_m"], 1),
                        "px_width": round(bgt["px_width"], 2),
                    })
            rec["birds"] = bird_recs
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            kept += 1
            if kept % 20 == 0:
                print(f"  {kept}/{n} frames", flush=True)

    ok_frac = (sum(1 for e in verify_err if e <= 15.0) / len(verify_err)
               if verify_err else None)
    meta = {
        "world": args.world, "width": W, "height": H, "fx": round(FX, 2),
        "interval_s": args.interval, "frames": kept,
        "visible_frames": visible_n, "buckets": buckets, "note": args.note,
        "stale_frames_skipped": stale_skipped,
        "thermal": ({"width": 640, "height": 512, "hfov": 0.4189,
                     "fx": round(320 / np.tan(0.4189 / 2), 1),
                     "kelvin_per_count": 0.01} if args.thermal else None),
        "label_verification": {
            "checked": verify_checked, "with_dark_object": len(verify_err),
            "median_px_err": round(float(np.median(verify_err)), 2) if verify_err else None,
            "fraction_within_15px": round(ok_frac, 4) if ok_frac is not None else None,
        },
        "occlusion_assets": ({"manifest": args.occlusion_manifest,
                              "heightmap": args.occlusion_heightmap}
                             if args.occlusion_manifest else None),
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"\nsaved {kept} frames to {out}/ ({visible_n} with the drone in frame)")
    print("size buckets:", buckets or "(none visible)")
    if stale_skipped:
        print(f"skipped {stale_skipped} frames with a stale camera stream")
    if verify_err:
        print(f"label check: median {np.median(verify_err):.1f} px error, "
              f"{ok_frac * 100:.0f}% within 15 px "
              f"({len(verify_err)}/{verify_checked} frames had a findable dark target)")
        if ok_frac < 0.9:
            print("WARNING: labels disagree with the rendered image — DO NOT "
                  "evaluate against this clip until the cause is fixed")
    else:
        print("WARNING: could not verify any labels against rendered pixels")


if __name__ == "__main__":
    main()
