#!/usr/bin/env python3
"""Run drone detection against the live simulator camera.

    conda activate dronesim
    python camera/detect_live.py            # detection only
    python camera/detect_live.py --track    # detection + tracking (ID, velocity)

Subscribes straight to the Gazebo camera topic (lower latency than going via
ROS; the ros_gz bridge remains available for other consumers). Prints the
Phase 2 report format and optionally appends JSON lines for later analysis.
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from camera.detector import DroneDetector  # noqa: E402

from gz.msgs10.image_pb2 import Image  # noqa: E402
from gz.transport13 import Node  # noqa: E402

CAM_TOPIC = "/detection_station/camera/image"

_latest: dict = {}
_lock = threading.Lock()


def img_cb(msg: Image):
    with _lock:
        _latest["img"] = np.frombuffer(msg.data, dtype=np.uint8).reshape(
            msg.height, msg.width, 3).copy()
        _latest["seq"] = _latest.get("seq", 0) + 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--track", action="store_true",
                    help="track targets across frames (adds ID and velocity)")
    ap.add_argument("--sahi", action="store_true",
                    help="sliced inference: slower, better on distant targets")
    ap.add_argument("--conf", type=float, default=None)
    ap.add_argument("--tracker", default="centroid", choices=["centroid", "bytetrack"],
                    help="centroid: distance-gated Kalman, the default because "
                         "IoU trackers lose ID on targets this small")
    ap.add_argument("--device", default=None, help="cuda|mps|cpu (default: auto)")
    ap.add_argument("--seconds", type=float, default=0.0, help="0 = run until stopped")
    ap.add_argument("--jsonl", default=None, help="also append results here")
    ap.add_argument("--classify", action="store_true",
                    help="use track motion to reject birds (implies --track)")
    ap.add_argument("--classify-threshold", type=float, default=0.5)
    args = ap.parse_args()

    kw = {"device": args.device, "use_sahi": args.sahi}
    if args.conf is not None:
        kw["conf"] = args.conf
    det = DroneDetector(**kw)
    print(f"model: {det.names} | device: {det.device} | "
          f"mode: {'SAHI sliced' if args.sahi else 'full frame'}"
          f"{' + tracking' if args.track else ''}", flush=True)
    det.warmup()

    node = Node()
    if not node.subscribe(Image, CAM_TOPIC, img_cb):
        raise SystemExit(f"could not subscribe to {CAM_TOPIC}")

    deadline = time.time() + 20
    while time.time() < deadline:
        with _lock:
            if "img" in _latest:
                break
        time.sleep(0.2)
    else:
        raise SystemExit("no camera frames — is the simulator running?")

    # One-off safety check: MPS post-processing has a history of corrupting box
    # coordinates, which would be silent without this.
    with _lock:
        probe = _latest["img"]
    if det.detect(probe):
        problems = det.check_device_parity(probe)
        if problems:
            print("WARNING: device parity check failed: " + "; ".join(problems), flush=True)
        else:
            print(f"{det.device} verified against cpu on a live frame", flush=True)

    if args.classify:
        args.track = True
    clf = None
    if args.classify:
        from camera.classify import MotionClassifier
        clf = MotionClassifier.load()
        if clf.w is None:
            raise SystemExit("no trained motion classifier — run: "
                             "python camera/classify.py train --drone <clips> --bird <clips>")
        print("motion classifier active: birds rejected on flight behaviour", flush=True)

    tracker = None
    if args.track and args.tracker == "centroid":
        from camera.tracking import CentroidTracker
        tracker = CentroidTracker()

    sink = open(args.jsonl, "a") if args.jsonl else None
    t_end = time.time() + args.seconds if args.seconds else None
    last_seq, frames, hits, birds_seen = -1, 0, 0, 0
    print("\nwatching for drones (Ctrl-C to stop)\n", flush=True)

    try:
        while t_end is None or time.time() < t_end:
            with _lock:
                frame, seq = _latest.get("img"), _latest.get("seq", 0)
            if frame is None or seq == last_seq:
                time.sleep(0.01)
                continue
            last_seq = seq
            frames += 1
            now = time.time()

            if args.track and tracker is not None:
                dets_all = det.infer(frame)
                # only drone-class detections may raise alarms; a two-class
                # model's bird detections are already-rejected distractors
                drone_dets = [d for d in dets_all if d.cls_name == "drone"]
                birds_seen += sum(1 for d in dets_all if d.cls_name == "bird")
                results = tracker.update(drone_dets, timestamp=now)
                verdicts = {}
                if clf is not None:
                    for t in results:
                        verdicts[t.track_id] = clf.classify_track(
                            t.history, args.classify_threshold)
                    # Only declare targets whose motion says drone. A track too
                    # short to judge is pending, not an alarm.
                    results = [t for t in results if verdicts[t.track_id][0] == "drone"]
                reports = []
                for t in results:
                    r = t.report()
                    if t.track_id in verdicts:
                        r += f"\nClassified: drone (p={verdicts[t.track_id][1]:.2f} vs bird)"
                    reports.append(r)
                payload = [{"id": t.track_id, "bbox": t.box,
                            "conf": t.confidence, "centre": t.position,
                            "velocity": t.velocity, "coasting": t.coasting,
                            "p_drone": verdicts.get(t.track_id, (None, None))[1]}
                           for t in results]
            elif args.track:
                results = det.track(frame, timestamp=now)
                reports = [t.report() for t in results]
                payload = [{"id": t.track_id, "bbox": t.detection.xyxy,
                            "conf": t.detection.confidence,
                            "centre": t.detection.centre,
                            "velocity": t.velocity} for t in results]
            else:
                dets_all = det.infer(frame)
                birds_seen += sum(1 for d in dets_all if d.cls_name == "bird")
                results = [d for d in dets_all if d.cls_name == "drone"]
                reports = [d.report() for d in results]
                payload = [{"bbox": d.xyxy, "conf": d.confidence,
                            "centre": d.centre} for d in results]

            if results:
                hits += 1
                for r in reports:
                    print(r)
                print(f"[{det.last_inference_ms:.0f} ms]\n", flush=True)
            elif frames % 15 == 0:
                print(f"no drone detected  [{det.last_inference_ms:.0f} ms]", flush=True)

            if sink:
                sink.write(json.dumps({"t": round(now, 3), "targets": payload}) + "\n")
                sink.flush()
    except KeyboardInterrupt:
        pass
    finally:
        if sink:
            sink.close()
        if frames:
            print(f"\n{frames} frames processed, drone present in {hits} "
                  f"({hits / frames * 100:.0f}%)"
                  + (f"; {birds_seen} bird sightings identified and ignored"
                     if birds_seen else ""), flush=True)


if __name__ == "__main__":
    main()
