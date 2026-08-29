"""Where does fused coverage get lost: detection, tracking, or the gates?"""
import json, sys, math
sys.path.insert(0, r"C:/Users/sctho/Projects/drone sim/fpv-drone-detection-sim")
from camera.classify import MotionClassifier, drone_vote_fraction, features_from_history
from camera.detector import Detection, merge_close
from camera.evaluate import hits_object, is_hit
from camera.fuse_eval import fuse_measurements, load_clip
from camera.tracking import CentroidTracker
from pathlib import Path

clip = Path(sys.argv[1]); rgb_mode = sys.argv[2] if len(sys.argv) > 2 else "full"
clf = MotionClassifier.load()
rows = load_clip(clip, 0.05, 0.2, rgb_mode)
labels = {r["frame"]: r for r in (json.loads(l) for l in open(clip / "labels.jsonl"))}

# raw per-stream sight of the drone
seen = {"rgb": 0, "ir": 0, "mv": 0, "any": 0, "vis": 0}
for f, gt, rgb_dets, aux in rows:
    if not gt["visible"]:
        continue
    seen["vis"] += 1
    r = any(is_hit(d.xyxy, gt) for d in rgb_dets)
    i = any(is_hit(d.xyxy, gt) for d in aux if d.cls_name == "hotspot")
    m = any(is_hit(d.xyxy, gt) for d in aux if d.cls_name == "mover")
    seen["rgb"] += r; seen["ir"] += i; seen["mv"] += m; seen["any"] += (r or i or m)

tracker = CentroidTracker(class_consistent=True, suppress_spawn_near_coasting=True)
stage = {k: 0 for k in ("track", "travel", "motion", "vote")}
for f, gt, rgb_dets, aux in rows:
    dets, sources = fuse_measurements(rgb_dets, aux)
    tracks = tracker.update(dets, timestamp=gt["t"])
    if not gt["visible"]:
        continue
    on = [tr for tr in tracks if is_hit(tr.box, gt)]
    if not on:
        continue
    stage["track"] += 1
    def travel(tr):
        xs = [h[1] for h in tr.history]; ys = [h[2] for h in tr.history]
        return math.hypot(max(xs) - min(xs), max(ys) - min(ys)) if len(xs) >= 2 else 0.0
    t2 = [tr for tr in on if travel(tr) >= 8.0]
    if t2: stage["travel"] += 1
    t3 = [tr for tr in t2 if (lambda ft: ft is not None and clf.probability(ft) >= 0.5)
          (clf.features(tr.history))]
    if t3: stage["motion"] += 1
    t4 = [tr for tr in t3 if drone_vote_fraction(
        [h for h in tr.history if len(h) < 6 or h[5] not in ("hotspot", "mover")]) >= 0.5]
    if t4: stage["vote"] += 1

v = seen["vis"]
print(f"\n--- {clip.name} [rgb={rgb_mode}] {v} drone-visible frames ---")
print(f"  RGB stream sees drone      {seen['rgb']:>4}/{v} = {seen['rgb']/v:.3f}")
print(f"  IR stream sees drone       {seen['ir']:>4}/{v} = {seen['ir']/v:.3f}")
print(f"  motion stream sees drone   {seen['mv']:>4}/{v} = {seen['mv']/v:.3f}")
print(f"  ANY stream sees drone      {seen['any']:>4}/{v} = {seen['any']/v:.3f}   <- detection ceiling")
print(f"  confirmed track on drone   {stage['track']:>4}/{v} = {stage['track']/v:.3f}   <- after tracking")
print(f"   ...survives travel gate   {stage['travel']:>4}/{v} = {stage['travel']/v:.3f}")
print(f"   ...survives motion clf    {stage['motion']:>4}/{v} = {stage['motion']/v:.3f}   <- the reported number")
print(f"   ...survives vote gate     {stage['vote']:>4}/{v} = {stage['vote']/v:.3f}")
