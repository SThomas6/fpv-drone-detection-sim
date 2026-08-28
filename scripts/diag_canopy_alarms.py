"""Where do the canopy config's alarms and coverage actually live?

Replicates fuse_eval's fused pipeline with v6 at the 84%@237 operating point
(and-confirm, thr 0.5, rgb-mode both, young-tracks pass) and attributes every
surviving track-frame: young vs judged, coasting vs fresh, travel, history
length, IR persistence.
"""
import json, math, sys
from collections import defaultdict, deque
import numpy as np
sys.path.insert(0, r"C:/Users/sctho/Projects/drone sim/fpv-drone-detection-sim")
from camera.classify import MotionClassifier
from camera.evaluate import hits_object, is_hit
from camera.fuse_eval import fuse_measurements, load_clip
from camera.tracking import CentroidTracker
from pathlib import Path

clip = Path("data/clips/terrain_ir_canopy")
clf = MotionClassifier.load("camera/motion_classifier_v6.json")
rows = load_clip(clip, 0.05, 0.2, "both")
tracker = CentroidTracker(class_consistent=True, suppress_spawn_near_coasting=True)
ir_seen = defaultdict(lambda: deque(maxlen=90))
surv = []          # surviving track-frames under the 84%@237 gates
drone_frames = defaultdict(list)
for f, gt, rgb_dets, ir_dets in rows:
    dets, sources = fuse_measurements(rgb_dets, ir_dets)
    tracks = tracker.update(dets, timestamp=gt["t"])
    boxmap = {tuple(round(v,1) for v in d.xyxy): s for d,s in zip(dets,sources)}
    for tr in tracks:
        src = boxmap.get(tuple(round(v,1) for v in tr.box))
        if tr.misses == 0 and src is not None:
            ir_seen[tr.track_id].append("ir" in src)
        seen = ir_seen[tr.track_id]
        irf = (sum(seen)/len(seen)) if seen else 0.0
        xs=[h[1] for h in tr.history]; ys=[h[2] for h in tr.history]
        travel = math.hypot(max(xs)-min(xs), max(ys)-min(ys)) if len(xs)>=2 else 0.0
        feats = clf.features(tr.history)
        p = clf.probability(feats) if feats is not None else None
        # the 84%@237 gate: (p>=0.5 or young) and travel>=8 and ir_frac>=0.3
        if not ((p >= 0.5 if p is not None else True) and travel >= 8.0 and irf >= 0.3):
            continue
        is_drone = bool(gt["visible"] and is_hit(tr.box, gt))
        on_bird = any(hits_object(tr.box, b) for b in (gt.get("birds") or []))
        rec = {"frame": f, "drone": is_drone, "bird": on_bird,
               "clutter": not is_drone and not on_bird,
               "young": p is None, "coasting": tr.misses > 0,
               "nhist": len(tr.history), "travel": travel, "irf": irf,
               "tid": tr.track_id}
        surv.append(rec)
        if is_drone:
            drone_frames[f].append(rec)

mins = 5.0
alarms = [r for r in surv if not r["drone"]]
print(f"surviving track-frames: {len(surv)}  drone-covered frames: {len(drone_frames)}/600")
print(f"alarm track-frames: {len(alarms)} = {len(alarms)/mins:.0f}/min "
      f"(bird {sum(1 for r in alarms if r['bird'])}, clutter {sum(1 for r in alarms if r['clutter'])})")

def frac(rows_, key): return sum(1 for r in rows_ if r[key]) / max(len(rows_), 1)
print("\n--- alarm anatomy ---")
print(f"  young (no classifier verdict): {frac(alarms,'young'):.2%}")
print(f"  coasting (no detection this frame): {frac(alarms,'coasting'):.2%}")
print(f"  young AND coasting: {sum(1 for r in alarms if r['young'] and r['coasting'])/len(alarms):.2%}")
for lo,hi in [(8,16),(16,32),(32,64),(64,1e9)]:
    n=sum(1 for r in alarms if lo<=r["travel"]<hi)
    print(f"  travel {lo:>3}-{hi if hi<1e9 else 'inf':>4}px: {n:>5} ({n/len(alarms):.1%})")
hist=[r["nhist"] for r in alarms]; hist.sort()
print(f"  history len: p10 {hist[len(hist)//10]} med {hist[len(hist)//2]} p90 {hist[9*len(hist)//10]}")

print("\n--- what would each candidate gate cost/save? ---")
cov0 = len(drone_frames)
def evaluate(name, keep_fn):
    kept=[r for r in surv if keep_fn(r)]
    cov=len({r["frame"] for r in kept if r["drone"]})
    al=[r for r in kept if not r["drone"]]
    print(f"  {name:38s} coverage {cov}/600={cov/600:.1%}  alarms {len(al)/mins:6.0f}/min")
evaluate("baseline (current 84%@237)", lambda r: True)
evaluate("no-coast (alarm only on detection)", lambda r: not r["coasting"])
evaluate("young must be non-coasting", lambda r: not (r["young"] and r["coasting"]))
evaluate("min-travel 16", lambda r: r["travel"]>=16)
evaluate("min-travel 32", lambda r: r["travel"]>=32)
evaluate("ir-persist 0.5", lambda r: r["irf"]>=0.5)
evaluate("drop young entirely (young=drop)", lambda r: not r["young"])
evaluate("young only if travel>=16", lambda r: (not r["young"]) or r["travel"]>=16)
evaluate("no-coast + min-travel 16", lambda r: not r["coasting"] and r["travel"]>=16)

print("\n--- drone coverage anatomy (what the drone's surviving frames rely on) ---")
dro=[r for r in surv if r["drone"]]
print(f"  drone track-frames young: {frac(dro,'young'):.2%}  coasting: {frac(dro,'coasting'):.2%}")
only_young = sum(1 for f,rs in drone_frames.items() if all(r['young'] for r in rs))
only_coast = sum(1 for f,rs in drone_frames.items() if all(r['coasting'] for r in rs))
print(f"  covered frames held ONLY by young tracks: {only_young}")
print(f"  covered frames held ONLY by coasting track-frames: {only_coast}")
tids = {}
for r in dro: tids.setdefault(r['tid'], 0); tids[r['tid']] += 1
print(f"  distinct drone track ids: {len(tids)} (top: {sorted(tids.values(),reverse=True)[:8]})")
