"""Single-feature AUC probe: which track statistics separate drone from bird?"""
import json, sys, math
import numpy as np
sys.path.insert(0, r"C:/Users/sctho/Projects/drone sim/fpv-drone-detection-sim")
from camera.classify import MIN_HISTORY
from camera.evaluate import hits_object, is_hit
from camera.fuse_eval import fuse_measurements, load_clip
from camera.tracking import CentroidTracker
from pathlib import Path

AUX = ("hotspot", "mover")

def feats(h):
    """Rich candidate feature dict from a track history of (t,x,y,w,conf,cls)."""
    if len(h) < MIN_HISTORY:
        return None
    a = np.asarray([(t, x, y, w, c) for (t, x, y, w, c, *_r) in h], dtype=float)
    cls = [e[5] if len(e) > 5 else "drone" for e in h]
    t, x, y, w, cf = a[:, 0], a[:, 1], a[:, 2], a[:, 3], a[:, 4]
    dt = np.diff(t); good = dt > 1e-6
    if good.sum() < 4: return None
    dx, dy, dt = np.diff(x)[good], np.diff(y)[good], dt[good]
    step = np.hypot(dx, dy); speed = step / dt
    path = step.sum(); net = math.hypot(x[-1]-x[0], y[-1]-y[0])
    mw = max(float(w.mean()), 1.0)
    head = np.arctan2(dy, dx); dh = np.diff(head)
    dh = (dh + np.pi) % (2*np.pi) - np.pi
    vx, vy = dx/dt, dy/dt
    acc = np.hypot(np.diff(vx), np.diff(vy)) / dt[1:] if len(dt) > 1 else np.array([0.0])
    rgbh = [c for c in cls if c not in AUX]
    span = float(t[-1] - t[0]) or 1e-6
    # short-window straightness (last 8 samples)
    k = min(8, len(x))
    p2 = np.hypot(np.diff(x[-k:]), np.diff(y[-k:])).sum()
    n2 = math.hypot(x[-1]-x[-k], y[-1]-y[-k])
    return {
        "straightness":     net/path if path > 1e-6 else 1.0,
        "straight_short":   n2/p2 if p2 > 1e-6 else 1.0,
        "hover_fraction":   float(np.mean(speed/mw < 1.0)),
        "width_cv":         float(w.std()/(w.mean()+1e-6)),
        "turn_rate":        min(float(np.mean(np.abs(dh))/np.mean(dt)) if len(dh) else 0.0, 50.0),
        "turn_rate_cv":     float(np.std(np.abs(dh))/(np.mean(np.abs(dh))+1e-6)) if len(dh) else 0.0,
        "speed_cv":         min(float(speed.std()/(speed.mean()+1e-6)), 10.0),
        "speed_in_widths":  min(float(np.mean(speed/mw)), 50.0),
        "abs_speed":        min(float(speed.mean()), 2000.0),
        "abs_speed_max":    min(float(speed.max()), 4000.0),
        "accel_mean":       min(float(acc.mean()), 20000.0),
        "accel_cv":         float(acc.std()/(acc.mean()+1e-6)),
        "vertical_ratio":   float(np.abs(dy).sum()/(np.abs(dx).sum()+np.abs(dy).sum()+1e-6)),
        "width_mean":       float(w.mean()),
        "width_trend":      float(abs(np.polyfit(t, w, 1)[0])/mw),
        "conf_mean":        float(cf.mean()),
        "conf_cv":          float(cf.std()/(cf.mean()+1e-6)),
        "rgb_frac":         len(rgbh)/len(cls),
        "ir_frac":          sum(1 for c in cls if c == "hotspot")/len(cls),
        "mv_frac":          sum(1 for c in cls if c == "mover")/len(cls),
        "drone_vote":       (sum(1 for c in rgbh if c == "drone")/len(rgbh)) if rgbh else 0.5,
        "support_rate":     len(h)/span,
        "n_samples":        float(len(h)),
        "y_norm":           float(y.mean()/720.0),
    }

def collect(clip):
    rows = load_clip(Path(clip), 0.05, 0.2)
    tk = CentroidTracker(class_consistent=True, suppress_spawn_near_coasting=True)
    X, yy = [], []
    for i, (f, gt, rgb, aux) in enumerate(rows):
        dets, _ = fuse_measurements(rgb, aux)
        for tr in tk.update(dets, timestamp=gt["t"]):
            if i % 3: continue
            od = bool(gt["visible"] and is_hit(tr.box, gt))
            ob = any(hits_object(tr.box, b) for b in (gt.get("birds") or []))
            if od == ob: continue
            fe = feats(tr.history)
            if fe: X.append(fe); yy.append(1.0 if od else 0.0)
    return X, np.array(yy)

def auc(v, y):
    v = np.asarray(v); pos, neg = v[y == 1], v[y == 0]
    if not len(pos) or not len(neg): return float("nan")
    return float((pos[:, None] > neg[None, :]).mean() + 0.5*(pos[:, None] == neg[None, :]).mean())

clips = ["data/clips/terrain_ir_canopy", "data/clips/terrain_ir_birds", "data/clips/eval_birds"]
data = {c: collect(c) for c in clips}
names = list(data[clips[0]][0][0].keys())
print(f"\n{'feature':>18}" + "".join(f"{Path(c).name[:14]:>16}" for c in clips))
scored = []
for n in names:
    a = [auc([d[n] for d in data[c][0]], data[c][1]) for c in clips]
    # informative either way: distance from 0.5
    scored.append((max(abs(x-0.5) for x in a[:2]), n, a))
for _, n, a in sorted(scored, reverse=True):
    print(f"{n:>18}" + "".join(f"{x:>16.3f}" for x in a))
