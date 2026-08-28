#!/usr/bin/env python3
"""Phase 2 regression test: detection and tracking, offline.

Runs against stored clips, so it needs no simulator:

    conda activate dronesim && python tests/test_phase2.py

Checks:
  1. weights load and expose the expected single 'drone' class
  2. MPS agrees with CPU on a real frame (guards silent box corruption)
  3. detection finds the drone on frames where ground truth says it is visible
  4. no false positives on empty-sky frames
  5. duplicate boxes on one tiny target are merged
  6. the Kalman tracker recovers a known constant velocity
  7. tracking holds one ID over a sequence
  8. the static-clutter map mutes a fixed blob but never a moving target

Exit code = number of failed checks.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from camera.clutter_map import StaticClutterSuppressor  # noqa: E402
from camera.detector import Detection, DroneDetector, merge_close  # noqa: E402
from camera.evaluate import is_hit  # noqa: E402
from camera.tracking import CentroidTracker  # noqa: E402

RESULTS: list[bool] = []


def report(name: str, ok: bool, detail: str = "") -> bool:
    RESULTS.append(bool(ok))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""),
          flush=True)
    return ok


def load(clip: Path, rec) -> np.ndarray:
    return np.array(Image.open(clip / "frames" / rec["frame"]).convert("RGB"))


def labels_of(clip: Path):
    return [json.loads(l) for l in open(clip / "labels.jsonl")]


def main() -> None:
    baseline = ROOT / "data/clips/baseline"
    empty = ROOT / "data/clips/no_drone"
    if not baseline.exists():
        print("no recorded clips — run scripts/capture_dataset.py first")
        sys.exit(1)

    print("=== Phase 2 regression test ===")

    print("[1/8] model loads")
    det = DroneDetector()
    ok_names = det.names in ({0: "drone"}, {0: "drone", 1: "bird"})
    report("expected class map", ok_names, str(det.names))
    det.warmup()

    recs = [r for r in labels_of(baseline) if r["visible"]]
    sample = recs[::max(1, len(recs) // 12)][:12]

    print("[2/8] device parity (mps vs cpu)")
    probe = next((r for r in sample if det.detect(load(baseline, r))), None)
    if probe is None:
        report("parity check", False, "no detections to compare on")
    else:
        problems = det.check_device_parity(load(baseline, probe))
        report("mps matches cpu", not problems, "; ".join(problems) or "identical")

    print("[3/8] detection on frames with a drone")
    hits = sum(1 for r in sample
               if any(is_hit(d.xyxy, r) for d in det.detect(load(baseline, r))))
    report("recall on sampled frames", hits >= len(sample) * 0.9,
           f"{hits}/{len(sample)}")

    print("[4/8] no false positives on empty sky")
    if empty.exists():
        blanks = labels_of(empty)[::20][:10]
        fps = sum(len(det.detect(load(empty, r))) for r in blanks)
        report("empty-sky false positives", fps == 0, f"{fps} on {len(blanks)} frames")
    else:
        report("empty-sky clip present", False, "data/clips/no_drone missing")

    print("[5/8] duplicate merging for tiny targets")
    dup = [Detection(100, 100, 106, 106, 0.6), Detection(103, 102, 109, 108, 0.4),
           Detection(400, 300, 406, 306, 0.5)]
    merged = merge_close(dup)
    report("near-duplicate boxes merged", len(merged) == 2,
           f"{len(dup)} -> {len(merged)}")

    print("[6/8] tracker recovers a known velocity")
    tr = CentroidTracker()
    est = None
    for i in range(25):
        t = i * 0.1
        x = 200.0 + 100.0 * t
        out = tr.update([Detection(x - 2, 298, x + 2, 302, 0.5)], timestamp=t)
        if out:
            est = out[0].velocity
    ok = est is not None and abs(est[0] - 100.0) < 10.0 and abs(est[1]) < 10.0
    report("constant velocity recovered", ok,
           f"estimated {est[0]:.1f},{est[1]:.1f} px/s vs true 100.0,0.0" if est else "no track")

    print("[7/8] one ID held across a sequence")
    seq = labels_of(baseline)[:60]
    tr2 = CentroidTracker()
    ids = set()
    for r in seq:
        for t in tr2.update(det.detect(load(baseline, r)), timestamp=r["t"]):
            ids.add(t.track_id)
    report("single track ID for a single drone", len(ids) <= 1, f"{len(ids)} IDs")

    print("[8/8] static-clutter map: mutes a fixed blob, spares a mover")
    # Both run 30 frames at 0.2 s. The blob sits still; the mover crosses at
    # 18 px/frame, the drone's measured pace on the real footage. An earlier
    # version of the map drifted its anchors toward each new observation, so
    # the anchor FOLLOWED the mover and suppressed it - recall collapsed to
    # 0.46. This pins that behaviour: persistence alone is not enough, the
    # anchor's spatial extent has to be what decides.
    fixed = StaticClutterSuppressor()
    moving = StaticClutterSuppressor()
    blob_muted = mover_muted = 0
    for i in range(30):
        t = i * 0.2
        _, drop_f = fixed.step([Detection(500, 500, 560, 560, 0.5)], t)
        x = 100.0 + 18.0 * i
        _, drop_m = moving.step([Detection(x, 500, x + 60, 560, 0.5)], t)
        blob_muted += len(drop_f)
        mover_muted += len(drop_m)
    report("fixed blob suppressed once established", blob_muted > 15,
           f"{blob_muted}/30 frames muted")
    report("moving target never suppressed", mover_muted == 0,
           f"{mover_muted}/30 frames muted")

    fails = RESULTS.count(False)
    print(f"\n=== {len(RESULTS) - fails}/{len(RESULTS)} checks passed ===")
    sys.exit(fails)


if __name__ == "__main__":
    main()
