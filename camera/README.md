# camera — visual drone detection (Phase 2, complete)

Consumes the station camera feed and reports whether a drone is present, where
it is, and where it is going.

| File | Purpose |
|---|---|
| `detector.py` | YOLOv11s drone detector; full-frame or SAHI sliced inference, duplicate merging, MPS/CPU parity check |
| `tracking.py` | `CentroidTracker` — distance-gated constant-velocity Kalman tracking, built because IoU trackers fail on targets this small |
| `detect_live.py` | CLI: run detection/tracking against the running simulator |
| `evaluate.py` | Score detections against ground truth; two-stage so thresholds sweep without re-running inference |
| `evaluate_tracking.py` | ID stability, continuity and velocity accuracy |
| `visualize.py` | Render detections vs ground truth, with zoomed crops |
| `drone_bytetrack.yaml` | Tuned ByteTrack config, kept for comparison |

Quick start (simulator must be running):

```bash
conda activate dronesim && python camera/detect_live.py --track
```

Results: [docs/phase2-results.md](../docs/phase2-results.md).
Model choice: [docs/model-selection.md](../docs/model-selection.md).
Engineering gotchas worth reading before changing anything here:
[docs/detection-notes.md](../docs/detection-notes.md).
