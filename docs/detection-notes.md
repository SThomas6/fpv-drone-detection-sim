# Detection engineering notes

Non-obvious things learned building Phase 2. Most cost real debugging time and
would silently degrade accuracy if reintroduced.

## The measurement was wrong before the model was

The first evaluation reported 31% recall and a suspiciously flat confidence
sweep. The model was fine — the **dataset labels were wrong**. The recorder
paired the newest camera frame with the newest pose, and when the camera stream
stalled it kept writing stale pixels against fresh poses. Every detection was
scored against a label pointing at empty sky.

Two safeguards now make this class of bug impossible to ship:

1. `capture_dataset.py` skips frames whose image sequence number has not
   advanced, or whose image is more than 2 s old.
2. Every capture **verifies its own labels against the rendered pixels**: the
   drone is by far the darkest thing against sky, so its true position can be
   found without a model. Median error is reported (currently ~1 px) and the
   capture warns loudly if under 90% of frames agree within 15 px.

Lesson worth keeping: when results look bad, verify the measurement before
tuning the thing being measured. When they look *good*, verify it too.

A related aliasing bug lived in the tracking evaluation: it stored references to
live track objects that the tracker mutates in place, so every frame read back
the tracker's *final* state. That made a working Kalman filter look like it was
reporting zero velocity. Snapshot values, never object references.

## Colour order differs per code path

Ultralytics interprets a numpy array as **BGR** (matching OpenCV). Gazebo
publishes **RGB**. SAHI, however, documents RGB input and flips internally.

- `model.predict()` / `model.track()` → pass `frame[:, :, ::-1]`
- `get_sliced_prediction()` → pass the frame **unconverted**

Getting either wrong costs accuracy silently — nothing errors.

## imgsz must be a tuple

`imgsz=1280` can letterbox a 1280x720 frame to 1280x1280 depending on flags
that are set several layers down. `imgsz=(736, 1280)` pins native scale under
every branch, so a 6 px drone is never downscaled. Both values are stride-32
multiples. Never use `imgsz=640` here — it halves the target to 2.5 px, below
the finest detection grid.

## The default tracker is not ByteTrack

In ultralytics 8.4.130 `cfg/default.yaml` sets `tracker: tracktrack.yaml`.
Omitting the argument silently selects an algorithm nobody chose. The tracker
YAML is passed by absolute path, and lives outside the package tree because
`check_yaml()` globs the package and raises on a duplicate filename.

## IoU association fails on targets this small

Measured on the station camera: the drone moves further than its own box width
in **27% of frames**. Consecutive boxes then have IoU exactly 0, and every
tracker shipped with ultralytics (ByteTrack, BoT-SORT, OC-SORT, TrackTrack)
associates by IoU. Result: 14 identities for one drone.

`camera/tracking.py` associates on **centre distance** against a
constant-velocity Kalman prediction, with a gate that grows with target size,
speed and time coasted. Result: 1 identity, 0 switches.

If ByteTrack is ever used again, `fuse_score: false` is essential — with the
shipped default the gate becomes `IoU >= (1 - match_thresh) / score`, so a
detection scoring below ~0.20 can never match at any IoU. Distant drones score
0.2–0.5. The tuned config is in `camera/drone_bytetrack.yaml`.

## NMS cannot deduplicate tiny targets

Two boxes 3 px apart on a 4 px drone barely overlap, so NMS keeps both and the
tracker sees two targets — which then spawn phantom tracks that coast for
seconds. `merge_close()` in `detector.py` merges detections by centre distance
before tracking, keeping the highest-confidence box. Same principle as the
tracker: at this scale, distance is the meaningful metric and IoU is not.

## OpenMP: three runtimes in one process

conda ships `libomp` and `libiomp5`; pip's torch bundles its own `libomp`.
Importing torch aborted with `OMP: Error #15`. The documented
`KMP_DUPLICATE_LIB_OK=TRUE` workaround is explicitly unsafe — it can silently
produce wrong numbers, which is unacceptable in a detector. Both libraries are
LLVM builds with the same ABI (compat version 5.0.0), so torch's copy is
symlinked to conda's:

```
ln -sf $CONDA_PREFIX/lib/libomp.dylib \
       $CONDA_PREFIX/lib/python3.12/site-packages/torch/lib/libomp.dylib
```

The original is kept as `libomp.dylib.bak`. Verified afterwards that CPU and
MPS matmuls agree exactly.

## MPS needs a startup parity check

Ultralytics has a history of strided-tensor bugs on MPS corrupting box
coordinates, with the signature that x-coordinates collapse to the frame
height. The clamp bug is fixed in 8.4.111+, but `scale_boxes` still performs
unguarded strided in-place ops. `check_device_parity()` compares MPS against
CPU on a frame **containing a drone** (comparing empty frames proves nothing)
and is run once at startup by `detect_live.py`. Both sides must go through
identical post-processing or the comparison is meaningless — that mistake cost
a failing test.

## Other settings that matter

- `PYTORCH_ENABLE_MPS_FALLBACK=1` must be set before torch is imported.
- `cv2.setNumThreads(2)` stops OpenCV oversubscribing against torch's pool.
- Prediction kwargs must be identical between calls — ultralytics rebuilds the
  predictor (destroying tracker state) if `device` differs even in spelling.
- Keep the input shape constant; do not call `torch.mps.empty_cache()` per
  frame. Memory growth comes from the shape-keyed compile cache, which
  `empty_cache()` does not reclaim.
- `r.boxes` is never `None` for detection, but `r.boxes.id` **is** `None`
  whenever no confirmed tracks exist, even when boxes are present.
