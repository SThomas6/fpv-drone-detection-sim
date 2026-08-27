# Phase 2 results — camera detection

Measured 2026-08-27 on the machine described in the README (Apple Silicon,
8 GB, MPS), against clips recorded from the simulator with ground-truth labels
derived from the simulator's own pose data.

Model: `sapoepsilon/yolov11s-drone-detector` **as downloaded — no fine-tuning**.
Inference at native scale, imgsz (736, 1280).

## Detection

| Clip | Frames (drone visible) | conf | Recall | Precision | FP/min |
|---|---|---|---|---|---|
| baseline — mission pattern, 40–78 m | 300 (250) | 0.10 | 1.000 | 0.992 | 0.8 |
| crossing — lateral passes, 40–120 m | 300 (300) | 0.10 | 0.993 | 1.000 | 0.0 |
| low_altitude — terrain background | 299 (299) | 0.10 | 1.000 | 1.000 | 0.0 |
| range_sweep — 8–250 m | 1000 (1000) | 0.05 | 0.892 | 0.986 | 1.9 |
| no_drone — empty sky | 300 (0) | any | — | — | **0.0** |

Zero false positives across 300 frames of empty sky at every threshold tested,
which is the result that makes the recall numbers meaningful.

## Recall by target size

The `<32 px` buckets are the ones that decide the model; everything else is easy.

| Bucket | Recall (range_sweep, conf 0.05) |
|---|---|
| `<16 px` | 0.887 (848/956) |
| `16–32 px` | 1.000 (37/37) |
| `32–96 px` | 1.000 (7/7) |

## Detection range curve

Recall stays at 1.000 out to the 190–220 m bin and drops to 0.853 at 220–260 m.
At 220 m the drone is roughly **1.7 px wide**.

| Range | Recall |
|---|---|
| 0–130 m | 1.000 |
| 130–160 m | 1.000 |
| 160–190 m | 0.967 |
| 190–220 m | 1.000 |
| 220–260 m | 0.853 |

## Confidence sweep

Chosen operating point: **conf = 0.10**. Full recall inside 130 m, no false
positives. 0.05 adds recall past 200 m at the cost of a few false positives;
above 0.30 recall falls away quickly because distant targets score low.

## Throughput

| Mode | Latency (median) | FPS | Notes |
|---|---|---|---|
| Full frame, imgsz (736,1280), MPS | 58–75 ms | 13–17 | camera runs at 15 Hz |
| SAHI sliced, 320 px slices | 560 ms | 1.8 | accuracy mode only |

MPS output verified identical to CPU on real frames (guards a known class of
silent box-corruption bug on Apple GPUs).

## SAHI: measured, not adopted

On the hardest frames (range ≥ 200 m, 755 frames):

| Mode | Recall @0.05 | Recall @0.10 | FPS |
|---|---|---|---|
| Full frame | 0.858 | 0.592 | 17 |
| SAHI sliced | 0.889 | 0.856 | 1.8 |

Slicing buys ~3 points of recall at 0.05 — and notably makes detections much
more confident (0.856 vs 0.592 at conf 0.10) — but at 9.5x the compute it
cannot keep up with a 15 Hz camera. Kept as an opt-in accuracy mode
(`--sahi`), not the default.

## Tracking

Dense 10 Hz clip, 999 frames, single drone.

| Metric | CentroidTracker (ours) | ByteTrack (ultralytics) |
|---|---|---|
| Tracking recall | **0.999** | 0.997 |
| Distinct IDs for one drone (ideal 1) | **1** | 14 |
| ID switches | **0** | 24 |
| Spurious tracks | **0** | 14 |
| Velocity error, absolute | **1.7 px/s** | 1.9 px/s |
| Velocity error, relative (targets >20 px/s) | **0.09** median, 0.31 p90 | 0.11 median, 0.69 p90 |
| Longest gap without a track | 1 frame | 1 frame |

Why a custom tracker: measured on this camera, the drone moves further than its
own box width in **27% of frames**, so consecutive boxes do not overlap and IoU
is exactly zero. Every tracker shipped with ultralytics associates by IoU, so
all of them fragment identity here. Associating on centre distance against a
constant-velocity Kalman prediction fixes it outright. See
[detection-notes.md](detection-notes.md).

## Honest caveats

- **These numbers are optimistic and should not be read as real-world
  performance.** The simulated sky is uniform and noise-free, so a dark
  2 px speck is trivially separable. Real skies bring cloud, haze, sensor
  noise, and birds. Detecting a 1.7 px target at 220 m would not survive
  contact with a real camera.
- No fine-tuning was needed *for the simulator*, which says more about how easy
  synthetic imagery is than about the model. The fine-tune plan in
  [model-selection.md](model-selection.md) still stands for real footage.
- Only one drone is ever in frame. Multi-target association is untested.
- Bird/drone discrimination was untested when the table above was measured.
  It has since been measured, and it is bad — see "Birds and lighting" below.
- Distances beyond 260 m were not measured.

---

# Birds and lighting (added 2026-08-27)

## Lighting: not a problem

Two hard lighting cases were added (`simulator/lighting.py`, five presets) and
re-flown. The detector did not care:

| Condition | Recall | Precision |
|---|---|---|
| midday (baseline) | 1.000 | 0.992 |
| backlit — low sun ahead of the camera | 1.000 | 0.998 |
| dusk — dim, low, warm light | 1.000 | 1.000 |

A dark target against a bright sky stays a dark target against a bright sky.

## Birds: the real problem

Birds were added (`simulator/birds.py`) as procedurally-generated models across
five size classes, flying drifting soaring circles with a 2.5–7 Hz roll
oscillation standing in for wingbeat. At 100 m a gull is ~13 px across while
the drone is ~4 px, so **the distractor is the larger target**.

With birds flying and **no drone present at all**:

| conf | False alarms | Per minute | On a bird |
|---|---|---|---|
| 0.05 | 1434 | 662 | 1414 (99%) |
| 0.25 | 988 | 456 | — |
| 0.50 | 412 | 190 | — |

The detector calls birds drones, constantly. This is the single biggest
weakness found in Phase 2, and it was completely invisible until birds existed —
the earlier "zero false positives" result only ever tested empty sky.

### Confidence cannot fix it

| | mean | median | max |
|---|---|---|---|
| detections on the drone | 0.574 | 0.587 | 0.741 |
| detections on birds | 0.358 | 0.367 | **0.852** |

Birds score *higher at the top end than the drone ever does*. Thresholding
trades one error for the other and never separates them:

| threshold | drone detections kept | bird alarms kept |
|---|---|---|
| 0.25 | 100% | 66% |
| 0.50 | 81% | 28% |
| 0.70 | 6% | 4% |

## Why not a second (bird) model

The user proposed running a bird detector alongside and suppressing the drone
when the bird model is more confident. Researched and rejected, for three
reasons:

1. **The scores are not comparable.** Ultralytics confidence is not a
   probability; in this stack it is effectively a regressed box-overlap
   quality, which is why it tracks pixel size so cleanly in the table above.
   Comparing that number across two independently-trained networks is not a
   principled decision rule.
2. **The available bird models are weak.** The best verified candidate
   (`Javvanny/yolov8m_flying_objects_detection`) reports 68% true-positive on
   its own bird class and costs ~3.7x the compute of the current model — far
   too weak to act as a veto, and it breaks the frame-rate budget.
   `Hibou-Foundation/rtdetr-drone-detection`'s second class is "other", not
   bird, and it declares no license.
3. **There are no pixels left to argue about.** Past ~150 m the target is 2–3
   px. A second appearance model would be guessing on the same handful of
   pixels, at half the frame rate.

The principled version of the idea — one detector with both classes sharing a
feature extractor, so the scores are comparable by construction — is the right
long-term answer, but it requires fine-tuning on drone+bird data.

## What was implemented instead: motion

Birds and multirotors *move* differently, and motion survives at ranges where
appearance does not. `camera/classify.py` classifies a TRACK (not a frame) with
six features from history the tracker already keeps: straightness, hover
fraction, projected-size variability (the flapping signature), turn rate, speed
variability, and speed in target-widths per second. A six-weight logistic
regression, fitted in numpy — no new dependency, negligible compute.

The learned weights are physically sensible rather than arbitrary:

| feature | weight | reading |
|---|---|---|
| width_cv | **−4.83** | oscillating projected size ⇒ bird (flapping) |
| hover_fraction | **+3.35** | holding station ⇒ drone (birds can't) |
| speed_cv | +2.96 | stop/start ⇒ drone |
| turn_rate | −2.11 | wheeling ⇒ bird |

Held out on `birds_drone` (range sweep with 10 birds, never used in training),
declaring a target only once a track has enough history to judge:

| threshold | drone track-frames kept | bird alarms/min | reduction |
|---|---|---|---|
| none (before) | 100% | 335 | — |
| 0.50 | 69% | 53 | **84%** |
| 0.95 | 62% | 31 | **91%** |

Delay from first sighting to first confirmed call: **2.4 s**.

### Honest reading

This is a large improvement, not a solution. 31–53 false alarms per minute is
still far too many for a real system, and ~30% of drone track-frames are given
up to get it (the drone is still declared, just not on every frame).

It is also fitted to *simulated* bird motion. The mechanism is real — flapping
oscillation, wheeling flight, and the fact that a multirotor can hover and a
bird cannot — but the specific decision boundary is not evidence about real
birds. Treat the 84–91% reduction as a demonstration that motion carries the
signal, not as a performance claim.

The next real step is the one the research recommends: fine-tune a **two-class
(drone, bird) detector** so appearance and motion evidence can be combined
properly, and validate on real Drone-vs-Bird footage.
