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

---

# The two-class fine-tune (added 2026-08-28, measured on the PC / RTX 3060 Ti)

Eight training runs and an overnight measurement campaign later, the two-class
detector is installed (`camera/weights/drone_bird_v1.pt`) — but what it buys is
different from what was hoped, and the honest account matters more than the
headline.

All numbers below are from the strictly held-out `eval_birds` clip (mission
flight + 8 birds, never used in training) and the drone-only `baseline` clip,
scored by the project evaluator (centre-distance matching).

## What was hoped, and what happened

The hope: a detector that keeps drone recall ≥ 0.99 while its bird class
absorbs the ~439/min false alarms birds cause on this clip. What happened, in
every recipe tried: **appearance alone cannot do both at 4–6 px.** The
detector-alone Pareto frontier would not move:

| recipe axis probed (40–60 epochs each) | recall ≥ 0.99 costs | low FP costs |
|---|---|---|
| longer training / pinned AdamW control | ~284 FP/min | recall ≤ 0.93 |
| upscale-only scale jitter (label-safe) | ~320 FP/min | recall ≤ 0.92 |
| cls loss ×3 | never reaches 0.99 | recall ≤ 0.87 |
| combo + per-class BCE weights | ~278 FP/min | recall ≤ 0.83 |
| P2 stride-4 head graft | never reaches 0.99 | recall ≤ 0.86 |
| 3× drone-tile oversampling | never reaches 0.99 | recall ≤ 0.95 |
| mosaic 0.1 (IJCNN winner recipe) | ~352 FP/min | recall ≤ 0.84 |

Two measurement traps found on the way (both verified in ultralytics 8.4.131
source): `optimizer='auto'` silently ignores `lr0` (it ran AdamW @ 0.001667,
not the printed 0.01), and `best.pt` is selected by pure mAP50-95 — noise on
2–10 px boxes — so every run saved per-epoch checkpoints and selection used
the held-out evaluator instead.

## The diagnosis that explains the frontier

For every fine-tune, essentially **all** lost drone frames are *hard class
flips*: the drone is perfectly localised but labelled `bird`, with zero
drone-class score even at a 0.03 floor, concentrated at 48–77 m where the
drone is 4–6 px. (The old single-class model cannot make this mistake — that
is the only reason it scores recall 1.000.) The mirror image: when the
detector false-fires `drone` on a bird, it emits *no* bird box at that spot,
so cross-class vetoes at frame level were measured to do nothing. At these
sizes the two classes are not separable per-frame; the information is simply
not in the pixels. Phase 2's "no pixels left to argue about" conclusion,
re-confirmed the hard way.

## Where the fine-tune actually wins: the track level

Per-frame flips average out over a track, and motion still separates the
classes. Two pipeline changes (in `camera/tracking.py` / `classify.py` /
`detect_live.py`):

1. **Track every class.** Filtering to drone-class detections before tracking
   punched holes in the drone's own track exactly on flip frames — and,
   ironically, made *bird* tracks fragmentary too, starving the motion
   classifier (an intermittently-detected bird has no readable wingbeat).
   Class is now recorded per history sample instead of used as a pre-filter.
2. **Appearance votes per track** (`drone_vote_fraction`) — measured honestly:
   weak with this model (birds vote "drone" too), kept as a cheap extra gate.

The stacked system on `eval_birds` (detector conf 0.05, motion threshold
sweep):

| system | drone track-frames kept | alarms/min |
|---|---|---|
| old detector, no filter | 100% | 438.8 |
| old + motion 0.50 | 89.3% | 9.8 — **its floor is ~9.5/min at any threshold** |
| two-class + motion 0.50 | 82.1% | 7.5 |
| two-class + motion 0.90 | 71.9% | **0.5** |
| two-class + motion 0.95 | 67.7% | **0.0** |

That last column is the result of the campaign: the two-class detector plus
continuous tracking gives the system a **~20× lower achievable false-alarm
floor** (0–0.5/min vs 9.5/min). The cost is frame coverage, not silence: the
drone stays continuously tracked (single ID, 2.4 s from first sighting to
first confirmed call) and is declared on 68–72% of its visible frames instead
of 89%.

Detector-alone regression gates, installed weights (`drone_bird_v1.pt` =
control-run epoch 40, selected by held-out sweep): `baseline` recall **1.000**,
precision 1.000, 0.0 FP/min at conf 0.10 — no regression; `eval_birds` recall
**1.000** at conf 0.05; the model also self-identifies birds (325 bird-class
hits on birds at the operating floor). Offline regression tests: 6/7 (the
seventh needs `data/clips/no_drone`, which was not migrated to this PC).

## Honest caveats

- Checkpoint and threshold selection used the held-out clip, which makes it a
  dev set. The numbers above are fair comparisons between systems, but the
  absolute values need one fresh, never-touched clip for a final quote (noted
  in TODO).
- The 68–72% coverage at ≤0.5 alarms/min is a *different operating point*, not
  strictly better than old+motion at 89%/9.8 — which trade is right depends on
  how expensive a false alarm is downstream. Both configurations remain
  available (the old model still loads if `drone_bird_v1.pt` is removed).
- Everything here is synthetic: simulated birds, clean skies. The mechanism
  (track-level fusion beats per-frame appearance at tiny sizes) is the
  transferable result; the specific numbers are not.

---

# Terrain backgrounds and EO/IR fusion (added 2026-08-28)

The clean-sky caveat above is now measured. A procedural field/forest world
(`scripts/gen_terrain_world.py`: heightmap meadow, forested hillside, rock
mountain band; PX4-free scripted capture via `scripts/drive_scene.py`, with
occlusion-aware labels) plus a station thermal camera produced 4,900 new
labelled frames across nine clips.

## What terrain does to the RGB system

| Held-out clip | recall @op | FP/min | note |
|---|---|---|---|
| terrain_baseline (mission, mixed bg) | 0.923 | 8.8 | modest degradation |
| terrain_birds (mission + 8 birds) | 0.911 | 197.6 | birds still the FP source |
| terrain_canopy (low legs, terrain bg) | **0.125** | 319.6 | recall collapse |
| terrain_sweep (30-250 m vs mountains) | 0.315 | 0.3 | blind past ~60 m |
| terrain_only (no targets) | — | **0.0** | static terrain never false-fires |

Two mechanisms, both diagnosed: the drone at ≤6 px against textured
rock/canopy is **contrast-limited** (a human squints at it too — verified on
zoomed crops), and fragmented bird tracks against terrain get straightened by
the tracker's coasting, which **breaks the motion filter's bird rejection**
(bird alarms pass at ~300/min vs ~9/min over sky).

A terrain-positive retrain was attempted and appeared to fail — **but that
conclusion was later found to be an artifact of a silent dataset bug** and is
retracted. Ultralytics resolves only `./`-prefixed lines of a txt train list
against the dataset root; the retrain's `../finetune_terrain/...` lines were
resolved against the working directory and silently dropped as "corrupt", so
the run trained on zero terrain tiles (which is why it exactly matched the
sky model). The corrected mixed retrain below shows terrain recall IS
substantially trainable (canopy 0.125 → 0.458), though not to 1.0 — the ≤6 px
contrast ceiling against rock is real, just higher than a sky-only model
reaches. Post-mortem lesson (the project's third): when a retrain changes
nothing at all, first suspect that it trained on nothing at all — read the
scanner's corrupt count.

## The thermal camera, honestly

Gazebo Harmonic's thermal sensor (Boson-class 640x512, 24° HFOV, 16-bit
Kelvin) with **honest temperatures**: birds are warm (292 K torso — they glow
in real LWIR too), the drone is mostly ambient airframe with 315 K motor
bells and a 305 K battery stack, and a quarter of the bushes are sun-warmed
distractors. A degradation shim (NETD noise + optics blur) runs before any
detection. The detector is deliberately classical (`camera/ir_detector.py`:
threshold + connected components + peak-contrast gate) — raw sim thermal is a
temperature-ID map any learned model would trivially memorise.

Result: on the canopy profile where RGB sees the drone in 12.5% of frames,
the thermal channel sees it in **100%** — terrain is ambient, motors are not.
And as predicted by the literature (US Army Research Lab), thermal does NOT
separate birds from drones; it is a clutter/night/range channel, and the
motion classifier keeps the bird-discrimination job.

## Track-level EO/IR fusion (`camera/fuse_eval.py`)

One shared tracker consumes both sensors' detections (co-located measurements
merge; per-track IR persistence recorded); alarms additionally require net
track travel ≥8 px, which kills stationary warm-clutter tracks outright.

| Held-out clip | policy | drone coverage | alarms/min |
|---|---|---|---|
| terrain_ir_sweep (long range) | RGB-only | 88/400 (22%) | 3.9 |
| | **fused** | **358/400 (89.5%)** | **1.5** |
| terrain_ir_birds | RGB-only | 492/573 (86%) | 375 |
| | AND-confirm | 298/573 (52%) | 99 |
| terrain_ir_canopy | RGB-only | 86/600 (14%) | 584 |
| | AND-confirm | 153/600 (26%) | 288 |

The long-range result is the headline: fusion quadruples coverage while
halving alarms. The bird clips remain poor in every policy — cross-modal
confirmation cuts bird alarms 76%, but the underlying per-frame bird problem
is unchanged by thermal (by design of the honest benchmark).

## Honest caveats

- **The IR range gain is not yet attributed.** The thermal camera's 24° optics
  put ~36% more pixels on target than the 60° RGB at any range; a narrow-FOV
  RGB ablation (queued in TODO) is required before claiming the gain comes
  from thermal physics rather than lens choice. Real systems resolve the
  FOV-vs-coverage tension with slew-to-cue, not one wide staring camera.
- Sim thermal has no emissivity/atmosphere/AGC physics; per-visual uniform
  temperatures only. It is evidence about geometry, persistence, and fusion
  logic — never about thermal appearance models.
- The trees do not sway and the thermal scene is static-warm; the ≥8 px
  travel gate's clean kill of clutter is an upper bound on real performance.
- Bird-track fragmentation breaking the motion filter over terrain is now the
  biggest open problem in the system; track gap-filling and a track-sequence
  classifier (Drone-vs-Bird literature's winning increments) are the queued
  levers, followed by real-footage validation on Anti-UAV (paired RGB+IR).

---

# Real footage, and the model that holds every domain (added 2026-08-28, evening)

## The reality check

The full pipeline was run against real drone videos (Anti-UAV300 test split,
12 sequences, hand-annotated; imported to our clip format by
`scripts/import_antiuav.py` — pseudo-range from box width, noted in each
clip's meta). Verdict on the sim-trained detector: **catastrophic forgetting.**
Mean recall 0.08 at 2,000-3,700 FP/min — while the ORIGINAL base checkpoint
(trained on real photos) scores recall 0.99-1.00 at ~6 FP/min on the same
sequences. Forty sim-only epochs had erased what the base model knew about
real imagery. Every sim result in this file describes a model that, as of
that measurement, did not work outside the simulator.

## The fix: one mixed retrain across all four domains

25 epochs over 8,668 tiles — sim sky + sim terrain + zoom-lens terrain + real
Anti-UAV train-split frames (real tiles double-weighted; test sequences never
touched; checkpoint chosen by held-out sweep). Deployed as
`camera/weights/drone_bird_v1.pt` (m1 epoch 24; the previous sim-specialist
weights are superseded). Detector-level, max recall @ FP/min:

| Held-out clip | sim-specialist (before) | mixed model (after) |
|---|---|---|
| real seq 1_4 | 0.280 @ 2,048 | **1.000 @ 0** |
| real seq 1_5 | 0.200 @ 3,326 | **0.952 @ 260** |
| real seq 1_9 | 0.136 @ 2,966 | **0.766 @ 35** (beats even the base model's 0.609 @ 16) |
| baseline (sim sky) | 1.000 @ 0 | 0.940 @ 0 |
| terrain_birds | 0.911 @ 170 | 0.915 @ **52** |
| terrain_canopy | 0.125 @ 306 | **0.458 @ 69** |
| terrain_zoomsweep | 0.186 @ 0 | **0.693 @ 1** |

System-level on sky-birds (tracker + motion + class votes): 60% drone
track-frame coverage at **0.2 alarms/min** — and the per-track appearance
votes, useless under the old model (birds voted "drone"), now genuinely work.
The cost of cross-domain competence is ~6 pts of sim-sky recall: a fair trade
for a model that functions on real video.

## Honest caveats

- 12 real test sequences from one dataset, one country, mostly urban/sky
  backgrounds; real-world bird pressure remains unmeasured (Anti-UAV has no
  bird labels — the bird class is still sim-taught).
- Real sequences carry a burned-in tracking-turret reticle/OSD; the model
  trained on tiles containing it, which flatters FP counts on this dataset.
- Thermal fusion is not yet validated on real data: Anti-UAV's IR is 8-bit
  processed video from a separate, unregistered camera — the sim fusion
  design needs a contrast-based port before it can be tested there.
