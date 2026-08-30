# Handoff — continue this project in a fresh session

Written 2026-08-28, updated continuously through that evening's session,
specifically so a **new agent — possibly a different tool entirely (Gemini
Antigravity, Codex, a fresh Claude Code session)** — can pick up the
fusion-improvement campaign without re-deriving anything. Read this file,
then [phase2-results.md](phase2-results.md) for full measured numbers, then
start work.

**First message to give the new session:** "Read docs/handoff.md and
continue the fusion campaign toward the 90% target."

## Environment — read before running anything

- Repo root: `C:\Users\sctho\Projects\drone sim\fpv-drone-detection-sim`
  (Windows 11, RTX 3060 Ti 8 GB).
- **Always use the venv interpreter**, not bare `python`: `.venv/Scripts/python.exe`
  (Python 3.12.8, torch 2.13.0+cu126 with working CUDA, ultralytics 8.4.131,
  sahi 0.12.6). A bare `python` on this box has no torch.
- `scipy` and `sklearn` are **not installed**. Everything is numpy-only by
  design; keep it that way unless you deliberately add a dependency.
- The `.sh` scripts in `scripts/` are bash and do not run on native Windows.
  All the Python entry points are cross-platform.
- **Writing Python files: use UTF-8 explicitly.** `Path.read_text()` /
  `write_text()` default to cp1252 here and will silently corrupt the em-dashes
  in these source files. This already broke `tests/test_phase2.py` once.
- Tests: `.venv/Scripts/python.exe tests/test_phase2.py` — expect **8/9**.
  The single failure is a missing `data/clips/no_drone` (data was never
  migrated to this PC), not a code regression. Anything below 8/9 is real.

## Quick command reference

```bash
# fused evaluation (the number the campaign is judged on)
.venv/Scripts/python.exe camera/fuse_eval.py --clip data/clips/terrain_ir_canopy \
    [--rgb-mode full|sahi|both] [--classifier camera/motion_classifier_v2.json] \
    [--motion-thr 0.5] [--min-travel 8] [--young-tracks drop|pass] [--clutter]

# plain-RGB real footage (no IR/motion streams exist for these)
.venv/Scripts/python.exe camera/track_eval.py --clip data/clips/real_rgb_20190925_111757_1_5_masked \
    --conf 0.10 --no-coast --clutter --clutter-radius 90 --clutter-window 16 \
    --clutter-persist 6 --clutter-extent 40 --exempt-travel 0

# detector alone, two stages (ALWAYS pass --tag, see the landmine below)
.venv/Scripts/python.exe camera/evaluate.py run     --clip <clip> --mode full|sahi --tag <tag>
.venv/Scripts/python.exe camera/evaluate.py analyze --clip <clip> --mode full|sahi --tag <tag>

# retrain the track-level motion classifier (no GPU, cached detections only)
.venv/Scripts/python.exe scripts/train_motion_classifier.py \
    --clips data/clips/baseline data/clips/birds_only data/clips/backlit_birds \
            data/clips/train_terrain_canopy data/clips/train_terrain_mission \
    --holdout data/clips/eval_birds data/clips/terrain_ir_canopy data/clips/terrain_ir_birds \
    --out camera/motion_classifier_v3.json
```

**Train/eval split — do not violate.** Training clips: `baseline`,
`birds_only`, `train_terrain_canopy` (seed 6), `train_terrain_mission`
(seed 5), `train_zoom_a` (seed 9), `train_zoom_b` (seed 10). Evaluation
clips, never to be trained on: `eval_birds`, **`backlit_birds`**,
`terrain_ir_canopy` (seed 13), `terrain_ir_birds` (seed 42),
`terrain_ir_sweep`, `terrain_canopy`, `terrain_birds`, `terrain_zoomsweep`,
`terrain_baseline`, and all `real_rgb_*` sequences.

**`backlit_birds` is EVAL, not training.** Its own `meta.json` note says
`HELD-OUT EVAL: crossings + 10 birds seed ...`, and the clip name gives no
hint of it. The v2 and first-v3 classifiers in this session were trained with
it included — a real train/eval violation, caught and corrected the same
evening. **Check `meta.json`'s `note` field before adding any clip to a
training set**; the `train_` prefix is a convention, not a guarantee, and the
absence of one is not proof a clip is fair game.

## LONG-RANGE OPTICS + 15 Hz + m6 INSTALL (2026-08-29, CURRENT)

Four user questions drove this session: how passive radar would work and how
big it is; implement the telephoto; why not just raise the frame rate; is the
colour retrain done. All four are now measured rather than argued.

### The far-clip wall — check this before trusting ANY long-range number

`detection_world_terrain_ir.sdf` set both cameras' `<clip><far>` to **1000 m**.
Nothing beyond the far plane is rendered at all, so every past >1 km
measurement in this project was capped by the renderer, not by optics or by
the detector. Two new worlds move it to 2000 m and change nothing else:

- `simulator/worlds/detection_world_terrain_far.sdf` — 60 deg lens, far 2000
- `simulator/worlds/detection_world_terrain_tele.sdf` — **6 deg lens**, far 2000

Both keep world NAME `detection_world_terrain`, so the existing
`terrain_manifest.json` + `terrain_assets/heightmap.npy` still apply and the
terrain is byte-identical to the benchmark world. Captures MUST export
`STATION_HFOV` to match the SDF (0.10472 for the telephoto) or every
projected label is wrong.

### The telephoto works; the trained detector could not use it

`scripts/drive_scene.py --profile telesweep` flies 300 m → 1.35 km along the
camera's boresight (z = 2.5 + tan(0.15)·x), which is what a pan-tilt mount
does once cued. New clips, same flight, same seed, ONLY the lens differing:

| clip | lens | range | median target |
|---|---|---|---|
| `range_tele` | 6 deg | 303–1366 m | **5.50 px** |
| `range_tele_wide` | 60 deg | 555–1366 m | **0.40 px** |
| `range_tele_hr` | 6 deg, 15 Hz | as above | for TBD |
| `range_1km_hr` | 60 deg, 15 Hz | **19–1011 m**, 4106 vis | full-range primary |

Signal against local clutter (target dip ÷ background texture; >1 = stands out):

| range | 6 deg telephoto | 60 deg wide |
|---|---|---|
| 550–700 m | **4.51** | 0.08 |
| 700–850 m | **1.98** | 0.04 |
| 1000–1150 m | **1.40** | 0.04 |
| 1300–1400 m | **1.31** | 0.01 |

The telephoto holds the target above the clutter to 1400 m; the wide camera
has it 10–100x BELOW. A second, less obvious win: magnification also
*resolves the clutter* — terrain texture measured 8.7–9.5 DN in the telephoto
against 62–64 DN in the wide camera, because each rock feature is spread over
~11x11 px instead of sitting inside one pixel.

Despite that, the deployed YOLO scored **0% past 700 m** on the telephoto
clip. That is a training-data problem, not a sensor problem: the model has
never seen 6 deg imagery, so a 4 px blob on 11x-magnified rock is out of
distribution. **`scripts/point_target_detect.py`** is the right detector for
this channel — a point target has no shape to learn, so it uses the same
local-contrast logic the thermal channel already uses:

| range | YOLO | point detector | false alarms |
|---|---|---|---|
| 300–450 m | 32% | **88%** | 0.0/frame |
| 450–550 m | 26% | **95%** | 0.0/frame |
| 550–700 m | 4% | **46%** | 0.0/frame |
| >700 m | 0% | 0% | ≤0.6/frame |

Caveat to keep attached to these numbers: the telesweep line of sight grazes
the terrain, so the drone is against magnified **rock**, not sky. That is the
hard background (and realistic for a terrain-hugging FPV drone), but it is
not the easy case and should not be quoted as one.

### FIX PASS: two real bugs fixed, one alarm RETRACTED (2026-08-30, latest)

**1. Acoustic bearing assignment - REAL BUG, FIXED.** `fuse_bearings` looped
tracks in `(-hits, misses)` order, so the longest-lived track claimed the
nearest bearing and a stale clutter track could take the cue meant for the
target - the same flaw already fixed in `fuse_pcl`. Now shortest-link-first,
matching `update()`. Measured with `--acoustic`, vote-gated coverage:

| clip | before | after |
|---|---|---|
| terrain_ir_sweep | **0/400** | **111/400** |
| terrain_ir_canopy | 207/600 | **370/600** |
| eval_birds | 243/504 | **300/504** |

The sweep clip was at ZERO: the acoustic gate was starving the real target
because its cue kept going elsewhere. Alarms rise too (canopy 54.6 -> 92.6/min)
since more tracks now survive the gate; these are bare-default runs, so the
deployed-config numbers still need re-measuring.

**2. Multi-target identity - MY METRIC WAS WRONG, NOT THE TRACKER.** The
reported "3.8% of frames had one id covering two drones" counted every shared
id, but `is_hit` allows `max(12, 2*width)` px and all 37 events had the pair
**1.4-23.9 px apart (median 10.7)** - every one inside tolerance, so no swap
had occurred. Scoring only pairs separated by MORE than the tolerance gives
**2 frames (0.2%)**. `multi_target_eval.py` now reports the two categories
separately. The tracker was fine; the measurement was not.

**3. Wind - THE ALARM WAS OVERSTATED. Blanking is correct.** The channel-level
finding stands: under wind the motion detector goes from 65% to 2% on the
drone. But the FUSED system barely notices, and the fix I built made it worse:

| | fused coverage | alarms/min |
|---|---|---|
| no wind (control) | 539/990 = **54%** | 255 |
| wind, blank (as shipped) | 553/963 = **57%** | 298 |
| wind, truncate (my "fix") | 189/963 = **20%** | 146 |

~39 movers/frame spawn rival tracks that steal association from the drone's
own, so emitting the flood is far worse than emitting silence. **The original
comment - "emitting hundreds of movers is worse than emitting none" - was
right, and `blank` stays the default.** `--flood-mode truncate` is kept as a
DIAGNOSTIC (it makes a flood visible rather than silent), never for
deployment.

The static-clutter map was also tried on the wind case and does NOT work: at
its 60 px default it mutes everything including the drone (0% found); at
r20/e14 the drone drops to 19%; at r12/e8 it catches nothing (63%, identical
to no suppression). A swaying crown oscillates ~8-16 px at these ranges,
which is the same scale as the drone's own per-frame motion, so a
place-and-extent test cannot separate them. `--clutter` is kept and
documented as ineffective for wind. **A real vegetation fix needs temporal
frequency structure (foliage oscillates and returns; a drone does not), not a
spatial test - still open.**

**4. Rig geometry - I WAS WRONG that the mic array and radar were missing.**
`mic_array_link` and `radar_link` were already modelled as pose markers. What
genuinely did not exist and now does: `gimbal_link` + `tele_barrel` (the
pan-tilt head carrying the cued 6 deg telephoto) and `pcl_panel` +
`pcl_reference` (the PASSIVE radar receive panel, drawn at the 0.67 m the
LTE1800 physics implies, plus the reference Yagi). 6 links -> 10.

**Trap hit again:** writing the SDF with `Path.write_text()` and no encoding
corrupted its existing em-dashes through cp1252 and broke the XML. The
handoff already warns about this for Python files; it applies to EVERY file.
Recovered with `git checkout`, redone with an encoding-safe editor.

**Still owed:** the deployed-config re-benchmark. The model swap (m6 epoch5)
and now the acoustic fix have both moved bare-default numbers; nothing has
re-measured the >=90% headline configuration.

### WIND BREAKS THE MOTION CHANNEL + MULTI-TARGET (2026-08-30, latest)

**Every tree and bush in this project is `<static>true</static>` with no wind
plugin, so the motion channel had never once seen moving vegetation** - the
classic false-alarm source for a static-camera background model.
`drive_scene.py --sway N --wind-mps X` now spawns dynamic crown proxies at
REAL treetop positions from the terrain manifest and drives them with two
incommensurate frequencies (Gazebo will not move a `<static>` model, hence
proxies). `data/clips/sway_canopy` (40 crowns, 8 m/s) against
`data/clips/nosway_canopy` - the SAME seed, profile, birds and interval,
sway the only difference.

| | movers/frame | frames blanked | drone found | false alarms/min |
|---|---|---|---|---|
| no wind (control) | 4.3 | 1% | **65%** | 905 |
| wind, as shipped | 1.6 | **94%** | **2%** | 452 |
| wind, `--flood-limit 300` | 77.2 | 1% | 70% | **22,068** |

Wind produces ~77 movers/frame, 18x the control. The shipped flood guard
(`if len(dets) > flood_limit: dets = []`, MAX_CANDIDATES=40) then discards
the ENTIRE frame - a cliff, not a graceful degradation, so one mover past 40
takes the output from 41 detections to zero. **The channel does not flood
under wind; it silently switches itself off**, and the alarm rate IMPROVES
(905 -> 452/min) while detection collapses to 2%. Nothing in the metrics
flags it. Raising the limit restores detection (70%) at 22,068 alarms/min,
which is unusable. So both settings fail: this needs a real vegetation fix,
not a threshold.

Ruled out along the way, each by measurement: the scene-motion guard
(`--no-scene-motion` byte-identical output), the adaptive threshold
(`--noise-k` 3.0/4.5/6.0 all byte-identical - the fixed DIFF_THRESH
dominates), ego-motion (`_global_shift` returns exactly 0 on both clips), and
occlusion by the proxies (drone contrast 46-49 in BOTH clips, label offset
6-10 px). Four dead ends before the right one; check `dets = []` sites first
next time.

**Multi-target** (`scripts/multi_target_eval.py`, `data/clips/multi_drone`,
three drones + birds, 954/970 frames with all three in view):

| drone | detected | tracked | track ids | id switches |
|---|---|---|---|---|
| target_drone | 90% | 91% | 36 | 67 |
| target_drone_2 | 96% | 96% | 39 | 69 |
| target_drone_3 | 91% | 90% | 38 | 77 |
| *single-drone control* | *98%* | *97%* | *46* | *94* |

Detection and tracking hold at N=3. Fragmentation is NOT caused by the extra
drones - the single-drone canopy control fragments MORE per frame (94
switches / 600 frames = 0.157 vs ~0.073). What IS new: **one track id covered
two different drones in 3.8% of frames.** Recall looks fine and a hand-off
would point at the wrong aircraft.

**Trap fixed:** `EntityFactory` takes the model name from the SDF TEXT, not
from any argument, so spawning `target_drone/model.sdf` three times produced
three "spawn ok" lines and ONE entity - the extras never appeared on the pose
topic. Birds only work because `bird_sdf` bakes the name in. The extra drones
now get their model name rewritten in the text. The first capture silently
recorded a single-drone clip labelled as multi-drone.

**The rig** (`simulator/worlds/detection_world_inspect.sdf` +
`scripts/grab_inspect.py`): first ever picture of the station rather than the
view from it. It is a 1.2 m equipment box, a thin mast, and a camera body -
a placeholder. There is no modelled gimbal, no microphone array geometry, no
radar antenna, and the "cameras" are sensor origins on a link.

### THE CHAIN CLOSES: cue -> slew -> track -> range (2026-08-29, latest)

`data/clips/range_dual` is the first clip with TWO cameras on one mount
(`detection_world_terrain_dual.sdf`: wide 60 deg + telephoto 6 deg + thermal,
separate topics; `capture_dataset.py --zoom`). 3663 frames, both views
frame-synchronised, 150 kph, 308-1385 m, sky background.
`labels_zoom.jsonl` is derived by scaling the wide bbox about the principal
point - exact, since both cameras share a pose and sensor, and it avoids a
second projection path that could silently disagree with the first.

`scripts/cued_handoff.py` runs the whole chain with the slew charged
honestly (0.4 s slew+settle during which the telephoto is unavailable):

| | without the telephoto | **with the cued telephoto** |
|---|---|---|
| tracked | 10.8% | **99.6%** |
| tracked WITH a measured range | 0.0% | **99.0%** |
| cue -> ranged track | never | **0.17 s** |

Telephoto point detection on the zoom view: 98-100% in every bin to 1500 m.

**Two bugs, both found because a number looked wrong rather than because
anything errored.**

1. *The zoom was never commanded* - 0 looks in 2804 cue frames. The trigger
   fired only on cues `fuse_pcl` left UNMATCHED, but a cue matches ANY track
   within the gate and clutter tracks are always somewhere near. Matching a
   clutter track is not knowing what is out there. Trigger is now "no
   ESTABLISHED track (hits >= 3) at this bearing", which is the question the
   zoom answers.
2. *Range attached to the wrong track* - 99.6% tracked but only 8.3% carrying
   a range. `fuse_pcl` looped tracks in `(-hits, misses)` order, so the
   longest-lived track claimed the nearest cue first and a stale clutter
   track stole the range from the real target. Now assigns shortest-link
   first, as `update()` already did: 8.3% -> 99.0%.

**`fuse_bearings` (acoustic) still has flaw 2** - same seniority-ordered
loop. Left alone deliberately so a benchmarked number does not move silently;
fix it and re-measure the acoustic rows together.

### PCL FUSED INTO THE TRACKER + RANGE TABLE RERUN (2026-08-29, latest)

`CentroidTracker.fuse_pcl(cues)` folds passive-radar cues in after the camera
update, same sequential order `fuse_bearings` requires. The BEARING goes
through the identical 1-D Kalman update (H=[1 0 0 0]). The RANGE cannot: the
filter is image-plane and has no range dimension, so it is attached to the
matched track as `range_m` / `range_sigma_m` / `range_age`. That is the one
quantity no passive optical channel supplies and the one an effector
hand-off needs.

| range | tiled | thermal | motion | acoustic | **PCL** | ANY | TRACKED | RANGED | TELE |
|---|---|---|---|---|---|---|---|---|---|
| 0-30 m | 100% | 39% | 100% | 92% | 40% | 100% | 100% | 27% | - |
| 150-200 m | 92% | 25% | 90% | 57% | 84% | 100% | 96% | 66% | - |
| 200-300 m | 50% | 9% | 66% | 23% | 79% | 96% | 90% | 25% | - |
| 300-400 m | 15% | 2% | 25% | 0% | **76%** | 80% | 57% | 4% | 100% |
| 400-550 m | 7% | 1% | 5% | 0% | **80%** | 82% | 22% | 1% | 100% |
| 550-700 m | 1% | 2% | 0% | 0% | **79%** | 79% | 13% | 2% | 100% |
| 700-850 m | 1% | 1% | 0% | 0% | **77%** | 77% | 4% | 1% | 100% |
| 850-1000 m | 0% | 0% | 0% | 0% | **87%** | 87% | 0% | 0% | 100% |
| 1000-1150 m | 0% | 0% | 0% | 0% | 37% | 37% | 0% | 0% | 100% |

PCL carries the entire 300-1000 m band on its own: `ANY` goes from 33% to
80% at 300-400 m, from 3% to 79% at 550-700 m, and from 0% to 87% at
850-1000 m. Every camera channel is at 0-2% there.

**But look at RANGED: 0-4% beyond 300 m, and that is the finding.** A PCL cue
is deliberately never spawned as a track - it has no elevation, so it cannot
start one - and beyond 300 m no camera channel can spawn one either. So there
is no track object for the range to attach to, and PCL's best product (a
measured range, 3-12 m accurate) never reaches the track layer. PCL detects
at long range and simultaneously cannot be tracked at long range, using
cameras alone to start tracks.

**That is precisely the architecture the two sensors imply:** PCL cue (bearing
+ range, 300-1000 m) -> slew the 6 deg telephoto to that bearing -> the
telephoto detection spawns the track (measured 100% to 1.4 km against sky)
-> PCL range then attaches to it. Neither sensor closes the loop alone. The
current table cannot show that loop because the telephoto flew a separate
sortie, so TELE is still merged per range bin rather than frame by frame. A
single clip carrying BOTH a wide and a telephoto view is what would close it,
and does not exist yet - it needs two cameras in one world.

### CROSSING TEST CORRECTS THE GATE STORY + PCL BUILT (2026-08-29, latest)

**The tracker was never the bottleneck.** `data/clips/range_cross`
(`--profile crosssweep`, 150 kph, 15 Hz) finally stresses image motion
properly: median 12.3 px/frame, p90 41.3, p99 90.3, against the skysweep's
median 1.5. Raw hold rate looks bad - 70.7% overall, 22% in the 30-60
px/frame bin - and widening the gate does almost nothing (30 -> 60 -> 120
moves it 70.7% -> 71.3% -> 71.5%).

Attributing it properly: **given a detection on the target, the tracker holds
it 98.2% of the time - 97% even in the fast bins.** The 70.7% is the
DETECTOR missing frames (1976 detections across 2727 visible frames), not
association failing. `class_consistent` makes no difference either (98.2% vs
98.5%).

So the earlier recommendation - raise `base_gate_px` for the 150 kph
requirement - was **wrong in its reasoning**. The per-bin hold rates that
motivated it on `range_fast_sky` were detection-limited too. The change to 60
is KEPT, but on its own merit: it cuts clutter tracks (canopy 13 -> 9/min,
sky 11 -> 7/min) because a wider gate lets an existing track absorb a
detection instead of a rival spawning. It buys nothing for speed.

Why the crossing clip detects poorly is also a profile flaw, not physics:
+/-50 m of lateral travel at 410 m is +/-7 deg against a 3 deg half-FOV, so
whenever the target IS in frame at close range it is near the edge
(mean |x-640| = 341 px) where the background is no longer clean sky
(sigma 4.95 vs 0.00 mid-frame). Scale the lateral sweep with range before
quoting any crossing detection number.

### PASSIVE RADAR (PCL) IMPLEMENTED - `scripts/pcl_sim.py`

Built as a sensor model over ground truth, same discipline as `radar_sim.py`.
Bistatic range sum + bistatic Doppler from a chosen illuminator, with the
three failure modes that actually dominate: finite direct-signal
cancellation, the zero-Doppler ridge, and the baseline null.

**The absolute range scale is CALIBRATED, not derived, and that matters.**
Written as a from-scratch link budget the model predicted detection to 5 km
with 34 dB spare - contradicting every published micro-UAV field result,
because the residual-clutter term that really sets the floor is not
analytically tractable. `--anchor-km` (default 1.5 km for 0.01 m^2 at 80 dB
cancellation) pins the scale to published performance; the physics then
supplies only RELATIVE scaling across RCS, power, cancellation and geometry.

Calibrated envelope (DVB-T, 600 MHz, 8 MHz, 50 kW ERP, 20 km away):
detects to ~1.5 km, gone by 2 km. Cancellation is the make-or-break number -
at 1200 m: 60 dB -> -1.9 dB (no detection), 70 dB -> 7.0, 80 dB -> 11.9,
90 dB -> 13.0, 100 dB -> 13.1. **Above ~90 dB there is no further gain**, so
that is the engineering target, not "as much as possible".

On `range_fast_sky` (150 kph, 308-1386 m): **90.3% of dwells detected**, the
9.7% lost entirely to the zero-Doppler ridge. Per range bin 71-96%, with
**median range error 3.1 m at 300 m rising to 11.8 m at 1400 m** - and range
is information no passive optical channel provides at all.

**Verdict: worth building, for narrower reasons than first argued.** It is
NOT the thing that unlocks range - the telephoto already does that (100% to
1.4 km against sky). Its value is what no camera can do: it measures RANGE,
it has no revisit gap (decisive at 150 kph, where a 69 s raster revisit lets
the target close 2.9 km), it works in cloud/fog/night, and it still works
against a terrain-masked low flyer where the camera is background-limited to
~420 m. Its own blind spot is a target with near-zero bistatic Doppler,
which is exactly when a camera sees it best - the two fail in different
places, which is the whole point of putting them together.

### 150 kph TARGET: tracker and motion blur both hold (2026-08-29, latest)

User requirement: the threat flies 150 kph (41.7 m/s). New clip
`data/clips/range_fast_sky` - skysweep at `--speed-scale 3.0 --accel 15.0`
(verified max 42.0 m/s = 151 kph), 15 Hz, 4435 visible frames, 308-1386 m.

**Detection is unaffected**: the point detector holds 100% in every range bin
to 1386 m at 150 kph, identical to the 14 m/s clip. Detection is per-frame,
so speed cannot hurt it - only blur and tracking can.

**Tracking holds at 15 Hz** (`scripts/tracker_speed_test.py`): 99.8% held,
**6 handovers** across 4435 frames. The 2 Hz 14 m/s clip managed 18 handovers
in 600 frames - so CAPTURE RATE dominates target speed for track continuity,
which is the same lesson the TBD work produced.

The gate does strain where image motion is large. `base_gate_px=30` holds
only 40-50% in the 15-60 px/frame bins and 0% above 60; `base_gate_px=60`
recovers 30-120 px/frame. Worst case is a pure CROSSING target: 41.7 m/s at
500 m through the 6 deg lens is 1018 px/s = 68 px/frame at 15 Hz. The
skysweep is mostly radial (median 1.5 px/frame), so this clip under-stresses
the gate - a dedicated crossing profile would be the honest test.

Widening the gate is close to free, and helps clutter:

| clip | gate 30 | gate 60 |
|---|---|---|
| terrain_ir_canopy | held 94.2%, 13 clutter/min | held 91.2%, **9 clutter/min** |
| eval_birds | held 81.7%, 11 clutter/min | held **82.7%**, **7 clutter/min** |

A wider gate lets an existing track absorb a detection instead of spawning a
rival, so clutter tracks FALL. Recommend `base_gate_px=60`; canopy hold costs
3 points, everything else improves.

**Motion blur is a non-issue above 1/125 s** (`scripts/motion_blur.py`).
blur_px = (v/range) * focal * exposure; at 500 m through the telephoto that
is 17 px at 1/60 s and 1.0 px at 1/1000 s. Measured peak contrast falls
188 -> 116 -> 56 DN across 1/125 -> 1/60 -> 1/30. But detection stayed 100%
at every exposure, because **the rendered sky has noise sigma 0.00** - a flat
colour, so even a badly smeared target clears a 6-sigma threshold. Adding
realistic sensor noise is what makes the test mean anything: at sigma 6 DN,
1/60 s finally drops to 84% while 1/125 s and faster stay at 100%.

So: shoot 1/125 s or faster in daylight and blur never matters. At night the
exposure budget and the blur requirement collide - that is the real
constraint, and it is unmeasured.

### RASTER SCAN: the telephoto can find its own targets (`scripts/raster_scan.py`)

Once the telephoto measured 100% to 1366 m against sky, the open question
stopped being sensitivity and became SCHEDULING: a 6 deg beam sees 1% of the
wide field, so it finds a target only if it happens to be pointed there while
the target is in range. That is revisit time against closing speed, and it is
simulated directly - raster the beam over a search volume, check whether the
target was inside the footprint at that instant, roll against the MEASURED Pd
curve (sky and rock both interpolated from real clips, never assumed).

Baseline scan: 6 deg beam, 60 deg/s slew, 0.1 s settle, 0.2 s dwell (the
point detector needs one frame), 10% overlap, elevation 0-24 deg. Target
closes at 14 m/s from 1600 m, random bearing, unsynchronised scan phase,
400 trials.

| search sector | pointings | revisit | detected | median range | beyond 700 m |
|---|---|---|---|---|---|
| 60 deg | 88 | 34 s | **99.8%** | **978 m** | 99.8% |
| 90 deg | 136 | 53 s | 100% | 818 m | 66% |
| **120 deg (x3 = 360)** | 176 | 69 s | **99.5%** | **748 m** | 56% |
| 180 deg | 264 | 103 s | 79% | 609 m | 32% |
| 360 deg, one gimbal | 536 | 209 s | **39%** | 650 m | 19% |

Against ROCK the same 60 deg scan gives median 420 m and 0% beyond 700 m -
the camera is background-limited and no amount of scanning fixes that.

**Verdict on PCL vs raster.** For a known threat corridor the raster scan
makes PCL redundant for CUEING: 99.8% at a median of 978 m, using a gimbal
the design already needs. One gimbal trying to cover 360 deg fails (39%);
doubling slew and halving dwell only reaches 76%. THREE gimbals at 120 deg
each restore it (99.5%, median 748 m) for far less than a passive-radar
receiver chain. PCL's remaining, real value is what no camera can do:
all-weather (cloud/fog/night without thermal), simultaneous all-round
coverage with no revisit gap, and RANGE - the camera gives bearing only.
Decide PCL on those three, not on range extension.

**Trap:** clip-replay mode is only valid when the clip samples faster than
the dwell. The 2 Hz sky clip scored 0.2% detected against 99.8% for the same
scanner in the approach model, because a 0.2 s look opens and closes between
two 0.5 s frames. `run_clip` now warns when `interval_s > dwell`. Frame
coverage is anyway the wrong figure of merit for a scanning sensor (a beam
on 1 of 88 cells at 51% duty is in-beam 0.6% of frames by construction);
time-to-detection is the right one.

### SKY BACKGROUND CHANGES EVERYTHING (2026-08-29, later)

The user pushed back on the rock-background caveat: at a kilometre a drone is
usually high in the sky, not silhouetted on a hillside. Correct, and it is
the single biggest variable measured in this project.

`detection_world_terrain_tele_sky.sdf` pitches the camera to 13 deg (the
ridge tops out at ~10.2 deg elevation at 650 m, so this clears it) and
`drive_scene.py --profile skysweep` holds the target on that boresight.
`STATION_PITCH` is a new env override alongside `STATION_HFOV` - at 12212 px
focal length a 0.1 deg pitch mismatch is 21 px of label error, so the
projection MUST be told. Labels verified at 1.0 px median offset.

Same lens, same ranges, only the background differs (`data/clips/range_tele`
vs `range_tele_sky`):

| range | target | point det. SKY | point det. ROCK | YOLO SKY | YOLO ROCK |
|---|---|---|---|---|---|
| 300-450 m | 11.4 px | **100%** | 88% | 100% | 0% |
| 550-700 m | 7.2 px | **100%** | 47% | 98% | 0% |
| 700-850 m | 5.9 px | **100%** | 0% | 85% | 0% |
| 1000-1150 m | 4.4 px | **100%** | 0% | 51% | 0% |
| 1300-1500 m | 3.5 px | **100%** | 0% | 46% | 0% |

Target contrast against sky is 58-193 DN on a uniform 219 DN background;
against rock it is 12-40 DN on texture of ~9 DN. **The pixel threshold is a
property of the BACKGROUND, not of the lens.** Whole-system table with the
sky telephoto: 100% in every bin from 0 to 1500 m.

So the honest summary is two numbers, not one:
- **against sky: 100% to 1.4 km** (the limit of the captured clip, not of
  the sensor - the target is still 3.5 px and at full contrast)
- **against terrain: ~95% to 550 m, 47% to 700 m, then nothing**

### Bug: point-detector confidence was the CENTROID score, not the peak

`detect()` reported `score[centroid]`, but a blob need not contain its own
centroid (an annulus around a bright core does not), so the centroid pixel
could sit below threshold - measured at **-0.05 on a real detection**. Any
downstream `conf >= 0` filter then silently dropped a correct detection,
costing 3-12 points per range bin (sky 1300-1500 m read 81% instead of
100%). Confidence is now the blob's peak score. Caught only because the
detector's own table and the fused table disagreed - keep cross-checking two
independent paths to the same number.

### How many zoom levels are worth having? (`scripts/optics_ladder.py`)

| FOV | focal px | range @2.5 px | range @8 px | share of wide field | pointings to search it |
|---|---|---|---|---|---|
| 60 deg | 1109 | 151 m | 47 m | 100% | 1 |
| 24 deg | 3011 | 409 m | 128 m | 16% | 6 |
| 12 deg | 6089 | 828 m | 259 m | 4% | 25 |
| 6 deg | 12212 | 1661 m | 519 m | 1% | 100 |

A fixed ladder of staring cameras is the wrong trade. Three cameras
(60/24/6) cost **117% of one RTX 3060 Ti** at 15 Hz native inference and buy
no more reach than one steerable 6 deg lens; five cost 195%. A narrow lens
staring straight ahead covers 1% of the wide field, so it is not a search
sensor - it can only ever confirm. Wide at 15 Hz + ONE steerable 6 deg at
5 Hz costs 52% of the card and reaches the same distance over the whole sky.
The one defensible extra rung is a ~12-24 deg tier to bridge the cueing gap,
and the 24 deg thermal already occupies it.

### Video: H.264 is unavailable in this OpenCV build

`cv2.VideoWriter` reports `isOpened()` for avc1/H264/X264 but the OpenH264
DLL is absent, so it silently falls back and the MP4s are mp4v, which many
players refuse. Deliver GIFs (`Image.save(save_all=True)`) until ffmpeg or
openh264 is installed.

### WHOLE-SYSTEM DETECTION RATE vs RANGE (the headline number)

`scripts/system_range_table.py --clip data/clips/range_1km_hr --tele-clip
data/clips/range_tele`. Primary clip is the 15 Hz longsweep (19-1011 m,
4106 drone-visible frames, every passive stream rebuilt on m6 epoch5).

| range | wideRGB | tiled | thermal | motion | acoustic | ANY | TRACKED | TELE | **SYSTEM** |
|---|---|---|---|---|---|---|---|---|---|
| 0-30 m | 100% | 100% | 39% | 100% | 92% | 100% | 100% | - | **100%** |
| 30-60 m | 100% | 100% | 100% | 90% | 87% | 100% | 100% | - | **100%** |
| 60-100 m | 77% | 100% | 99% | 91% | 88% | 100% | 100% | - | **100%** |
| 100-150 m | 14% | 96% | 61% | 79% | 67% | 100% | 99% | - | **99%** |
| 150-200 m | 0% | 92% | 25% | 90% | 57% | 100% | 97% | - | **97%** |
| 200-300 m | 0% | 50% | 9% | 66% | 23% | 82% | 95% | - | **95%** |
| 300-400 m | 0% | 15% | 2% | 25% | 0% | 33% | 65% | 80% | **80%** |
| 400-550 m | 0% | 7% | 1% | 5% | 0% | 11% | 20% | 97% | **97%** |
| 550-700 m | 0% | 1% | 2% | 0% | 0% | 3% | 11% | 47% | **47%** |
| 700-850 m | 0% | 1% | 1% | 0% | 0% | 1% | 3% | 0% | **3%** |
| >850 m | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% | **0%** |

Read it as: **essentially perfect to 200 m, 95% to 300 m, 80-97% from
300-550 m on the telephoto, half the time at 550-700 m, blind past 850 m.**

Five caveats that must travel with this table:
1. **The telephoto flew a separate sortie** (different lens = different
   camera), so TELE is merged per range bin with max(), not OR-ed per frame.
   Conservative where they overlap.
2. **The telephoto is CUED and nothing can cue it past ~300 m.** Acoustic
   dies ~320 m, thermal ~200 m, wide RGB ~150 m, tiling ~300 m. Beyond that
   the 6 deg channel is a soda straw with nothing to aim it. Cueing, not
   sensitivity, is now the binding constraint — the PCL-shaped hole.
3. Background is magnified **rock**, the hard case, not sky.
4. No atmospheric attenuation is modelled; real long-range will be worse.
5. `thermal 39% at 0-30 m` is a FOV artefact, not a failure: the LWIR camera
   is 24 deg against the RGB's 60 deg, so a close high-elevation target falls
   outside the thermal frame entirely.

### m6 epoch5 regression A/B (cached m5 pass vs new, bare `fuse_eval` defaults)

| clip | coverage m5 -> ep5 | alarms/min m5 -> ep5 | clutter tracks |
|---|---|---|---|
| terrain_ir_canopy | 410 -> 364 /600 | 217.2 -> **188.4** | 954 -> 644 |
| terrain_ir_sweep | 359 -> 358 /400 | 22.8 -> **12.0** | 76 -> 40 |
| terrain_ir_birds | 492 -> 466 /573 | 122.4 -> **78.4** | 505 -> 330 |
| eval_birds | 361 -> 354 /504 | 77.0 -> **75.8** | 148 -> 181 |

epoch5 trades a few points of raw coverage for a large cut in false alarms
and clutter tracks (sweep -47%, birds -36%). Canopy detector-only recall at
equal frame counts (limit 400): m5 0.393 -> ep5 0.345 original, 0.370 ->
0.315 camo, white flat, camo NAMING 0.331 -> 0.548. **Still owed: the same
A/B under the full deployed config** (tiling + motion classifier + clutter
map + bearing fusion + zoom-confirm), which is where the >=90% headline
numbers come from. These bare-default numbers do not settle it.

### Trap: a "detection rate" measured with the alarm cap full is not one

Running the same point detector on the WIDE camera gave 0% at 550–700 m and
then **100% at 850–1000 m** — a rate RISING with range, which is impossible.
Cause: at 0.4 px the target is below the clutter, the 30-detection cap fills
with terrain texture, and one alarm lands inside the `is_hit` tolerance by
coincidence. `point_target_detect.py` now flags any bin above
`--alarm-budget` (default 3/frame) as CLUTTER-SATURATED and refuses to let it
be read as signal. `data/clips/range_1km/detections_point.jsonl` was deleted
for this reason; the point channel is **telephoto-only**.

### Bug fixed: `ir_detector.py` hardcoded the RGB focal length

`RGB_FX = 1108.5` was a module constant, but `ir_to_rgb` re-projects thermal
blobs into the visible frame by focal-length ratio. On a telephoto clip the
true fx is 12212, so every hot spot would have landed ~11x too close to the
principal point — silently, with no error. It now reads the clip's own
`meta.json` (`_adopt_clip_intrinsics`) and prints what it adopted.

### Frame rate: yes, just raise it — and it is a bigger lever than expected

The 15 Hz machinery already existed from the wingbeat campaign; the
long-range clip had simply never been captured that way. Beyond the
manoeuvre argument there is a harder one: measured image motion in the
telephoto clip is **8.5 px/frame at 2 Hz**. A velocity-bank search scales as
the square of per-frame motion, so 2 Hz needs ~961 hypotheses (3.5 GB,
unaffordable) where 15 Hz needs ~25. Raising the rate makes TBD roughly
**38x cheaper** as well as valid.

`scripts/tbd_detect.py --method dp` adds a Barniv-style dynamic program that
maximises over PATHS instead of assuming one constant velocity:
`I_k(x,v) = r_k(x) + max_{v' adj v} I_{k-1}(x-v, v')`. It handles manoeuvre
and represents sub-pixel speeds by alternating between neighbouring integer
velocities. The velocity-neighbour maximum is a separable 3x3 max over the
velocity grid, so each frame costs a few whole-array passes.

This matters because past 700 m the telephoto leaves the target at 1.4–2.0x
the background — just under the single-frame bar, where stacking ~16 frames
buys ~4x. That is the next experiment, on `range_tele_hr`.

### m6 colour retrain: install epoch 5, NOT best.pt

Training finished all 12 epochs. `best.pt` (= epoch 11/12) overtrained on the
repainted tiles: naming improved but recall fell. The checkpoint sweep found
epoch 5 dominates it (`eval_birds`, 400 frames):

| | m5 (was deployed) | **m6 epoch5 (installed)** | best.pt |
|---|---|---|---|
| original recall | 0.968 | **0.983** | 0.935 |
| white recall | 0.910 | 0.907 | 0.860 |
| **white named 'drone'** | **0.060** | **0.490** | 0.605 |
| skymatch recall | 0.370 | **0.430** | — |
| camo named 'drone' | 0.478 | **0.682** | — |

Installed 2026-08-29: `camera/weights/drone_bird_v1.pt` = m6 epoch5, md5
**f0df4f1703c0**. Previous m5 saved as `camera/weights/m5_backup_pre_m6ep5.pt`
(md5 7b7a0173278d). **Every untagged `detections_*.jsonl` predating the swap
is stale** — the sixth measurement error in this project. `range_1km` and
`range_tele` were rebuilt; anything else must be re-run before comparison.

Open item: on `terrain_ir_canopy` epoch5 measured 0.345 original recall at
limit 400 versus m5's 0.520 at limit 150 — **different frame counts, not
comparable**. Re-run both at the same limit before drawing any conclusion
about a canopy regression.

### Passive detection beyond 550 m — assessment

Ranked, after research and the measurements above:

1. **Passive coherent location (PCL)** — the real answer. Receives a
   third-party transmitter (DVB-T / LTE / 5G), emits nothing, and works
   against fibre-optic drones because it does not need the target to
   transmit. Antenna size is set by wavelength: FM needs a 12 m array,
   DVB-T ~2 m, **LTE1800 0.67 m, 5G n78 0.34 m** — and the higher bands also
   give far better range resolution (1.5–7.5 m vs 1000 m for FM). No access
   to the towers is required; it is a receiver. Honest caveat: the radar
   equation predicts >10 km, but the direct signal arrives **95 dB stronger
   than the echo**, and suppressing that — not thermal noise — sets real
   performance. Plan for 1–2 km on a micro-UAV, not the 2–8 km often quoted.
2. **The 6 deg telephoto** — built and measured above. Cheapest real range
   extension, but it is a soda straw and must be cued.
3. **TBD at 15 Hz** — free, unproven, next experiment.
4. Cooled MWIR with large optics; polarisation imaging (immature).

**The architectural gap this exposes:** the telephoto reaches ~550–700 m but
nothing else sees past ~300 m to point it (acoustic dies ~320 m, thermal
~200 m, wide RGB ~150 m). Cueing, not sensitivity, is now the binding
constraint — which is exactly the hole PCL fills.

### New files this session

- `scripts/point_target_detect.py` — local-contrast point detector + saturation guard
- `scripts/system_range_table.py` — whole-system detection rate vs range
- `scripts/run_telephoto_clips.sh`, `scripts/run_telephoto_hr_clip.sh` — captures
- `simulator/worlds/detection_world_terrain_{far,tele}.sdf`
- `scripts/drive_scene.py` — `telesweep` profile
- `scripts/tbd_detect.py` — `--method dp`
- `camera/ir_detector.py` — `_adopt_clip_intrinsics`
- `scripts/colour_robustness.py` — `--weights` for candidate checkpoints

## ARCHITECTURE SPEC — user's design intent, now authoritative (2026-08-29)

The user corrected the sensor employment model. This section OVERRIDES any
earlier assumption of a constantly-scanning radar:

1. **Passive-first.** The declaration chain is EO camera + IR camera +
   microphone array (Phase 3, still unbuilt), fused. The system must not
   emit while searching — it is meant to be undetectable/unlocatable.
2. **Radar is a CUED CONFIRMER only.** It turns on briefly, aimed at a
   track, ONLY after the passive stack is already confident it is a drone —
   a short dwell to double-check, then off. Emission time (duty cycle) is a
   first-class metric to report, not an afterthought.
3. **The threat is FIBRE-OPTIC drones: RF control-link sensing is USELESS**
   for this system. Do not propose or build RF sensing.
4. Phasing stands: camera (done) → microphone array (Phase 3, NEXT sensor
   to model) → radar (Phase 4, exists as scripts/radar_sim.py but must be
   employed cued, never continuous).

The constantly-on radar rows measured below remain valid as an UPPER BOUND
on what radar information can do; the cued-confirm implementation is the
operationally meaningful one.

## ROBUSTNESS AUDIT + BEARING FUSION (2026-08-29, earlier)

Four questions the user asked, all now measured rather than assumed.

**1. Microphone vs weather/noise — was NOT tested; now is.** The first model
had one crude `wind` scalar. `scripts/acoustic_sim.py` now carries eight
named environments with separate range / bearing-error / false-cue effects
and a spectral rationale each. Drone heard (sweep clip): still 91%, breeze
79%, leaves 73%, light rain 70%, traffic 66%, urban 61%, windy 53%, heavy
rain 36%. **Traffic/urban engine harmonics are the one false source that
genuinely mimics a multirotor** to a harmonic detector — birds never do.

**2. Bearing fused as a PROPER measurement** (`CentroidTracker.fuse_bearings`
/ `_update_bearing`): 1-D Kalman observation `H=[1 0 0 0]` — sharpens image
column and x-velocity, leaves the row untouched, claims no certainty the
sensor lacks. Verified (y bit-identical, x-variance shrinks). Unmatched
bearings are returned as SLEW CUES, never spawned as tracks. Support is
persistent per track (rolling fraction), because one bearing can only be
assigned to one track per frame — a per-frame requirement starved the target
(sky fell to 42%). Result: sky 446/504 @ 22.8 alarms/min (from 141),
vote-gated 409/504 @ **1.8/min**.

**3. Colour / camouflage — detection is colour-robust, NAMING is not**
(`scripts/colour_robustness.py`, repaints the airframe, geometry untouched):

| Repaint | Sky recall | named 'drone' | Canopy recall |
|---|---|---|---|
| original (black) | 0.983 | 0.788 | 0.517 |
| black | 0.992 | 0.546 | 0.367 |
| **white** | 0.892 | **0.037** | 0.417 |
| midgrey | 0.992 | 0.361 | 0.450 |
| camo | 0.992 | 0.395 | 0.450 |
| **sky-matched** | **0.350** | 0.000 | 0.175 |

So: it finds black, camo, grey and white drones. But a **white drone is
called 'bird' 96% of the time** — dangerous, because bird-mute would
suppress it. Fix = colour-augmented training tiles (repaint the airframe in
existing tiles, same trick as the blur augmentation that produced m5).
A deliberately **sky-matched** paint is the genuine optical weakness — and
precisely the case thermal/acoustic/radar exist to cover.

**4. Cloud — every sim number before this was measured against a
structureless sky** (`scripts/sky_conditions.py`). Composited, temporally
coherent, drifting: overcast 0.980 (harmless), bright_edge 0.941, stormy
0.683, **broken cumulus 0.440 detector / 0.623 with tracking**. And the
operational finding: **the motion channel floods under moving cloud**
(8,055 alarms/min) because a static-camera background model treats drifting
cloud as foreground. Disable it, or make it scene-motion-aware, in cloud.

**Queue:** colour-augmented retrain (white-drone naming); global-motion
compensation for the motion channel; thermal-equipped cloud clip to prove
the LWIR channel's cloud immunity; acoustic as soft cue rather than hard gate.

## COMPLETE SENSOR CHAIN — the user's full architecture (2026-08-29)

All four phases now exist as honest sensor models + fusion:

**HEAR → WATCH → LOOK CLOSER → LOCK**
1. **Acoustic (Phase 3, `scripts/acoustic_sim.py`)** — passive mic array,
   hears the multirotor harmonic comb, reports a BEARING. No bird problem
   (birds are silent). ~180 m range, cues where to look. `--acoustic` gates
   alarms on engine-sound corroboration: sky vote-gated 141 → 2.8/min.
   Clears 74-79% of camera false tracks. Bearing-only limit recorded.
2. **Wide EO + IR + motion (Phase 2, deployed m5+v10)** — detect & track,
   ≥90% every scenario, fully silent.
3. **Zoom camera (`--zoom-confirm`)** — silent look-closer; the ONLY
   identifier. Specks it can't resolve are re-looked on a cadence, NEVER
   handed to radar.
4. **Radar as FIRE-CONTROL (`--radar cued`)** — off until a drone is
   CONFIRMED, then paints it continuously (laser-hold: aim error p50 0.7-5
   px, ~100% within 25 px). **0% emission when no drone present (verified
   on birds_only).** `--radar-certainty 0.9`: a persistently unresolvable
   speck the passive stack is ≥90% sure of earns ONE brief ID dwell.

Full-chain flags: `--acoustic --zoom-confirm --radar cued` on top of each
scenario's per-scenario passive config.

**Remaining build/polish queue** (nothing blocks the architecture):
- Acoustic as a soft cue (early-warning slew + FP filter) rather than a hard
  coverage gate — wire it to LOWER other sensors' thresholds in its bearing
  and to mute no-engine-sound tracks, instead of gating coverage.
- Fuse acoustic bearing into the tracker as a real measurement (tighten the
  elevation band with array geometry if a 2nd mic node is modelled).
- Capture 15 Hz + acoustic on the same clips for a true multi-phase eval.

## FULL CONFIRM CHAIN (2026-08-29, earlier — superseded by the chain above)

The user's architecture is now implemented end to end:
**passive watch (EO+IR+motion, silent) → zoom look (silent, ~2.5 s) →
radar dwell (1.5 s, ONLY for zoom-unresolvable specks)**. Zoom verdict
stats measured from our own zoom clips (687/687 + 141/141 correct when
resolved; modelled 0.99). Verdict lifecycle: sticky per object, inherits
across duplicates (sustained-overlap gate), revoked only on overwhelming
camera contradiction (0.8 gate — 0.5 crushed canopy, measured).

| Scenario | Passive (spotted) | Verified tier | False alerts (events) | Radar on-air |
|---|---|---|---|---|
| Sky | 95.4% | 84.9% | **~0/min** | **3 s per 4 min (1.2%)** |
| Long-range | 93.75% | **92.25%** | 4.5/min | 1.5 s per 3.3 min (0.8%) |
| Terrain+birds | 93.5% | 88.5% | 7/min | 9 s per 5 min (3.0%) |
| Canopy (quiet) | 91.2% | 77.0% | 10/min | 12 s per 5 min (4.0%) |
| Canopy (thorough, no revocation) | 91.2% | 89.2% | 36/min | 2.5% |

Run flags for the chain: `--radar cued --radar-dwell 1.5 --radar-recue-s 20
--zoom-confirm` on top of each scenario's per-scenario config.

**Queue:** Phase 3 mic-array model (thins bird cues before the zoom is even
busy — the zoom camera is the new scarce resource at 50-60 looks/clip);
canopy verified-tier gap (its wide-camera votes are too noisy for clean
revocation — more naming work or acoustic disambiguation).

## USER CLARIFICATIONS + INHERITANCE (2026-08-29, earlier)

**Two user questions answered, recorded so the framing never regresses:**

1. **"Does this include the zoom mechanism?" — NO, and it must be built.**
   What exists: the detector is TRAINED on zoom-lens imagery and a zoom eval
   clip exists; the slew-to-cue concept was endorsed in the docs. What does
   NOT exist: an actual zoom-confirm stage. Per the user's passive-first
   architecture it belongs BEFORE radar in the confirm chain: passive
   suspicion → slew the zoom camera (silent) → detector on the zoomed view
   (a 6 px speck becomes ~30 px where naming is excellent — use OUR OWN
   measured naming-by-size stats to model it, methodology of radar_sim) →
   radar dwell only for zoom-ambiguous cases. This is the top build item.
2. **"73% is way too low" — that number is NOT the detection rate.** The
   passive tier detects/tracks the sky drone in 95.4% of frames regardless.
   73% is confirmed-track FRAME coverage, and the gap is bookkeeping
   (coasting frames + transient duplicate tracks), not missed drones:
   measured, the sky drone lives on ~4 long tracks, each confirmed ~2 s
   after birth — the OBJECT is verified for essentially its whole presence.
   Report per-frame AND object-level; never present the frame number alone.

**Verdict inheritance built** (death-morgue 40 px/5 s + confirmed-only
co-location at box-overlap scale). Two rules paid for with measurements:
denied NEVER propagates by proximity (at 90 px it spread bird-denials to
the drone: coverage 39%); propagation radius must be same-object scale (at
25 px canopy birds inherited the drone's confirmation: alarms 10→38 ev/min).

**Confirmed-tier frontier now (dwell 1.5 s, recue 20 s):**

| | Coverage (frames) | Alarms | Duty |
|---|---|---|---|
| Canopy, propagation ON | **90.8%** | 84/min (30.4 ev) | 25.5% |
| Canopy, propagation OFF | 82.0% | **26.6/min (10.2 ev)** | 30.5% |
| Sky | 73.0% frames (≈whole object) | **2.0/min (1.0 ev)** | 15.0% |

**Build queue:** (1) zoom-confirm stage (passive; lifts sky confirmed
coverage + slashes duty), (2) Phase 3 mic-array model (thins birds before
any cue), (3) only then radar as the final tiebreak — exactly the user's
original sensor ladder.

## CUED-RADAR RESULTS (2026-08-29, earlier — superseded numbers)

`fuse_eval --radar cued` implements the spec above. Two-tier output:

| Tier | Sky | Canopy | Radar duty |
|---|---|---|---|
| Passive (silent, internal) | 95.4% @ 141/min | 91.2% @ 177/min | 0% |
| **Radar-confirmed (operator alert)** | **73.0% @ 0.5/min (0 ev)** | **82.0% @ 26.6/min (10.2 ev)** | 16.3% / 30.5% |
| Canopy long-dwell variant (3 s) | — | 87.0% @ 30.4/min (10.6 ev) | 53% |

Confirm chain: passive declaration → dwell must RETURN (no echo = spurious,
never re-cue) → micro-Doppler 'bird' verdict = permanent deny → else
CONFIRMED, sticky for the track's lifetime. Cue gate = full passive opinion
(camera-named birds are not cued); young fragments may cue under the
young-tracks policy (without this, sky sat 20 pts lower).

**The two binding items, measured across four implementation iterations:**
1. **Confirmation cannot outlive a track id.** Fragmentation caps the
   confirmed tier's coverage (sky 73%, canopy 82-87% vs passive 95/91) and
   burns dwells (each fragment re-cues). CONFIRMATION INHERITANCE across
   fragments (successor track within gate of a dead confirmed track inherits
   its verdict) is the single highest-value integration task left.
2. **Duty cycle needs the acoustic layer.** Most dwells are first-checks on
   birds. Phase 3 (microphone array) sits BEFORE radar in the user's
   phasing precisely so the passive stack thins the bird population before
   anything emits. Build the acoustic sensor model next
   (multirotor harmonic comb vs bird; ~150-300 m envelope; bearing-only) the
   same way ir_detector/radar_sim were built.

## ALARM-PRICE ARC: research → wingbeat (marginal) → RADAR upper bound (2026-08-29, history)

The user flagged canopy's 177 and sky's 141 alarm-frames/min. Three moves:

**1. Events currency added** (`fuse_eval` prints `events/min`): an operator
acknowledges a track once, not per frame, and the frame currency scales with
capture rate. 177 → 62.4 events/min, 141 → 21.75. Both currencies always
printed; neither replaces the other.

**2. Optical wingbeat: built, measured, MARGINAL — recorded.** Four 15 Hz
clips captured (`hr_*`, seeds 31–34; `scripts/run_highrate_clips.sh` now
documents five WSL traps it survived), feature sets wb/wb2, classifiers
v11/v12. End-to-end A/B ≈ flat: at the ranges where alarms live, wingbeat
width-modulation is sub-pixel against box quantization. This is WHY fielded
systems read micro-motion with radar, not cameras.

**3. Micro-Doppler radar sensor model — the fielded answer, working.**
`scripts/radar_sim.py` (literature-parameterised, never an oracle: 0.93/0.92
per-dwell classification, noise, clutter) + `fuse_eval --radar` (OPT-IN —
the stream's mere presence changes every fused row; without the flag all
benchmarks stay exact) + rows radar-mute/rdr+mute/radar-gate:

| Benchmark | Before | With radar-gate |
|---|---|---|
| Sky | 95.4% @ 141/min (21.75 ev) | **89.9% @ 10.5/min (2.0 ev)** |
| Canopy | 91.2% @ 177/min (67.6 ev) | **84.8% @ 46.6/min (18.0 ev)** |

13× and 3.8× alarm cuts. The coverage dips under radar gating are
ASSOCIATION PLUMBING, not physics: young track fragments hold no accumulated
radar opinion yet (sky is 1 frame short; canopy ~5 pts). Next integration
tasks, in order: (a) let radar detections seed/merge into fragments faster
(tighter radar-to-track association, e.g. widen fuse merge tol for
radar-tagged boxes or carry radar opinion across re-spawns); (b) per-scenario
radar configs for terrain+birds/sweep (their non-gated rows flood when the
stream is on — radar-spawned displaced bird tracks need the clutter map or
radar-aware scoring); (c) then re-run the full matrix with --radar
everywhere and re-install decision.

**Real-world recommendation for the user** (they asked what real systems
have that we don't): a micro-Doppler radar (Robin ELVIRA-class) or RF
control-link sensor is the missing modality; the sim now demonstrates the
radar's effect end-to-end with honest error modelling. RF would be the
second sensor model to add (drones emit, birds don't; autonomous drones
evade it — model that).

## FINAL STATE — m5+v10 installed (2026-08-29, history below)

**Installed:** detector = **m5** (`camera/weights/drone_bird_v1.pt`, md5
`7b7a0173278d`; = m3 + 1320 motion-blurred real tiles, 10-epoch continuation)
and classifier = **v10** (v3 features on m5 votes). Backups: m3
(`m3_ep12_backup.pt`, `ff2e44b992e7`), m1 (`m1_ep24_backup.pt`,
`e794dc0a2855`), v7/v1 classifier JSONs alongside. All untagged caches match
the installed pair; tests 8/9 (known data gap).

**Final scoreboard (deployed defaults, per-scenario configs as below):**

| Scenario | Coverage | Alarms/min |
|---|---|---|
| Long-range sweep | **93.75%** | **6.6** |
| Terrain + birds | **93.5%** | 55.2 (89.4% @ 18.6 low-alarm) |
| Canopy | **91.2%** | 177 (**88.0% @ 74.6** balanced) |
| Sky | **95.4%** | 141 (**85.5% @ 10.8** balanced) |
| Real — easy / medium / hard | 100% / 94.4% / **65.5%** | 0 / ~58 / 14.4 |

**Why the loop stops here — the frontier, measured:** canopy's and sky's
coverage-first alarm prices are now REAL BIRDS correctly tracked (not naming
errors — m5 names birds 91-96% everywhere); discriminating them further at
2 Hz sampling is bounded by physics this suite already exploits. The real
hard clip's remaining misses sit beyond the detector's blur-augmented
ceiling (~0.72 at conf floor) — extreme blur + partial out-of-frame. Every
gate in the policy layer has been swept per-scenario. **What would move the
numbers now is new data**: ≥20 Hz captures (wingbeat/track features), more
real footage diversity, a third terrain seed, and a sky-with-birds training
capture. Design lesson that produced m5 after m4 failed: augment the
measured failure mode; never re-weight what works.

## CAMPAIGN TARGET MET (2026-08-29, history below — superseded numbers)

With the INSTALLED pair (m3 + v7) and per-scenario configs, fresh caches:

| Scenario | Coverage | Alarms/min | Config essentials |
|---|---|---|---|
| Long-range sweep | **94.25%** | 12.9 | and-confirm, thr 0.3, young=pass, mt=4 |
| Terrain + birds | **92.3%** | 49.6 | rgb-only, thr 0.5, coast=none, mt=0, young=pass |
| Terrain + birds (low-alarm) | 89.5% | **13.2** | same minus young=pass |
| Canopy | **90.2%** | 179 | or-fusion, thr 0.2, both, young=pass, coast=none, clutter, mt=4 |
| Canopy (balanced) | 81.7% | **74** | same, row and+mute, ir-persist 0.15 |
| Sky | **93.1%** | 114.5 | rgb-only, thr 0.05, young=pass, coast=none, clutter, **--no-class-consistent** |
| Sky (balanced) | 82.5% | **16.2** | same, vote-gated row, thr 0.1 |
| Real — easy | 96.8% | 0 | track_eval headline config |
| Real — medium | 93.2% | 57.6 | + `--vote 0.5` |
| Real — hard | 63.0% | 13 | ← the one remaining sore spot |

**The sky unlock (last structural find):** `class_consistent` association
protects terrain tracks from bird pollution but LOCKS OUT the target's own
detections when the detector misnames the TARGET — m3 flips the sky drone
to 'bird' in 25% of frames near birds, and those ≥8 px detections were then
barred from joining the drone-majority track. `--no-class-consistent` is
per-scenario: OFF on sky, ON on terrain (measured worse OFF there), moot on
the sweep. Same pattern as every gate in this system: the right setting is
scene-dependent, and both settings are principled.

**Still open, in value order:** (1) m4 retrain in flight — targets the
naming-driven alarm populations behind canopy's 179/min and sky's 114.5/min
coverage-first prices, and the real-hard recall; evaluate exactly as m3 was
(naming sweep incl. sky columns → tagged → v-next classifier → matrix →
install only if dominating). (2) The real hard clip (63%) is the last
sub-90 number anywhere. (3) If m4 disappoints, the alarm prices ARE the
remaining frontier — the coverage target is met.

## INSTALLED SYSTEM CHANGED (2026-08-28, history below this line)

**m3 + v7 is now the INSTALLED default**, per the user's decision to build on
the line with the best improvement potential:

- `camera/weights/drone_bird_v1.pt` = **m3** (b/epoch12). md5 prefix now
  **`ff2e44b992e7`**. The old m1 is preserved at
  `camera/weights/m1_ep24_backup.pt` (md5 `e794dc0a2855`).
- `camera/motion_classifier.json` = **v7** (v3 feature set, 12 weights).
  Old v1 preserved at `camera/motion_classifier_v1_backup.json`.
- Every untagged detection cache was regenerated with the new deployed model
  immediately after install (the discipline errors #4 and #6 bought). The
  `_m3`-tagged files from the pre-install evaluation are now redundant
  duplicates of the untagged ones.
- Any older section below saying "v1 is still installed" or quoting the m1
  hash as deployed is HISTORY as of this section.

**Post-install verification (all with the INSTALLED pair, no flags):**
tests 8/9 (same data gap); terrain+birds 85.9% @ 13.2 and sweep 89.75%
reproduce exactly from untagged caches. Real footage re-measured under m3
(track_eval headline config): easy 0.968 @ 0; **medium 0.932 @ 61 FP/min —
the campaign's original 262 FP/min problem, and this point dominates m1's
alarm-first row (0.908 @ 107)**; hard 0.609 @ 10.8 (m1: 0.702 @ 16 — m3's
recall trade bites on the smallest real targets; the checkpoint re-pick
below should watch this too).

**Improvement loop result #1 — canopy balanced row RECOVERED by gate
re-tune (no retraining):** the m3 coverage loss on the IR-confirmed rows was
a calibration artifact, not lost capability — m3's drone tracks carry lower
`ir_frac` ratios (changed track composition), so the old `--ir-persist 0.3`
cut them. At **`--ir-persist 0.15`** the installed system's balanced canopy
row is **81.3% @ 72.8/min — beating m1+v6r's 81.0% @ 77.4**. Cross-checked:
sweep and-confirm unchanged at 89.75% (alarms 1.8→2.1/min, harmless);
terrain+birds unaffected (its headline row doesn't use and-confirm). Use
ir-persist 0.15 with the installed pair on canopy. m1's ONLY remaining edge
anywhere is now the sweep's 3 frames (90.5% vs 89.75%).

**Improvement loop results #2-#4 — THREE SIM SCENARIOS NOW AT/ABOVE 90%
(2026-08-28, very late; all with the INSTALLED pair, config-level only):**

| Scenario | Config (installed m3+v7) | Coverage | Alarms/min |
|---|---|---|---|
| **Long-range sweep** | and-confirm, thr 0.3, `--young-tracks pass --min-travel 4` | **94.25%** | **12.9** |
| **Terrain + birds** | rgb-only, thr 0.5, `--coast-alarms none --min-travel 0 --young-tracks pass` | **92.3%** | 49.6 |
| Terrain + birds (low-alarm) | same minus young-tracks | 89.5% | **13.2** |
| **Canopy** | or-fusion, thr 0.2, `--rgb-mode both --young-tracks pass --coast-alarms none --clutter --min-travel 4` | **90.2%** | 179 |
| Canopy (balanced) | same config, row and+mute, `--ir-persist 0.15` | 81.7% | 74 |
| Sky (eval_birds) | rgb-only, thr 0.7, `--coast-alarms none` | 68.5% | 16.2 |

The three findings that did it, each measured per scenario:
1. **The travel gate is per-scenario, not global.** On terrain+birds it cost
   3.7 pts of drone coverage (hovering legs) while removing ZERO alarms —
   the clutter map + coast=none already cover static clutter there. mt=0 on
   t+b; mt=4 on canopy and the sweep (mt=0 EXPLODES the sweep: 372/min).
2. **young-tracks pass belongs on the sweep and canopy** (fragmented-track
   coverage; and-confirm's IR gate keeps it disciplined) and on t+b only if
   50/min is acceptable.
3. **Canopy threshold down to 0.2 is nearly free** (165→177 alarms/min for
   86.8→90.0%): the alarm population was already classifier-passed at 0.5,
   so the threshold only releases boundary drone frames.

Stage attributions under the installed pair (the map for what remains):
canopy ceiling 97.5%/tracker 96.0%; t+b 96.9%/91.8%; **sky 99.4%/97.2% with
the classifier eating 26.6 pts** — sky loss is pure drone-track bird-vote
contamination (m3 sky drone naming 0.745), the one place NAMING still binds.

**Improvement loop #5-#7 (same night, continued):**
- **Sky classifier dead ends measured** (v8 symmetric size-trust on votes:
  sky 0.773→0.571, the sky drone's 7 px votes are 3:1 correct; v9
  asymmetric: a wash — its bird-flips are on the LARGER frames). Sky's fix
  must be detector naming. The context trap: m3 names the sky drone 0.982
  ALONE but 0.745 NEAR BIRDS — any birdless sky clip is blind to the
  failure; only the original sky tiles (built from birds_drone on the Mac)
  contain the context.
- **Checkpoint re-pick: dead end** — extended sweep (m3_naming_sweep_v2)
  shows epoch12 already the best row.
- **m4 running**: m1 base + mixed_v4 (sky share 19%→29%, real 22%→25%,
  terrain/SAHI tiles unchanged), full batch. Evaluate with
  sweep_m3_naming --runs runs/experiments/m4_rebalance, then the tagged →
  fused pipeline as for m3. Aimed at: sky-near-birds naming, real-hard
  recall, keeping every m3 gain.
- **Real-footage motion channel: NEGATIVE, closed** (see commit ccec621) —
  132-240 movers/frame at thresh 35 even pan-gated; fusion degrades. Needs
  a redesign, not tuning. The cv2 blob pass (100x) and the ego-gating
  recipe survive for whoever attempts it.

**The improvement loop the user asked for (continue it):**
1. Re-pick the m3 checkpoint with SKY NAMING in the selection metric —
   `sweep_m3_naming.py` now scores `baseline` (sky drone naming) and
   `birds_only` (sky bird naming) alongside the sweep clip. 14 sibling
   epochs already exist; B/epoch6 probed sky drone naming 0.775 vs
   epoch12's 0.734. If another epoch dominates, install it the same way
   (backup → copy → hash → regenerate untagged → verify).
2. Recover the sweep's 3 frames (89.75% vs the MET 90.5% under m1+v6r) —
   likely the same re-pick, else a short m4 with half-weight bird tiles.
3. Attribute the canopy and-confirm coverage loss under m3 (81→66 on the
   balanced row; or-fusion is fine, so it is the IR-persistence interaction
   with m3's changed track composition).
4. Real-footage numbers under m3 need re-verifying (`track_eval` on the
   three masked clips) — the caches were refreshed but the headline
   0.960@178 was measured under m1.

## m3 RESULTS + the SIXTH measurement error (2026-08-28, history)

**This supersedes every sky (eval_birds/backlit_birds) number below it.**

While validating m3, a live-inference probe contradicted the cached naming
table and exposed the untagged-cache landmine AGAIN (error #6): the
`eval_birds`, `backlit_birds`, `baseline` and `birds_only` untagged caches
all predated the 13:13 deployed-model install. Every sky number this session
was computed on a phantom model, and the stale caches fed sky tracks into
the v5/v6 classifier training sets. All four caches regenerated; `v6r` =
v6's recipe retrained clean (better than v6 on canopy). **Rule, again, now
with teeth: `ls -la --time-style` the untagged cache against
`camera/weights/drone_bird_v1.pt`'s mtime BEFORE using it. Every time.**

**The real installed system (m1 + v1 classifier) scores 56.5% @ 18.8/min on
sky** — not the 86.1% @ 32.8 recorded earlier. The real m1 names the sky
drone 'drone' only 59.1% of frames.

**m3 detector** (`runs/experiments/m3_birdnamer_b/weights/epoch12.pt`,
selection table `runs/m3_naming_sweep.json`): canopy SAHI bird naming
0.795→0.896, SAHI drone recall 0.440→0.497, sky drone naming 0.591→0.745
(an IMPROVEMENT — "m3 broke sky" was the stale-cache artifact), zero
real-footage forgetting. **v7** = v3-feature classifier retrained on m3's
votes (`--rgb-tag m3`).

**FINAL two-system scoreboard (fresh caches everywhere):**

| Scenario | m1 + v6r | m3 + v7 | winner |
|---|---|---|---|
| Canopy or-fusion (coverage-first) | 87.3% @ 182 | 86.8% @ 165 | ~tie |
| Canopy and+mute (balanced) | **81.0% @ 77** | 66.0% @ 46 | m1+v6r |
| Terrain + birds | 83.9% @ 25.4 | **85.9% @ 13.2** | m3+v7 |
| Long-range sweep | **90.5% @ 29.4 — MET** | 89.75% @ 18.0 (misses by 3 frames) | m1+v6r |
| Sky | 62.9% @ 16.0 | **69.4% @ 21.5** | m3+v7 |

Neither dominates. m1+v6r holds the campaign's one met target (sweep) and
the best balanced canopy point; m3+v7 wins terrain+birds and sky outright.
The m3 canopy and-confirm/and+mute coverage loss (81→66) traces to m3's
lower canopy RGB recall changing track composition under the IR-persistence
gate — mechanism not fully attributed, see next steps.

**Next steps for whoever continues:**
1. A middle checkpoint may dominate both: `B/epoch6` probes sky drone naming
   0.775 (vs epoch12's 0.734) at sahiBird 0.795. Selection never scored sky
   naming — add `birds_only` (bird naming) + `baseline` (drone naming) to
   `sweep_m3_naming.py`'s metrics (they are training-pool sky clips; NEVER
   use eval_birds for selection) and re-pick.
2. The sweep's 3-frame miss under m3 and the canopy and-confirm loss both
   look like m3-recall side effects; a shorter retrain (fewer epochs, or
   half-weight on the new bird tiles) is the obvious knob.
3. Deploy decision is the USER'S: m1+v6r vs m3+v7 vs stay on m1+v1.

## m3 "bird-namer" detector retrain — original design notes (2026-08-28, late night)

**Motivation, measured:** the canopy balanced config's remaining alarms are
blocked from bird-mute by NAMING, not by the trust gate — 37/min of bird
alarms have median bird_vote 0.04 (the same birds systematically called
'drone'), only 1/min is blocked by the med_w<8 gate. Naming rates on eval
clips: canopy full-frame 93.2% of bird-hitting detections say 'bird', but
the **SAHI pass says only 79.5%** (it infers 320 px slices at 2x and the
model never trained on upscaled imagery); sky birds are worst at 17.5% but
sky is not the campaign gap. Bonus: bird-mute mutes anything the detector
CALLS bird, so clutter naming helps too.

**Design:** m3 = continued fine-tune FROM the deployed m1 weights on
`data/finetune/dataset_mixed_v3.yaml` = the entire m1 four-domain mix
(8668 tiles, unchanged — forgetting guard) + 1322 native tiles from
`train_ir_canopy` (new terrain, new birds) + 2133 **SAHI-scale tiles**
(`--tile 320 --upscale 2`, from train_terrain_canopy/mission +
train_ir_canopy). `train_ir_sweep` is deliberately absent from the new
tiles: it is the checkpoint-SELECTION clip (training-pool, never eval).

**Run state:** run A `runs/experiments/m3_birdnamer/` completed epochs 1–4
(epoch4.pt) at 2-core affinity, then was deliberately stopped and CONTINUED
as run B `runs/experiments/m3_birdnamer_b/` from A's last.pt — 15 epochs,
batch 8 / workers 2 / 4-core affinity (user approved 33% CPU while gaming;
BelowNormal priority throughout). The LR schedule restarts over 15 epochs,
which is fine: checkpoint selection is by measurement, not schedule. Sweep
checkpoints from BOTH directories (A epochs 1–4, B epochs 1–15). Logs:
`runs/m3_birdnamer.log`, `runs/m3_birdnamer_b.log`.

**Windows trap hit here, worth keeping:** the venv `python.exe` is a
launcher that SPAWNS the real interpreter as a child, so affinity/priority
set on the launched PID die with the launcher or miss the child entirely
(the child pre-dates the setting). Launch via `cmd /c start /belownormal
/affinity F ...` so creation-time attributes inherit, and verify with
Get-Process afterwards on ALL python PIDs.

**SWEEP RESULT (done):** winner `runs/experiments/m3_birdnamer_b/weights/epoch12.pt`.
On the selection clip: SAHI bird naming 0.783 → **0.872**, SAHI drone recall
0.440 → **0.497**, full-frame naming 0.995/0.911, real-footage guard 1.000
(no forgetting). Full table in `runs/m3_naming_sweep.json`. Tagged inference
(`--tag m3`) on the five eval-clip passes was launched next; end-to-end fused
numbers with `--rgb-tag m3` are the step after.

**When it finishes (the plan, so any agent can execute it):**
1. Pick a checkpoint: score epochs {4,8,12,16,20} on `train_ir_sweep`
   (drone recall + bird naming, both classes matter) with a guard check on
   one `realtrain_*` clip (real-domain recall must not sag). Use
   `scripts/sweep_checkpoints.py` conventions — centre-distance, never mAP.
2. Tagged inference on eval clips (`--tag m3` — NEVER untagged, the
   overwrite landmine): terrain_ir_canopy full+sahi, terrain_ir_birds full,
   terrain_ir_sweep full, eval_birds full.
3. Add `--rgb-tag` to fuse_eval (reads `detections_full_<tag>.jsonl` /
   `detections_sahi_<tag>.jsonl`) and re-run the canopy frontier configs
   with v6. Naming-rate table before/after.
4. Class votes feed the v6 classifier's `drone_evidence`/`bird_evidence` —
   if m3 shifts vote distributions much, retrain the classifier (v7) on m3
   detections over the training clips, then re-run the matrix.
5. Deploy decision stays with the user (m1 is still installed).

## AFTER v6 — where the campaign stands (2026-08-28, night; CURRENT)

**This is the current state. Read this and the CODEX VERDICT below it; the
sections after those are history and reasoning.**

`camera/motion_classifier_v6.json` = the v5 recipe retrained with the two new
thermal training clips (`train_ir_sweep`, `train_ir_canopy`) in the training
set. With the thermal domain finally represented, `ir_frac` learns a real
weight (+0.138, was structurally +0.000) and the "RGB sees it ⇒ drone"
shortcut washes out. **v1 is still the installed default; v6 is the model to
evaluate with** (`--classifier camera/motion_classifier_v6.json`).

**Classifier quality (AUC, drone vs bird track-windows, held out):**

| Clip | v1 live | v5 | **v6** |
|---|---|---|---|
| terrain_ir_canopy | 0.419 | 0.930 | **0.939** |
| terrain_ir_birds | 0.585 | 0.983 | **0.989** |
| eval_birds (sky) | 0.987 | 0.987 | 0.987 |
| backlit_birds | 0.875 | 0.869 | **0.739 ← v6's one regression** |
| sweep drone-windows passed @thr 0.3 | 0.992 | 0.797 | **1.000** |

**Campaign scoreboard with v6 (honest configs, alarms always attached):**

| Scenario | Config | Coverage | Alarms/min | 90% target |
|---|---|---|---|---|
| **Long-range sweep** | v6, or-fusion, thr 0.3, plain full RGB | **90.5%** | 29.4 | **MET** |
| **Long-range sweep** (conservative) | v6, and-confirm, thr 0.3 | **90.25%** | **4.5** | **MET** |
| Terrain + birds | v6, rgb-only, thr 0.5, `--coast-alarms none` | **83.9%** | **24.4** | short 6 pts; alarms cut 60% for 0.6 pts (coast=all: 84.5% @ 63–70) |
| **Canopy (coverage-first)** | v6, thr 0.5, `--rgb-mode both --young-tracks pass --coast-alarms none --clutter`, row `or-fusion` | **86.5%** | **177** | short 3.5 pts; was 68.8% @ 765 (v1) |
| Canopy (middle) | same config, row `vote-gated` | 85.2% | 142 | |
| **Canopy (balanced)** | same config, row `and+mute` | **80.2%** | **75** | alarm-matched frontier moved +30 pts coverage vs old 49.8% @ 71 |
| Sky (eval_birds) | v6, rgb-only, thr 0.9, `--coast-alarms none` | 86.9% | **5.0** | clutter alarms ZERO; v1 was 86.1% @ 32.8 |

**The coast-alarms discipline generalizes (measured):** terrain+birds
83.9% @ 24.4/min and sky 86.9% @ 5.0/min under `--coast-alarms none`, both
at ≤0.6 pts coverage cost. The ONE scenario where it must NOT be used is the
long-range sweep: its coverage genuinely lives in coasting frames (tracker
holds 97% against a 75.5% detection ceiling), and its alarms are already
4.5/min — keep `coast=all` there. The SAHI union likewise stays canopy-only:
on terrain+birds it adds alarms and no coverage (drone is RGB-visible 93%).

**Canopy alarm mechanics (2026-08-28, latest — how 237 became 75):** at the
84% operating point, 57% of alarm track-frames came from tracks NOT detected
on that frame (coasting bird/clutter tracks declaring up to 15 frames after
last sight), while the drone's coverage depended on coasting/young frames
almost not at all (20 and 7 frames of 504). New `--coast-alarms
all|judged|none` policy separates them: `judged` (young tracks must be
currently detected to declare) is FREE — identical coverage, 235→170/min.
`none` (every declaration needs a detection this frame) costs 3.3 pts for
another halving. Stack the `and+mute` policy row (thermal confirmation AND
appearance bird-mute — they cut different alarm populations: bird alarms are
94% RGB-fed real birds, clutter alarms are IR-co-located terrain FPs) and the
static clutter map for the 80.2% @ 75/min headline. Full-frame RGB collapses
to 44–49% in this config — the SAHI union is what feeds the young fragments
that thermal confirmation converts into coverage, so on canopy (only) SAHI
genuinely pays for itself. Default `--coast-alarms all` preserves every
historical number.

**Threshold caveat — be careful quoting one number:** v6's probability
calibration differs by scene. The sweep wants thr 0.3 (no birds → low thr is
nearly free), sky wants thr 0.9, terrain 0.5–0.8. Ranking (AUC) is fine
everywhere; it is the logistic calibration that shifted. A per-deployment
threshold is defensible (an installation knows whether it has a thermal
camera and what scene it watches), but say so when reporting.

**What remains, in order of value:**

1. **Canopy: coverage 80–84% at 75–172 alarms/min after the coast-alarms
   work** (see the mechanics note above; the "young fragments hold the
   coverage" hypothesis was MEASURED FALSE — only 7 of 504 covered frames
   were young-held, so sequence-level re-acquisition would add little
   coverage and is deprioritised). The remaining ~75/min at the balanced
   point are detected-now, classifier-passed, warm, travelling tracks the
   detector does not name bird — i.e. genuinely drone-like bird/clutter
   fragments. Remaining ideas, honestly ranked: (a) the last 96 uncovered
   frames sit mostly behind the and-confirm IR gate and detection gaps —
   check whether `or-fusion` + coast=none + clutter + mute beats and-confirm
   on the frontier; (b) richer appearance pressure only helps if the detector
   learns to NAME more birds (bird-mute then bites harder) — that is a
   detector-training idea with the usual caveats; (c) accept ~80%@75 as this
   sensor suite's canopy plateau and say so.
2. **backlit_birds AUC regression** (0.875 → 0.739): the thermal clips'
   1758 new bird windows shifted weights away from what separates backlit sky
   birds. Worth one experiment: class-weighting or a small backlit training
   capture. Do not trade the terrain/thermal wins away for it.
3. **Install decision** for v6 (user's call, as before): it beats v1
   everywhere except backlit_birds, including on v1's home turf (sky:
   86.9% @ 16.8 vs 86.1% @ 32.8, using thr 0.9).
4. Real-footage path is unchanged by all of this (v-classifiers don't run
   there): still 0.960 recall @ 178 FP/min via the clutter map.

## CODEX SESSION VERDICT (2026-08-28, night — audited by the next session)

Between the last Claude session and this note, a Codex session continued the
campaign and ran out of usage mid-task. Audit result, so nobody re-does or
blindly trusts it:

**What Codex did RIGHT (keep all of it):**
- **Captured the two thermal TRAINING clips** — the top task in this file.
  `data/clips/train_ir_sweep` (trajectory seed 17, 8 birds seed 211) and
  `data/clips/train_ir_canopy` (seed 19, birds seed 213), 600 frames each,
  RGB + IR + labels, in an **isolated generated world**
  (`detection_world_terrain_train_ir.sdf`, own assets + manifest) so the
  held-out eval world was never touched. Sweep label verification passed
  (median err 3.63 px, 90.3% within 15 px). Canopy verification checked 0
  labels — same as every canopy capture (the dark-object self-check only
  runs against clear sky), not a defect.
- Tooling: `scripts/run_thermal_train_clips.sh` (refuses to overwrite
  existing clips or a partial world), `--assets-dir/--manifest/--world-name`
  isolation in `gen_terrain_world.py`, occlusion-manifest args in
  `capture_dataset.py`. All sensible; committed by this session.
- Codex believed the canopy capture was still running when it died — it
  actually completed. Both clips are full 600/600.

**What Codex got HALF right — its claimed numbers, with the alarm column it
never printed** (all reproduce exactly; `--rgb-mode both --young-tracks pass`
+ v5):

| Codex claim | Reality (coverage @ alarms/min) | Verdict |
|---|---|---|
| "long-range 92.8%" | or-fusion 92.75% @ 45.9; **and-confirm 92.75% @ 6.0** | **REAL WIN — first scenario ≥90% with sane alarms** |
| "canopy 91.3%" | 91.3% @ **711 alarms/min** | empty-calorie: the alarm flood the user explicitly rejected |
| "terrain+birds 86.6%" | 86.4% @ **600 alarms/min** | same — honest config is still 84.5% @ 66 |

The sweep result is legitimate because `and-confirm` gates on IR persistence
directly, so passing young (<8-sample) tracks does not open the alarm gate —
the thermal confirmation still has to hold. On canopy/terrain+birds the same
relaxation just floods alarms.

**What Codex left unfinished** (done by this session, see below): the new
clips' derived streams (`train_ir_sweep` had only a tagged RGB pass —
copied to the untagged name the trainer reads; `train_ir_canopy` had none),
and the retrain that was the whole point of the captures.

## STATE OF PLAY — read this first (2026-08-28, end of evening session)

Everything below elaborates. If you read only one section, read this one.

> **SUPERSEDED (same night): the thermal training clips now EXIST** — the
> Codex session captured them (see CODEX SESSION VERDICT above) and the v6
> classifier below is trained on them. The text of this section is kept for
> the reasoning; its top task is done. The current top task list is in
> "AFTER v6 — where the campaign stands" further down.

**The single most important thing to do next: capture a THERMAL TRAINING
CLIP.** It is the one blocker that new analysis cannot remove. Every clip
that has a thermal stream (`terrain_ir_*`) is in the evaluation set, so the
track classifier has never seen a thermal-carried target. Two measurable
consequences: `ir_frac` learns weight exactly **+0.000** (constant in
training, so it standardises away — the model is structurally unable to use
thermal as evidence), and the model instead learns "RGB sees it ⇒ drone",
which is false precisely on the long-range sweep where RGB sees the drone in
only 36.7% of frames and thermal carries it. Use
`scripts/gen_terrain_world.py --thermal` then `scripts/drive_scene.py` with
new seeds (NOT 13, 42, or 0 — those are eval), and **read
`scripts/drive_scene.py`'s docstring first**; the sim-time clock trap
documented there has already cost this project real time.

**What is installed and live right now:** the m1-epoch-24 detector
(`camera/weights/drone_bird_v1.pt`, md5 prefix `e794dc0a2855`) and the **v1**
motion classifier (`camera/motion_classifier.json`). The user decided to leave
the newer classifiers uninstalled.

**Which classifier file to use.** All are selectable with
`fuse_eval --classifier <path>`:
- `motion_classifier.json` — **v1, live.** Keep for the long-range sweep.
- `motion_classifier_v5.json` — **best new model, use this one.** Clean split,
  zoom domain, size-gated shape features.
- `motion_classifier_v2/v3/v3clean/v4.json` — superseded iterations, kept only
  for the audit trail. v2 and v3 were trained with `backlit_birds` in the
  training set, which is an eval clip; their numbers are contaminated.

**Best measured numbers, corrected metric, as of this handoff:**

| Scenario | Coverage | Alarms/min | Config |
|---|---|---|---|
| Real — easy | 0.976–1.000 | 0 | `track_eval` |
| Real — medium (the 262 FP/min case) | **0.960** | **178** | `track_eval` + clutter map |
| Real — medium, alarm-first | 0.908 | 107 | + travel gate 150 px |
| Real — hard | 0.702–0.736 | 16–22 | |
| Sim: long-range sweep | **88.7%** | 12.3 | **v1**, or-fusion, thr 0.5 |
| Sim: terrain + birds | **84.5%** | **66** | **v5**, rgb-only, thr 0.5 |
| Sim: canopy (max coverage) | 61.0% | 168 | v5, or-fusion, thr 0.3 |
| Sim: canopy (alarm-matched vs v1) | 49.5% | **69** | v5, and-confirm, thr 0.3 |
| Sim: canopy (v1 for reference) | 68.8% | 765 | v1, or-fusion, thr 0.5 |

**No sim scenario meets the 90% target.** Nearest miss: the long-range sweep
at 88.7%. Canopy is far off and is the real problem.

**Classifier AUC (drone vs bird track-windows, held-out):**

| Clip | v1 live | v5 |
|---|---|---|
| canopy | **0.419** (below chance) | **0.930** |
| terrain+birds | 0.585 | **0.983** |
| eval_birds (sky) | 0.987 | 0.987 |
| backlit_birds | 0.875 | 0.869 |

**The four things this session established that change how to attack it:**

1. **Detection is not the bottleneck.** Some sensor sees the canopy drone in
   96% of frames and the tracker holds 93%. **Stop retraining the detector** —
   that is where the previous two campaigns spent their effort.
2. **The live motion classifier is anti-correlated on canopy** (AUC 0.419)
   because its features *invert* between domains: `hover_fraction`, its
   largest weight, scores 0.961 on sky and 0.355 on canopy. The v5 feature set
   fixes this and is the main asset produced this session.
3. **Every fused number recorded before this session was inflated** by a
   per-track-frame counting bug, now fixed. Do not compare against old figures.
4. **The remaining canopy gap is a bird-vs-drone problem, not a detection
   one**, and thermal does not solve it — the sim's birds are warm, which is
   physically correct, so `and-confirm` still raises 1692 bird alarms on v1.
   With v5 the canopy alarm rate is already 4.5x better; the missing coverage
   is the open question.

## Where the project stands

- **Phase 1 (simulator)**: done.
- **Phase 2 (camera detection)**: done, extensively measured, now in an
  **active improvement campaign** (this file). See below.
- **Phases 3–5 (audio, radar, fusion-as-a-phase)**: not started. The user
  redirected from "approve Phase 3" to "make the camera system as good as
  possible first" — treat that as still standing until the user says
  otherwise.

## THE CAMPAIGN — user's explicit target (given 2026-08-28, afternoon)

> ≥90% detection in **every** sim scenario with standard-lens + zoom-lens +
> thermal cameras **combined** (fusion counts, single sensors don't have to
> hit 90% alone). Real footage held as close to 90% as possible. The 260
> FP/min false-alarm rate on the real "medium" sequence needs serious
> improvement. The 46% (now corrected — see below) low-flight-vs-terrain
> case needs massive improvement. Long-range performance must not be worse
> than the real "hard" sequence (it isn't — see scoreboard).

The user also cares about **not wasting Claude usage** — they are budget-
conscious and switched to a cheaper model for this handoff. Favor decisive,
targeted experiments over broad research sweeps from here on; the research
groundwork (below) is already done.

### Deployed model — verify before trusting any number

`camera/weights/drone_bird_v1.pt` = **m1 epoch 24**, an ultralytics YOLOv11s
fine-tuned on FOUR domains at once (sim sky + sim terrain + sim zoom-lens +
real Anti-UAV footage, real tiles double-weighted). This is a different,
better model than the one described earlier in this file's history (below)
— that one (`e1b_ep40`, sim-only) **catastrophically forgot real imagery**
(8% real recall) and was replaced. Confirm what's installed before reading
any cached eval file:

```bash
python -c "import hashlib; print(hashlib.md5(open('camera/weights/drone_bird_v1.pt','rb').read()).hexdigest()[:12])"
# expect: ff2e44b992e7  (= m3, b/epoch12 — installed 2026-08-28 late)
# the older e794dc0a2855 = m1 epoch 24, now camera/weights/m1_ep24_backup.pt
```

**A landmine already hit once — check for it before trusting ANY untagged
eval file.** `camera/evaluate.py run/analyze --mode X` with no `--tag`
writes to `summary_<mode>.json` / `detections_<mode>.jsonl`, **overwriting**
whatever was there — including results from a *previous model*. Mid-campaign
a SAHI-vs-full comparison was run without first refreshing the untagged
full-frame baseline, silently comparing SAHI-with-the-new-model against
full-frame-with-the-OLD-model. It was caught and fixed (numbers below are
corrected), but the lesson stands: **before trusting an untagged
`summary_full.json`, either re-run it or check the file's mtime against
when the current model was installed.** Prefer `--tag` for anything you want
to compare later.

### Scoreboard (all held-out clips; corrected/verified 2026-08-28 evening)

**Detector alone**, max recall @ FP/min:

| Scenario | Recall | FP/min | vs target |
|---|---|---|---|
| Real footage — easy | 1.000 | 0 | met |
| Real footage — medium | 0.956 | **262** | recall met, FP/min far over |
| Real footage — hard | 0.766 | 35 | below 90%, but see fusion is N/A here (single RGB camera, no sim fusion available) |
| Sim: open ground | 0.940 | 0 | met |
| Sim: mixed terrain + birds, full-frame | 0.915 | 52 | close |
| Sim: mixed terrain + birds, **SAHI (2x tiled)** | 0.901 | 40 | close, fewer FPs |
| Sim: low-flight vs canopy/rock, full-frame | **0.458** | 69 | far below |
| Sim: low-flight vs canopy/rock, **SAHI (2x tiled)** | **0.750** | 142 | corrected number — SAHI genuinely helps, ~2x recall gain, ~2x FP cost |
| Sim: long-range (30-250m) zoom-lens sweep | 0.693 | 1 | below alone |

**Fused (RGB + thermal + motion channel, one shared tracker,
`camera/fuse_eval.py`)** — this is the number that matters for the 90%
target, since fusion is explicitly allowed.

> **CORRECTED 2026-08-28 (late).** Every fused coverage number recorded
> before this point was **inflated**. `fuse_eval.py` counted coverage per
> TRACK-FRAME against a denominator of FRAMES, so one frame with two tracks
> on the drone — routine once IR and motion join the same target — counted
> twice. Now fixed to count frames. **Do not compare against the old
> figures.** This is the fifth measurement error this project has shipped and
> caught; the standing rule held again (a surprisingly good result was the
> metric, not the system).

| Scenario | Fused coverage (or-fusion, corrected) | Alarms/min | vs 90% target |
|---|---|---|---|
| Long-range sweep (30-250m) | **88.7%** (was recorded 96.8%) | 12.3 | just short — NOT met |
| Terrain + birds | **79.6%** (was ~88%) | 446 | short |
| Low-flight vs canopy | **68.8%** (was 79.3%) | 765 | far short |

**Read this honestly: no sim scenario currently meets 90%.** The earlier
"long-range is MET" claim was an artefact of the counting bug.

**But the gap is not where the previous session thought.** Stage-by-stage
attribution (in phase2-results.md, "Where fused coverage is actually lost")
shows detection is NOT the bottleneck — some stream sees the canopy drone in
**96%** of frames and the tracker holds it in **93%**. The loss is the
**motion classifier**, which drops canopy from 92.3% to 68.8% on its own. It
is a logistic regression trained on clear-sky RGB-only tracks but applied to
fused, occlusion-fragmented terrain tracks: a train/serve mismatch. Fixing it
is the highest-value lever remaining, and it needs no GPU training.

### What's been tried this session, and the verdict on each

1. **SAHI (2x tiled inference) on terrain clips — WORKS, adopt it.**
   Canopy full-frame 0.458 → SAHI 0.750 recall (real, corrected comparison).
   Not yet wired into `fuse_eval.py` as a stream — **do this first**, it's
   free recall we're not yet using in the fused numbers above.
2. **Motion channel (`camera/motion_detector.py`) — WORKS, already fused.**
   Background-subtraction on the static station camera. Sees the canopy
   drone in 80.5% of frames (vs RGB's 46%) because it doesn't care about
   appearance, only movement. Feeds `fuse_eval.py` as a third stream
   already.
3. **OSD/reticle shortcut audit (`scripts/osd_mask.py`) — ran, negative
   result, useful.** Masked the real sequences' burned-in turret overlay and
   re-evaluated: numbers barely moved (real medium: 262 → 262 FP/min
   masked). **Confirmed the 260 FP/min is real foliage/building blur, not
   the model cheating off the overlay.** Don't re-investigate this angle.
4. **Hard-negative mining + retrain (`scripts/mine_hard_negatives.py`,
   "m2") — TRIED, NOT DEPLOYED, informative failure.** Mined 265 confirmed
   false-positive tiles from real training footage, tripled their weight,
   retrained 25 epochs (`m2_hardneg`, checkpoints still in the old session's
   scratchpad — see note below, may not persist). Result on the real medium
   sequence: cuts FP/min 262→7 but **also cuts recall 95.6%→86.8%**; on the
   hard sequence, recall 76.6%→60% for FP 35→1. **This is a trade along the
   precision/recall curve, not a Pareto improvement** — at every matched
   confidence threshold, m2 has both lower recall AND was still not
   reliably lower-FP than a correctly-thresholded m1. Interpretation: pure
   appearance-based negative pressure pushes the SAME ceiling this project
   already hit with the bird-flip problem — blurry real clutter and blurry
   real drones are not cleanly separable by single-frame appearance alone at
   these scales. **Do not just re-run this with a different weight and hope**
   without also trying step 5.
5. **Tracker association discipline
   (`CentroidTracker(class_consistent=True, suppress_spawn_near_coasting=True)`
   in `camera/tracking.py`, used by `fuse_eval.py`) — implemented, helps,
   not sufficient alone.** Stops a nearby bird's detections from
   contaminating the drone track's class-vote history, and stops duplicate
   track spawns next to a coasting track. Both off by default elsewhere
   (single-detector code paths unchanged; regression tests still 6/7, same
   known gap).

6. **SAHI wired into `fuse_eval.py` as a stream — DONE, mixed result, keep
   it as an accuracy mode.** `--rgb-mode full|sahi|both` (`both` = the
   deduplicated union). The standalone canopy win (0.458 → 0.750) largely
   **does not survive fusion**: the motion channel already sees that drone in
   83% of frames, so SAHI's extra detections are mostly redundant with recall
   fusion had recovered, while its extra clutter is new. Canopy or-fusion
   68.8% (`full`) → 80.5% (`both`) but alarms 765 → 868/min; on terrain+birds
   SAHI is a regression. It IS a clear win at a fixed alarm rate on canopy
   (vote-gated 41.5% → 55.0% at ~115 alarms/min) and on the sweep. 9x the
   compute (5.6 vs 45 FPS). **Don't make it the default.**
7. **Static-object suppression on the real-footage path — DONE, and the
   first Pareto win of this campaign.** Two findings, both non-obvious:
   - **The per-TRACK travel gate does not work on real footage.** Measured,
     clutter tracks travel FURTHER than the drone (984 px vs 754 px median).
     Root cause: `CentroidTracker`'s association gate is `30 + 4 × box width`
     — right for 2-13 px sim targets, absurd on a 175 px blur blob (a 730 px
     radius, a third of the frame), so unrelated clutter chained into one
     wandering track. Now capped via `max_size_gate_px` (default 250 px,
     a **measured no-op for the sim** — widest cached sim detection is 61 px,
     and canopy fusion output is byte-identical before/after).
   - **The evidence lives at the LOCATION, not the track.** 83% of the 218
     false positives sit in 22 fixed image spots (out-of-focus foreground
     branches, blurred building towers), and the drone passes near only 2 —
     but the tracker keeps dying and respawning there, so no track ever
     accumulates it. `camera/clutter_map.py` accumulates per location,
     strictly causally. A place is muted only if it keeps firing **AND**
     never spreads; persistence alone mutes the drone (an early version
     whose anchors drifted toward observations *followed* the target and
     collapsed recall to 0.46 — test 8/8 pins this).

   Real medium sequence, conf 0.10: **262 → 178 FP/min at 0.960 recall**
   (baseline 0.956). Push the travel gate to 150 px for 107 FP/min at 0.908.
   Cost elsewhere is small and real: easy 0.980 → 0.976, hard 0.719 → 0.702,
   no FP benefit on either (little clutter to suppress).
8. **Defocus/sharpness discriminator — measured, NOT adopted.** The FP crops
   are foreground foliage metres from a long lens, so heavily defocused;
   mean |Laplacian| ÷ local contrast keeps 91.7% of true detections and only
   38.5% of false ones on the medium sequence at threshold 0.06. But the cue
   **inverts** on the hard sequence (FPs sharper than the drone). Real signal,
   per-sequence calibration, not safe as a global default. Don't spend more
   on it before the classifier work below.

### The honest answer on "can training fix the 260 FP/min without losing
### recall" (the user asked this directly — keep this reasoning available)

Likely **not through more appearance-only retraining** — the hard-negative
experiment (item 4) is the second time in this project appearance-only
pressure has produced a recall/precision trade instead of a clean win (the
first was the entire bird-flip saga in the overnight campaign). The
principled fix, consistent with every win this project has actually banked,
is **track-level, not detector-level**: the foliage/building blobs causing
these false alarms are *static* — a real target, even hovering, has nonzero
net displacement over a few seconds; a stationary blur artifact has none.
`camera/fuse_eval.py` already has a `travel_px` gate for exactly this
reason (used in the sim IR/motion fusion). **The concrete next experiment**:
apply the same static-object suppression to the plain RGB-only real-footage
pipeline (there's no IR/motion stream for the real clips yet, but the travel
gate needs only the RGB detector's own track history — no new sensor
required). This should cut the foliage FPs at zero recall cost, unlike
retraining. Try this **before** another retraining attempt.

### Recommended next steps, in order

Steps 1 and 2 of the previous list are **done** (items 6 and 7 above). The
attribution table changed what matters, so this list is rewritten:

**DONE since this list was written** (see phase2-results.md for full numbers):

- Motion classifier retrained on fused tracks
  (`scripts/train_motion_classifier.py`, output
  `camera/motion_classifier_v2.json`). **Found the installed classifier is
  ANTI-correlated on canopy — AUC 0.419, below chance.** It prefers bird
  tracks to drone tracks there. Retraining fixes the sign (0.419 → 0.594
  canopy, 0.585 → 0.775 terrain+birds) at slight clear-sky cost
  (0.987 → 0.975). **v2 is NOT installed** — it is a different operating
  point, not a clean win: it takes the long-range sweep over 90% (90.25% at
  4.5 alarms/min, target MET) and gains 4 points on terrain+birds, but
  doubles clear-sky alarms (32.8 → 67.8/min) and is worse on canopy at
  moderate alarm rates. **Installing it is a user decision.**
- Travel gate re-tuned: it is already right. `--min-travel 0` costs 13x the
  clutter alarms on the sweep (98 → 1277/min) for identical coverage. Leave
  it at 8.
- `--young-tracks pass|drop` added: a track shorter than the classifier's
  8-sample minimum is currently dropped, not judged, which costs canopy ~2-4
  points of coverage. `pass` declares it instead (and costs alarms).

### Feature probe — WHY the classifier fails, feature by feature (2026-08-28, late)

**User decisions taken at this point:** (a) leave `motion_classifier_v2.json`
UNINSTALLED, keep v1 live and v2 selectable; (b) spend the remaining effort on
**building a better discriminator** for canopy, not on trading coverage
against alarms.

Single-feature AUC for separating drone track-windows from bird track-windows
(0.5 = chance; **below 0.5 means the feature is backwards on that clip**).
Held-out clips, fused tracks:

| Feature | canopy | terrain+birds | eval_birds (sky) | verdict |
|---|---|---|---|---|
| `drone_vote` (detector class votes over the track) | **0.820** | **0.978** | 0.614 | **best, and consistent** |
| `straightness` | 0.733 | 0.722 | 0.836 | good, consistent |
| `turn_rate` | 0.506 | 0.723 | 0.729 | useful off-sky |
| `width_mean` | 0.188 | 0.244 | 0.448 | strong, INVERTED (birds are bigger) |
| `rgb_frac` | 0.233 | 0.610 | 0.500 | inconsistent |
| `speed_in_widths` | 0.757 | 0.675 | **0.218** | **flips sign on sky** |
| `hover_fraction` | **0.355** | 0.541 | **0.961** | **flips sign on canopy** |
| `vertical_ratio` | 0.407 | 0.266 | **0.984** | sky-only |
| `y_norm` (image height) | 0.915 | 0.572 | 0.536 | **do NOT use — geometry overfit** |
| `support_rate` | 0.269 | 0.460 | 0.907 | sky-only |
| `speed_cv` | 0.603 | 0.663 | 0.947 | mild, consistent |

This explains the AUC 0.419 exactly. The installed classifier's two largest
weights are `hover_fraction` (+0.819) and `straightness` (+0.663).
`hover_fraction` scores **0.961 on sky and 0.355 on canopy** — it is the single
best feature in the domain it was trained on and backwards in the domain it is
applied to. `speed_in_widths` flips the other way. A model built from
sky-selected features cannot work on terrain; it inverts.

**Two conclusions that should drive the next work:**

1. **`drone_vote` is the strongest and most domain-stable signal available
   (0.820 / 0.978 / 0.614) and it is not a classifier feature at all** — it is
   used as a separate hard threshold in the `vote-gated` / `bird-mute`
   policies. Folding it into the classifier so the model can weigh it against
   the motion evidence is the highest-value single change left.
2. **Reject `y_norm` even though it scores 0.915 on canopy.** It is the drone's
   image height, i.e. it has memorised where this particular mission flies. It
   drops to 0.572 / 0.536 on the other clips. Using it would produce a great
   canopy number and a worthless system. Same caution applies to
   `support_rate` and `vertical_ratio` (sky-only).

### v3 classifier — BUILT, and it works (2026-08-28, late)

`camera/classify.py` now carries two feature sets. `MotionClassifier` records
which one it uses in its JSON (`feature_set`, defaulting to `v1` for old
files) and exposes `clf.features(history)`; `fuse_eval.py` goes through that,
so a model can never be fed the wrong vector. Train with
`scripts/train_motion_classifier.py --feature-set v3`.

**AUC, drone vs bird track-windows, held-out clips:**

| Clip | v1 (installed) | v2 (fused, v1 feats) | **v3 (domain-stable)** |
|---|---|---|---|
| terrain_ir_canopy | 0.419 | 0.594 | **0.922** |
| terrain_ir_birds | 0.585 | 0.775 | **0.978** |
| eval_birds (sky) | 0.987 | 0.975 | 0.958 |

Canopy goes from below-chance to strong. Learned weights are readable and
sensible: `drone_vote +2.119`, `dwell_fraction +1.607`, `width_cv -0.989`
(birds flap, so size varies — the wingbeat signature surviving as variance
even though the 2 Hz sampling cannot resolve the frequency itself),
`mv_frac +0.705`, `turn_rate +0.632`, `straightness +0.600`.

**End-to-end effect is a large cut in false alarms**, e.g. canopy or-fusion
765 → 187 alarms/min; terrain+birds `rgb-only` at thr 0.3 reaches 84.5%
coverage at 75 alarms/min, beating v1's 79.2% at 112 — better on both axes.

**KNOWN DEFECT in v3: the long-range sweep regressed** (v1 or-fusion 88.7% →
v3 ~70-74%). Two causes found; the first is fixed, the second is NOT and is
the top open task.

*Cause 1 (FIXED).* `drone_vote` was a ratio, and a track with no RGB
detections at all — most of the sweep, where RGB sees the drone in only 36.7%
of frames and thermal carries it — got a neutral 0.5, which in a linear model
is a constant push, not neutrality. Replaced by two evidence shares,
`drone_evidence` and `bird_evidence`, both computed over ALL samples, so an
aux-only track contributes exactly 0 to each. Matches this project's standing
convention that absence of an RGB opinion is not evidence against
(cf. `bird_vote is None` in `bird-mute`). This change alone did NOT recover
the sweep.

*Cause 2 (OPEN — do this next).* **The training set contains no long-range
and no thermal clip.** Training was `baseline` (40-78 m), `birds_only`,
`train_terrain_canopy`, `train_terrain_mission` — all short-range, all
RGB+motion only. Two consequences, both visible in the learned weights:
- `ir_frac` gets weight **+0.000**. It cannot be learned: no training clip
  has a thermal stream, so the feature is constant 0 and standardises away.
  The classifier is structurally unable to use thermal support as evidence,
  which is precisely what carries the long-range case.
- `dwell_fraction` (+1.58) and `width_cv` (−0.84) are fitted to *fragmented
  terrain* tracks. A smooth, slow, constant-velocity long-range track has low
  dwell and, at 2-3 px, high width jitter — so both features vote *bird* on
  exactly the target the sweep is made of.

**Fix:** add the long-range/small-target domain to training. `train_zoom_a`
(seed 9) and `train_zoom_b` (seed 10) are the dedicated training captures for
it; their detections are being generated as this was written
(`camera/evaluate.py run --mode full` then `camera/motion_detector.py run`,
~15 min per clip for the motion pass). For `ir_frac` to ever be usable, a
**thermal training clip must be captured** — none exists; every clip with a
thermal stream (`terrain_ir_*`) is in the evaluation set.

**Open next steps, in order:**

0. ~~Retrain on the clean, complete training set~~ — **DONE**, that is
   `motion_classifier_v5.json` (clean split + zoom clips + size-gated shape
   features). It did NOT fix the long-range sweep; see step 1 for why.
1. **Capture a thermal TRAINING clip** so `ir_frac` becomes learnable — the
   top open task, detailed at the top of this file. This is the only step
   here that requires the simulator (WSL2 + Gazebo) rather than pure analysis.
2. **Canopy coverage is the remaining blocker for the 90% target.** With v5
   the alarm rate is 4.5x better than v1, but coverage is 49-61% depending on
   policy against a tracker ceiling of 93%. Ideas not yet tried, roughly in
   order of expected value:
   - Let the classifier see the aux channels properly (needs step 1).
   - `--young-tracks pass` is worth ~2-4 points and already exists; measure it
     with v5 rather than v1 (all the young-track numbers on file are v1).
   - A track-SEQUENCE model rather than one verdict per window — the current
     design classifies each window independently and throws away the fact
     that a track is the same object over time.
   - Richer negatives: the training clips have 6-8 birds; the eval clips have
     the same. More varied bird behaviour would test whether v5's 0.93 canopy
     AUC is real or seed-specific. **Only two terrain training seeds exist
     (6 and 5), which is thin — treat v5's numbers as provisional until a
     third seed confirms them.**
3. ~~Build the v3 track classifier~~ — DONE. Original spec kept for
   reference: from the domain-stable features only:
   `drone_vote`, `straightness`, `turn_rate`, `width_mean`, `speed_cv`, plus
   `ir_frac`/`mv_frac` sensor support. Deliberately EXCLUDE `y_norm`,
   `support_rate`, `vertical_ratio`, and treat `hover_fraction` and
   `speed_in_widths` with suspicion (they invert between domains — if kept,
   they need a domain-robust formulation, not a raw value). Target: canopy AUC
   meaningfully above 0.594, without dropping `eval_birds` below ~0.97.
   Probe script that produced the table above:
   `scratchpad/feat_probe.py` (see the session's scratchpad; re-derivable from
   `scripts/train_motion_classifier.py` + the feature list here).
2. **Note the hard physical limit found while doing this**: bird wingbeat is
   2–8 Hz, the sim clips are captured at 0.5 s intervals = 2 Hz, Nyquist 1 Hz.
   **Wingbeat periodicity is not measurable in this data at all.** If you want
   that discriminator you must re-capture at ≥20 Hz. Do not spend time trying
   to extract it from the existing clips.
2. **Canopy needs a discriminator that does not exist yet.** Best measured
   canopy coverage is 87.3% and only at ~1300 alarms/min. Detection is fine
   (96% ceiling, 94.7% held by the tracker); the problem is that 6 birds and
   an occluded drone are not separable by these six motion features
   (AUC 0.594 even retrained), and thermal does not separate them either —
   `and-confirm` still raises 1692 bird alarms, i.e. the sim's birds are
   warm, which is physically correct. Candidates: richer track features
   (acceleration, periodicity/wingbeat spectrum, altitude-vs-terrain
   geometry), or a small learned track classifier instead of 6-feature
   logistic regression. **This is the remaining blocker for 90% everywhere.**
3. **The sweep's 75.5% per-frame detection ceiling** is the one place more
   detector recall would genuinely help (the tracker currently papers over it
   by coasting to 97%).
4. **Lowest priority**: hard-negative retraining with negatives weighted 1x
   (not 3x) and spot-checked first (some of the 265 mined tiles may be
   near-miss localizations on the real drone, contaminating the label).
5. **Re-run the full scoreboard** with `--tag` on every call, and update this
   file + phase2-results.md with final numbers before calling the campaign
   done. **Ask the user before declaring it done.**
6. Only then: return to the user's approved-but-parked Phase 3 (acoustic
   detection) question, or whatever they ask for next.

### Tooling reference (promoted into the repo this session — durable)

- `scripts/train_experiment.py` — run one fine-tune with kwarg overrides,
  save every epoch. Replaces the old `camera/finetune.py` for anything past
  the original baseline run.
- `scripts/sweep_checkpoints.py` — score every saved epoch against a
  held-out clip with the project's real metric (centre-distance recall, NOT
  ultralytics' mAP-based `best.pt`, which is measured noise at these target
  sizes). Always use this to pick a checkpoint, never trust `best.pt` blind.
- `camera/motion_detector.py` — static-camera background-subtraction stream.
- `camera/ir_detector.py` — classical thermal hot-spot detector (sim only).
- `camera/fuse_eval.py` — track-level fusion evaluator; policies `rgb-only`,
  `or-fusion`, `and-confirm`, `vote-gated`, `bird-mute`. This is where new
  fusion streams (SAHI, real-footage motion) should be wired in. New flags:
  `--rgb-mode full|sahi|both`, `--classifier <json>`, `--clutter`.
- `camera/track_eval.py` — plain-RGB track gating for the real footage (no
  IR, no motion channel): tracker + travel gate + static-clutter map, scored
  the same way `evaluate.py` scores detections so the numbers compare
  directly. `--stabilize` gates on ego-motion-compensated travel via phase
  correlation (the Anti-UAV turret is mostly locked, so it changes little
  there, but it is the right statistic if a clip does pan).
- `camera/clutter_map.py` — online, causal static-clutter location map. Mutes
  a place only if it keeps firing AND never spreads; both conditions matter.
- `scripts/train_motion_classifier.py` — retrain the track-level motion
  classifier on FUSED tracks from cached detections (no GPU). Writes to a
  `--out` path and does NOT install itself; verify first.
- `scripts/osd_mask.py` — auto-detect and neutralise burned-in overlay on
  real tracking-camera footage (temporal-std + gradient heuristic).
- `scripts/mine_hard_negatives.py` — harvest confirmed FPs from any clip as
  background training tiles.
- `scripts/import_antiuav.py` — convert Anti-UAV real sequences into this
  project's clip format (`--skip N` to import more without re-importing).
- `scripts/gen_terrain_world.py` — procedural field/forest/mountain world;
  `--thermal` adds the IR camera, `--rgb-hfov` builds a zoom-lens variant.
- `scripts/drive_scene.py` — PX4-free scripted drone+bird capture (WSL2
  gz-harmonic). Read its docstring before touching the sim-time clock logic
  — a subtle trap already cost real time once (250Hz clock subscription
  starves the Python gz-transport service requests; use the ~5Hz stats topic
  instead, already implemented).

### Data note — most derived tile sets are gitignored, not committed

`data/finetune_real/`, `data/finetune_zoom/`, `data/finetune_terrain/`,
`data/finetune_hardneg/`, `data/external/` (the 5.6GB Anti-UAV download) are
all in `.gitignore` — they're large and rebuildable. If a fresh clone is
missing them, re-run the relevant `scripts/import_antiuav.py` /
`scripts/build_finetune_dataset.py` / `scripts/mine_hard_negatives.py` calls
(exact commands are in this session's chat history if needed, or just re-run
the "Recommended next steps" above, which regenerates what's needed).

**Ephemeral, may not exist for you**: the previous session's `m2_hardneg`
checkpoints live under that session's OS temp directory, NOT in the repo.
If you need them, they may still be on disk at (Windows temp path shown in
git blame / prior chat), but treat this as best-effort — the checkpoint
itself was a documented dead end (item 4 above) and doesn't need recovering;
re-run `scripts/train_experiment.py` if you want to reproduce or extend it.

### Two GPUs available

The user has a second PC (RTX 2060 Super) that can be connected as a second
worker via Claude Code's cross-session messaging (`claude remote-control` on
the worker machine, `/remote-control` + `/list-agents` here). Useful for
running the evaluation battery on one machine while a retrain runs on the
3060 Ti here — ask the user if this is set up before assuming it isn't.

---

## Historical context (earlier campaigns, kept brief — full detail in phase2-results.md)

1. **Two-class (drone/bird) fine-tune, overnight campaign**: per-frame
   appearance cannot separate drone from bird at 4-6px; the fix was
   track-level fusion (all-class tracking + motion classifier), reaching
   0-0.5 alarms/min at 68-72% coverage on sim sky. See "The two-class
   fine-tune" in phase2-results.md.
2. **Terrain + EO/IR fusion campaign**: added a procedural field/forest/
   mountain world and a thermal camera. Found RGB collapses against
   terrain (later partly fixed — see below); thermal sees the canopy drone
   100% of the time; track-level fusion quadrupled long-range coverage.
   See "Terrain backgrounds and EO/IR fusion" in phase2-results.md.
3. **Real-footage validation**: the sim-only model catastrophically forgot
   real imagery (8% recall). A mixed four-domain retrain (sky+terrain+
   zoom+real) fixed it and is the currently deployed model (m1 epoch 24).
   Also retracted an earlier "terrain can't be trained" conclusion once a
   silent dataset-loading bug was found and fixed (ultralytics drops
   non-`./`-prefixed txt list entries as "corrupt" — **always use absolute
   paths in training-list txt files**, `scripts/train_experiment.py` and
   the dataset builders already do this correctly). See "Real footage, and
   the model that holds every domain" in phase2-results.md.
4. **This campaign (current)**: pushing every scenario toward 90% fused
   detection. See above.

### Standing engineering gotchas (still true, from every campaign)

- `evaluate.py analyze` counts only `cls=="drone"` detections as alarms.
- IoU is the wrong metric at these target sizes (2-13 px) for NMS, tracker
  association, AND evaluation — everything uses centre distance instead.
- Colour order differs per code path in `detector.py` (BGR predict/track,
  RGB SAHI) — see detection-notes.md.
- When a result looks surprisingly good OR bad, suspect the measurement
  first — this project has now shipped a mislabelled dataset, an aliased
  evaluation, a silently-dropped training set, AND a stale cached eval file
  (this session, item above). All are informative reading before assuming a
  model actually changed.
