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
`birds_only`, `backlit_birds`, `train_terrain_canopy` (seed 6),
`train_terrain_mission` (seed 5), `train_zoom_a/b`. Evaluation clips, never
to be trained on: `eval_birds`, `terrain_ir_canopy` (seed 13),
`terrain_ir_birds` (seed 42), `terrain_ir_sweep`, `terrain_canopy`,
`terrain_birds`, `terrain_zoomsweep`, and all `real_rgb_*` sequences.

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
# expect: e794dc0a2855  (= m1 epoch 24)
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

**Open next steps, in order:**

1. **Build the v3 track classifier** from the domain-stable features only:
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
