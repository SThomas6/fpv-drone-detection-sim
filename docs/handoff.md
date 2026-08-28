# Handoff — continue this project in a fresh session

Written 2026-08-28, end of a multi-day session, specifically so a **new**
Claude Code session (new context window, possibly a cheaper model) can pick
up the fusion-improvement campaign without re-deriving anything above. Read
this file, then [phase2-results.md](phase2-results.md) for full measured
numbers, then start work.

**First message to give the new session:** "Read docs/handoff.md and
continue the fusion campaign toward the 90% target."

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
target, since fusion is explicitly allowed:

| Scenario | Best fused coverage | Alarms/min | vs 90% target |
|---|---|---|---|
| Long-range sweep (30-250m) | **96.8%** (`or-fusion`) or 92.3%@4.5/min (`and-confirm`) | 4.5–12.3 | **MET** |
| Terrain + birds | **88.1%** (`bird-mute` policy) | 148 | just short; alarms still high |
| Low-flight vs canopy | **87.7%** coverage achievable (`or-fusion`/`and-confirm`) | 130–440 | coverage close; **alarms far too high** |

**Read this honestly**: coverage is basically within reach everywhere.
**The remaining gap is almost entirely false alarms, not missed detections.**
The bird/clutter alarm rate on the two hard clips (canopy, terrain+birds) is
the single blocking problem for hitting 90% cleanly.

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

1. **Wire SAHI into `fuse_eval.py`** as an additional/replacement RGB stream
   for terrain scenarios (it's proven standalone — item 1 above — just not
   in the fused pipeline yet). Re-measure canopy and terrain+birds fused
   coverage; expect a real jump toward 90%.
2. **Add the travel-gate / static-object suppression to a plain-RGB
   evaluation path** (adapt the logic already in `fuse_eval.py`'s
   `travel_px` check) and test on `real_rgb_20190925_111757_1_5_masked`
   (the 262 FP/min sequence). This is the highest-confidence fix for the
   real-footage alarm problem and costs no GPU training time.
3. **If (1)+(2) still leave a gap**, revisit hard-negative retraining but
   with the negatives weighted 1x (not 3x) and manually spot-checked first
   (a handful of the 265 mined tiles should be eyeballed — some may be
   near-miss localizations on the real drone itself, contaminating the
   "negative" label).
4. **Re-run the full scoreboard** (both tables above) with `--tag` on every
   call this time, and update this file + phase2-results.md with final
   numbers before calling the campaign done.
5. Only then: return to the user's approved-but-parked Phase 3 (acoustic
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
  fusion streams (SAHI, real-footage motion) should be wired in.
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
