# Handoff — continue this project on the PC

Written 2026-08-28 on the Mac, for a fresh Claude Code session (or human) on
the PC. The chat history and Claude's project memory do NOT transfer between
machines; this file plus the other docs carry the full state.

## Where the project stands

- **Phase 1 (simulator)**: done. PX4 SIH + Gazebo Harmonic + ROS 2; one-command
  start; 8/8 smoke checks. See README.
- **Phase 2 (camera)**: done and measured. YOLOv11s drone detector, custom
  centroid/Kalman tracker, SAHI accuracy mode, motion-based bird rejection.
  All numbers in [phase2-results.md](phase2-results.md); engineering traps in
  [detection-notes.md](detection-notes.md).
- **Phases 3–5 (audio, radar, fusion)**: not started. Phase order is strict;
  the user approves each phase before it begins.

## THE IN-FLIGHT TASK — COMPLETED on the PC, 2026-08-28

The two-class fine-tune ran as an 8-experiment overnight campaign on the PC
(RTX 3060 Ti). Outcome, in one paragraph: per-frame appearance cannot separate
drone from bird at 4-6 px (the misses are hard drone→bird class flips, and no
recipe — epochs, label-safe scale jitter, cls-loss weighting, per-class BCE,
P2 stride-4 head, drone oversampling, low mosaic — moved the frontier), but
the two-class model pays off at the TRACK level: with all-class tracking
(camera/tracking.py + classify.py + detect_live.py now track every class and
record per-frame class votes) plus the motion filter, the system reaches
**0-0.5 false alarms/min at 68-72% drone track-frame coverage** on the
held-out clip, versus the old system's floor of 9.5/min at 89% coverage.
Weights installed at `camera/weights/drone_bird_v1.pt` (DroneDetector picks
them up automatically); baseline recall stays 1.000. Full account with all
numbers and caveats: "The two-class fine-tune" section of
[phase2-results.md](phase2-results.md).

The original task instructions below are kept for reference (the commands
remain the way to reproduce the evaluation battery).

### Original task notes (historical)

Why this fine-tune: with birds in the air the current single-class detector
raises ~335 false alarms/min on the mixed clip (~660/min on birds-only
footage); the motion filter (`camera/classify.py`) cuts 84-91% of those but
31-53/min remain. The detector itself must learn what a bird is. Full
reasoning: "Birds and lighting" section of phase2-results.md.

All commands in this section are plain `python ...` and run on native Windows
(desktop app) as well as Linux/macOS — no WSL2 needed for this task. See the
native-Windows quickstart at the top of [pc-setup.md](pc-setup.md), especially
the CUDA-torch install trap.

Everything needed is IN THIS REPO:

- `data/finetune/` — the built training set (2,307 train tiles / 440 val
  tiles; provenance.json records which clips built it). Val split comes from
  `backlit_birds`, which therefore is NOT independent.
- `data/clips/eval_birds` — **strictly held-out** eval clip (mission flight +
  8 birds, bird seed 42, never used in training). The final judgement clip.
- `data/clips/backlit_birds` — second eval clip (crossings + 10 birds,
  seed 13, backlit sun) — used as training val, so quote it second.
- `data/clips/birds_only` — pure-bird footage (in training; sanity only).
- `data/clips/baseline` — drone-only clip for the no-regression check.

### Steps on the PC

```bash
# 1. train (≈15 min on the 3060 Ti; auto-selects cuda)
python camera/finetune.py --batch 32 --epochs 25
# -> installs camera/weights/drone_bird_v1.pt; DroneDetector picks it up
#    automatically from then on (default_weights() prefers it).

# 2. head-to-head on the held-out clip: NEW model...
python camera/evaluate.py run     --clip data/clips/eval_birds --mode full
python camera/evaluate.py analyze --clip data/clips/eval_birds --mode full
# ...vs OLD single-class model (downloads from HF, tagged separately):
python - <<'EOF'
from huggingface_hub import hf_hub_download
print(hf_hub_download("sapoepsilon/yolov11s-drone-detector", "best.pt"))
EOF
python camera/evaluate.py run --clip data/clips/eval_birds --mode full \
    --weights <path printed above> --tag old
python camera/evaluate.py analyze --clip data/clips/eval_birds --mode full --tag old

# 3. regression check (must stay ~1.000 recall):
python camera/evaluate.py run     --clip data/clips/baseline --mode full
python camera/evaluate.py analyze --clip data/clips/baseline --mode full

# 4. combined with the motion filter:
python camera/classify.py eval --clip data/clips/eval_birds

# 5. offline regression tests:
python tests/test_phase2.py
```

### What success looks like

- Drone recall on `baseline` and on `eval_birds` stays ≥ 0.99 at the chosen
  threshold (no regression from adding the bird class).
- Drone-class false alarms on birds ("FP/bird" column) collapse versus the
  old model's ~335/min — this is the whole point. Report before/after.
- The analyzer also prints how often the model *correctly identified* birds.
- Then re-run the motion filter on top and report the stacked result.
- Update phase2-results.md, TODO.md and this file with the outcome.

### Things that will bite you if forgotten

- `evaluate.py analyze` counts **only `cls=="drone"`** detections as alarms;
  detections jsonl now carries a `cls` field.
- `merge_close()` merges per class on purpose — do not "simplify" it.
- Colour order differs per code path (BGR for predict/track, RGB for SAHI) —
  see detection-notes.md before touching detector.py.
- IoU is the wrong metric at these target sizes, for NMS dedupe AND tracker
  association AND evaluation hits. Everything scale-sensitive uses centre
  distance. detection-notes.md explains each instance.
- The simulator start script expects miniconda at `~/miniconda3`, env
  `dronesim`, and a `PX4-Autopilot` symlink in the repo root
  ([pc-setup.md](pc-setup.md) covers the full PC install).
- When results look surprisingly good OR bad, suspect the measurement first:
  this project already shipped one mislabelled dataset and one aliased
  evaluation. Both post-mortems are in detection-notes.md.

## Terrain + EO/IR fusion campaign (added later on 2026-08-28)

Same-day follow-up on user direction: a procedural field/forest world and a
station thermal camera now exist (`scripts/gen_terrain_world.py`, PX4-free
capture via `scripts/drive_scene.py` + WSL2 gz-harmonic — see the scripts'
docstrings for the traps: stats-topic clock, batched set_pose_vector,
occlusion-aware labels). Measured: RGB collapses against terrain backgrounds
(contrast-limited — a terrain retrain failed and was not adopted); the
thermal channel sees the drone 100% of frames there; track-level EO/IR fusion
(`camera/fuse_eval.py`) quadruples long-range coverage at half the alarms.
Full numbers and caveats: "Terrain backgrounds and EO/IR fusion" in
[phase2-results.md](phase2-results.md). Next milestone (user-approved):
real-footage validation on the Anti-UAV paired RGB+IR dataset (TODO.md).

## After the fine-tune (updated 2026-08-28)

Done: weights + docs + pipeline changes committed. The user approved moving to
**Phase 3 (acoustic detection)** once the fine-tune campaign completed — that
approval was given the night of 2026-08-27→28, conditional on the campaign
finishing all its steps, which it did.

Known follow-ups (not blocking Phase 3, noted in TODO.md):

1. Capture one fresh never-evaluated clip for final quoted numbers
   (eval_birds served as the selection dev set overnight).
2. Consider retraining the motion classifier on two-class/all-class tracks.
3. Confidence calibration so reported confidence reads as a probability.
4. On the PC, optionally switch to full physics-in-Gazebo mode
   (`make px4_sitl gz_x500`, then `--mode gz`) — needs the WSL2 sim install
   (see pc-setup.md); the fine-tune/eval work needed none of it.
5. Note: `runs/detect/...` in git still holds the Mac's stale interrupted
   training artifacts (restored, not touched) — worth `git rm`-ing in a
   housekeeping commit if the user agrees.
