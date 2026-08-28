# TODO

## Phase 1 — Simulator setup (COMPLETE 2026-08-25 — awaiting approval for Phase 2)
- [x] Inspect machine (Apple Silicon, macOS 26.3.1, 8 GB RAM, ~13 GB free disk)
- [x] Research known-good PX4 + Gazebo + ROS 2 path for Apple Silicon macOS
- [x] Install simulation stack: conda env `dronesim` (ROS 2 Jazzy + Gazebo
      Harmonic 8.10 + gz python + MAVSDK via robostack/conda-forge), lean brew
      toolchain (cmake/ninja/ccache)
- [x] Build PX4 SITL: v1.18.0-beta2 SIH target (`px4_sitl_sih`); repo moved to
      ~/Projects/PX4-Autopilot (build system can't handle the space in this
      project's path; symlinked back)
- [x] World: `simulator/worlds/detection_world.sdf` — open field + station
      (camera 720p@15Hz, mic/radar mount links)
- [x] Target drone flies autonomously (MAVSDK offboard pattern 30-80 m from
      station, 8-30 m alt); SIH pose mirrored into Gazebo at 30 Hz
- [x] Sensor data verified in Python: gz-transport direct AND via ros_gz bridge
      to ROS 2 (`/detection_station/camera/image`, ground-truth pose topic)
- [x] One-command start script + stop script
- [x] Smoke test: `tests/test_phase1.py` — 8/8 checks pass (topics, camera via
      gz + ROS, PX4 telemetry, real movement in both PX4 and Gazebo truth)
- [x] README.md (install, architecture, run instructions, limitations)
- [ ] OPTIONAL later: full physics-in-Gazebo mode (`px4_sitl gz_x500` built
      against conda gz libs) — SIH mode is sufficient for Phase 2 work

## Phase 2 — Camera detection (COMPLETE 2026-08-27 — awaiting approval for Phase 3)
- [x] **Plan amendment (2026-08-19):** don't default to a generic detector — research
      the best available pretrained model for detecting SMALL FPV drones
      (tiny pixel footprint at range) and use that.
- [x] Model selection (2026-08-25): **sapoepsilon/yolov11s-drone-detector + SAHI
      tiling + ByteTrack** (primary), Hibou RT-DETR drone fine-tune (backup) —
      full rationale, license notes, fine-tune plan and test plan in
      docs/model-selection.md
- [x] Detection node: image in → detected yes/no, bbox, confidence, centre px
      (`camera/detector.py`, live CLI `camera/detect_live.py`)
- [x] Tracking (IDs, velocity) between frames — `camera/tracking.py`. Wrote a
      centroid/Kalman tracker: the drone moves further than its own box width
      in 27% of frames, so every IoU-based tracker (incl. ByteTrack) fragments
      identity. Ours: 1 ID, 0 switches vs ByteTrack's 14 IDs, 24 switches
- [x] Labelled dataset tooling with ground truth free from the simulator, and
      self-verification of labels against rendered pixels
- [x] Test matrix: range sweep 8-250 m, lateral crossings, terrain background,
      empty-sky false-positive baseline — full results in docs/phase2-results.md
- [x] SAHI measured (+3 pts recall at >200 m, 9.5x slower) — kept as opt-in
      accuracy mode, not the default
- [x] Regression test `tests/test_phase2.py` (7 checks, all passing)
- [ ] NOT DONE, deliberately: fine-tuning. Unnecessary for synthetic imagery
      (100% recall to ~200 m out of the box). Still expected to be needed for
      real footage — plan retained in docs/model-selection.md
- [x] Lighting variation (`simulator/lighting.py`, 5 presets) — measured, and it
      does NOT hurt: 100% recall backlit and at dusk
- [x] Birds added (`simulator/birds.py`) and measured — the detector calls birds
      drones ~660 times/min, and confidence cannot separate them (birds reach
      0.85, the drone never exceeds 0.74)
- [x] Bird rejection by track MOTION (`camera/classify.py`) instead of a second
      appearance model: removes 84-91% of bird alarms on held-out data, costs
      ~30% of drone track-frames. Improvement, not a fix
- [x] Fine-tune a TWO-CLASS (drone, bird) detector (2026-08-28, on the PC).
      8-run overnight campaign; every recipe axis probed. Verdict: per-frame
      appearance saturates at 4-6 px (drone↔bird hard class flips), but the
      two-class model + ALL-CLASS tracking + motion filter drops the system's
      achievable false-alarm floor ~20x (0-0.5 alarms/min at 68-72% drone
      track-frame coverage vs old floor 9.5/min at 89%). Weights installed
      (`camera/weights/drone_bird_v1.pt`), baseline recall 1.000 kept.
      Full account: "The two-class fine-tune" in docs/phase2-results.md
- [ ] Capture ONE fresh never-evaluated clip and quote final numbers on it
      (eval_birds was used for checkpoint/threshold selection = dev set)
- [x] Terrain backgrounds + EO/IR fusion (2026-08-28, same-day campaign):
      field/forest world + thermal camera + track-level fusion. Headline:
      fusion quadruples long-range coverage at half the alarms; RGB canopy
      blindness is information-limited (retrain failed, weights unchanged).
      Full account: "Terrain backgrounds and EO/IR fusion" in
      docs/phase2-results.md
- [x] Real-data validation (2026-08-28 evening): Anti-UAV300 downloaded; the
      sim-trained detector had CATASTROPHICALLY FORGOTTEN real imagery
      (recall 0.08 vs the base model's 0.99 on the same sequences). Fixed with
      a mixed four-domain retrain (sky+terrain+zoom+real) — deployed weights
      now score 0.77-1.00 on real sequences AND 3.7x the old terrain/zoom
      recall, at ~6pts sim-sky cost. En route, found + fixed a silent dataset
      bug (ultralytics drops non-"./" txt entries as corrupt) that had
      invalidated the earlier "terrain retrain failed" conclusion. Full
      account: "Real footage" section of docs/phase2-results.md
- [ ] Real thermal/fusion validation: port the IR detector to contrast-based
      detection for Anti-UAV's 8-bit unregistered IR videos, then test the
      fusion layer on real paired sequences
- [ ] Real bird pressure: Anti-UAV has no bird labels; find/label real
      drone-vs-bird footage (Drone-vs-Bird challenge data, registration)
- [ ] Narrow-FOV RGB ablation (mandatory before attributing the IR range gain
      to thermal physics rather than the 24° optics)
- [ ] Track gap-filling interpolation + track-sequence classifier (the
      Drone-vs-Bird winners' increments) — the lever for the newly-diagnosed
      bird-track fragmentation problem over terrain
- [ ] Consider retraining camera/motion_classifier.json with the two-class
      detector + all-class tracking (current weights were fitted on old-model
      tracks; they transfer but were not refitted)
- [ ] NOT DONE: multi-target tracking, real-world validation

## Phase 3 — Acoustic detection (NOT STARTED)
- [ ] Audio pipeline: signal → spectrogram → features → drone yes/no
- [ ] Survey drone-audio datasets (e.g. DroneAudioDataset, DREGON) — Gazebo does not
      simulate audio, so this will be synthesized/replayed audio driven by sim state
- [ ] Investigate direction-of-arrival feasibility

## Phase 4 — Radar (NOT STARTED)
- [ ] Simulated radar: range / bearing / elevation / radial velocity + noise model
- [ ] Validate against ground-truth pose from simulator

## Phase 5 — Sensor fusion (NOT STARTED)
- [ ] Association: do camera/audio/radar observations match one physical target?
- [ ] Fused track output with per-sensor + overall confidence

## Known gaps after Phase 2
- Results are from a uniform, noise-free synthetic sky and are therefore
  optimistic. A 1.7 px target at 220 m would not survive a real camera.
- Bird/drone discrimination is the biggest known weakness. UPDATE 2026-08-28:
  the two-class detector + all-class tracking + motion now reaches 0-0.5
  alarms/min at 68-72% drone track-frame coverage (see phase2-results.md);
  the residual weakness is coverage, not alarms, and it is a measured
  trade-off rather than an open problem.
- The motion classifier is fitted to SIMULATED bird flight. The mechanism is
  real; the specific decision boundary is not evidence about real birds.
- Only ever one drone in frame; multi-target association untested.
- No cloud, haze or sensor noise: the synthetic sky is unrealistically clean.

## Known constraints / risks
- Disk: only ~13 GB free before install. Keep installs lean; PX4 clone must be shallow.
- RAM: 8 GB — prefer running Gazebo headless (server only) except when visually debugging.
- macOS is Tier-3 for most of this stack; fallback paths documented in README as they are decided.
