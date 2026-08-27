# Moving to the PC (16 GB RAM, RTX 3060 Ti)

Short version: **yes, switch.** This project has been running on macOS in hard
mode — half the engineering effort so far went into working around things that
are first-class on Linux. The PC is better on every axis that matters:

| | M2 Air (8 GB) | PC (3060 Ti, 16 GB) |
|---|---|---|
| Fine-tune speed | ~1.3 s/iter → ~2 h/run | CUDA, expect ~8-10x faster → ~15 min/run |
| Inference | 13–17 FPS | comfortably 40+ FPS, SAHI becomes usable live |
| RAM headroom | forced headless, 720p camera | GUI + higher-res camera fine |
| PX4 + Gazebo + ROS 2 | community-supported (SIH workaround) | **first-class platform** — full physics-in-Gazebo mode (`gz_x500`) works natively |

## Fastest start: native Windows, no WSL2 (fine-tune + evaluation only)

The current task — training the two-class model and evaluating it — is pure
Python and runs natively on Windows with the desktop app. The SIMULATOR is the
only part that needs WSL2, and that can wait until sim work resumes.

In the Claude Code desktop app (or PowerShell), after installing Git and
Python 3.12 from winget/python.org:

```text
git clone <REPO_URL> fpv-drone-detection-sim
cd fpv-drone-detection-sim
python -m venv .venv
.venv\Scriptsctivate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install ultralytics sahi huggingface_hub pillow numpy "lap>=0.5.12"
python -c "import torch; print(torch.cuda.is_available())"
```

**The torch line matters**: on Windows, plain `pip install torch` (or letting
ultralytics pull it) installs a CPU-ONLY build and training silently crawls.
Install from the cu126 index first, as above, and confirm the last command
prints `True` before training. (`lap` is optional — only ultralytics' own
tracker needs it; skip it if it fails to install.)

Then train and evaluate exactly per [handoff.md](handoff.md):

```text
python camera/finetune.py --batch 32 --epochs 25
```

Note: the `.sh` scripts in `scripts/` are bash and will NOT run on native
Windows — but the whole fine-tune/eval task uses only `python ...` commands,
which are cross-platform. When you later want the simulator on this PC, come
back to the WSL2 section below.

## Simulator setup: WSL2 Ubuntu 24.04

Keeps Windows intact; NVIDIA CUDA passes straight through into WSL2 with any
recent Windows driver (no CUDA install inside WSL needed for PyTorch — its pip
wheels bundle it). Dual-boot Ubuntu is marginally better for Gazebo GUI
performance but WSL2 (with WSLg for GUI) is fine, and the project runs
headless by default anyway.

On Windows (PowerShell, once):

```text
wsl --install -d Ubuntu-24.04
```

Everything below happens inside the Ubuntu shell.

## 0. Train the model FIRST (no simulator needed)

The fine-tune dataset and the held-out evaluation clips are committed to the
repo, so the highest-value task runs immediately after cloning — before any of
the simulator stack below is installed. You only need the conda env's Python
side (or even a plain venv with `pip install ultralytics sahi huggingface_hub
pillow "lap>=0.5.12" numpy`):

```bash
python camera/finetune.py --batch 32 --epochs 25   # ~15 min on the 3060 Ti
```

Then follow the evaluation steps in [handoff.md](handoff.md).

## 1. Base tools

```bash
sudo apt update && sudo apt install -y git cmake ninja-build ccache build-essential
```

## 2. Clone this repo

```bash
git clone <REPO_URL> ~/fpv-drone-system && cd ~/fpv-drone-system
```

## 3. Conda environment (identical to the Mac one)

The scripts assume miniconda at `~/miniconda3` and an env named `dronesim`,
so keep those names:

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p ~/miniconda3
source ~/miniconda3/etc/profile.d/conda.sh

conda create -y -n dronesim --override-channels --strict-channel-priority \
  -c robostack-jazzy -c conda-forge \
  python=3.12 ros-jazzy-ros-base ros-jazzy-ros-gz-sim ros-jazzy-ros-gz-bridge \
  ros-jazzy-ros-gz-image gz-transport13-python gz-msgs10-python gz-tools2
conda activate dronesim
pip install mavsdk ultralytics sahi huggingface_hub pillow "lap>=0.5.12"
python -c "import torch; print('cuda:', torch.cuda.is_available())"   # expect True
```

(The pip torch that ultralytics pulls on Linux is a CUDA build. All detection
code auto-selects `cuda` when available — nothing to configure. The macOS
libomp symlink workaround in docs/detection-notes.md does NOT apply on Linux.)

## 4. PX4

Same version, same layout as on the Mac — cloned OUTSIDE the project with a
symlink in (scripts resolve the symlink):

```bash
git clone --depth 1 --branch v1.18.0-beta2 --recurse-submodules --shallow-submodules \
  https://github.com/PX4/PX4-Autopilot.git ~/PX4-Autopilot
ln -s ~/PX4-Autopilot ~/fpv-drone-system/PX4-Autopilot
cd ~/PX4-Autopilot
python3 -m venv .venv && ./.venv/bin/pip install future -r Tools/setup/requirements.txt
source .venv/bin/activate && make px4_sitl_sih
```

(On Ubuntu you can alternatively run `./Tools/setup/ubuntu.sh` first — it is
the maintained first-class setup script, unlike the macOS one.)

## 5. Run it

```bash
cd ~/fpv-drone-system
./scripts/start_sim.sh --birds 8
conda activate dronesim && python tests/test_phase1.py
python camera/detect_live.py --classify
```

## 6. Regenerate the rest of the data (not in git)

Recorded clips, the fine-tune dataset and trained weights are git-ignored.
Either copy `data/` and `camera/weights/` across manually, or regenerate:

```bash
./scripts/run_test_matrix.sh          # standard clips
./scripts/run_hard_matrix.sh          # birds + lighting clips
python camera/build_finetune_dataset.py --train ... --val ...   # see provenance.json
python camera/finetune.py --batch 32  # 3060 Ti: raise batch from the Mac's 8
python camera/classify.py train --drone data/clips/baseline data/clips/crossing \
                                --bird data/clips/birds_only
```

Training was interrupted on the Mac (thermals), so the weights are NOT in the
repo yet — the PC trains them (step 0 above) and should commit them:
`git add -f camera/weights/drone_bird_v1.pt`.

## Worth doing once on the PC (was impossible on 8 GB / macOS)

- Build the full physics-in-Gazebo mode: `make px4_sitl gz_x500` inside the
  PX4 checkout, then `./scripts/start_sim.sh --mode gz`. On Ubuntu this is the
  supported path; the drone then physically flies *in* Gazebo instead of the
  SIH pose-mirror arrangement.
- Raise the camera to 30 Hz or higher resolution in
  `simulator/worlds/detection_world.sdf` — the 720p@15Hz choice was a RAM
  budget decision, not a design one.
- Train with `--batch 32 --epochs 40` — the Mac runs were sized to fit 8 GB.

## WSL2 gotchas

- Everything (Gazebo, PX4, Python) must run inside WSL — don't mix Windows-side
  and WSL-side processes; the UDP loopback plumbing between them is painful.
- `gz sim -g` (GUI) renders through WSLg; if it's glitchy, run headless and use
  `scripts/capture_clip.py` for visuals, exactly as on the Mac.
- Keep the repo in the Linux filesystem (`~/...`), NOT under `/mnt/c/...` —
  cross-filesystem I/O is 10x slower and PX4's build will crawl.

## Claude Code on the PC (continuing this chat's work)

The conversation history and Claude's per-project memory live on the Mac and
do not sync between machines. The intended handoff is the repo itself:
[handoff.md](handoff.md) tells a fresh session exactly where the project
stands and what to do next.

Inside WSL2 Ubuntu:

```bash
curl -fsSL https://claude.ai/install.sh | bash    # or: npm install -g @anthropic-ai/claude-code
cd ~/fpv-drone-system
claude
```

First message to give it: "Read docs/handoff.md and continue from there."

To *view* a session from another device, Claude Code also has remote/cloud
options (the desktop and web apps at claude.ai/code can show cloud sessions,
and newer builds can hand a local session over to the cloud). Check `/help`
in your build for what it offers — but for actual work on the PC, a fresh
local session + handoff.md is the reliable route, and it keeps the heavy
compute on the PC where you want it.
