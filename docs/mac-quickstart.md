# Watching it work on a MacBook

No simulator, no sensors, no drone required to get started. This runs the
deployed detector and tracker against your webcam, a video file, or the
simulator frames already in the repo.

## Setup (once, ~5 minutes)

```bash
git clone https://github.com/SThomas6/fpv-drone-detection-sim.git
cd fpv-drone-detection-sim
python3 -m venv .venv && source .venv/bin/activate
pip install ultralytics opencv-python numpy
```

`torch` arrives with `ultralytics`, and on Apple silicon it uses the GPU
through Metal (MPS) without any extra work. The model weights
(`camera/weights/drone_bird_v1.pt`, 76 MB) are tracked in the repo, so the
clone brings them with it.

## Three ways to run it

**1. Prove it works, before trusting your own camera.** Replay simulator
frames where the answer is already known — this should light up almost every
frame:

```bash
python camera/watch.py --source data/clips/eval_birds
```

**2. Point the webcam at drone footage on a phone or second screen.** A
screen is a fair target: it is a real optical path, real focus, real sensor
noise.

```bash
python camera/watch.py --source 0
```

macOS will ask for camera permission on the first run. If it does not, enable
it under System Settings → Privacy & Security → Camera.

**3. A video file you have already recorded.**

```bash
python camera/watch.py --source ~/Movies/drone.mp4 --record out.mp4
```

Keys: `q` quit, `s` save the current frame, `SPACE` pause.

## What you should expect to see

A red box on each detection with its class and confidence, a green box with a
track ID once the tracker has held it for a few frames, a magnified inset of
the strongest target (small targets are invisible at full-frame scale), and a
bar showing frame rate, detection count and track count.

Roughly 15–30 FPS at 1280×720 on an M-series chip. If it lags, `--imgsz 640`
roughly doubles the rate and costs a little sensitivity on the smallest
targets.

## What this test can and cannot tell you

**It can** tell you the detector runs, how fast it runs on your hardware, that
the tracker holds a target through frames where detection blinks, and roughly
how it behaves on imagery that is not from the simulator.

**It cannot** tell you the system's real range. Everything measured in this
project says the wide camera cannot resolve a drone past about 150 m, and the
range comes from the telephoto, the thermal camera and the passive radar —
none of which you have yet. A drone at 20–50 m, or footage on a screen, is a
fair test. A speck at 500 m is not, and no amount of live running will change
that.

**Expect more false positives than the benchmarks show.** The model was
trained on simulator imagery plus the Anti-UAV sequences; a garden, a street
or a cluttered skyline is neither. `--conf 0.25` is the default here rather
than the benchmark's 0.10 for exactly that reason. Raise it to 0.4 if your
scene is busy; lower it toward 0.1 if you are looking at clean sky and the
target is small.

That gap between benchmark and back garden is the single most useful thing
this test can show you, and it is worth looking at carefully before choosing
any hardware.
