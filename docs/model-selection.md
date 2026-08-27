# Phase 2 model selection — small-FPV-drone visual detection

**Status: DECIDED (research 2026-08-25), IMPLEMENTED AND MEASURED (2026-08-27).**

> Outcome: the primary model was adopted **without fine-tuning** and reached
> 100% recall to ~200 m on simulated imagery. SAHI was measured and kept only
> as an opt-in accuracy mode. ByteTrack was **replaced** by a custom
> centroid/Kalman tracker — see [phase2-results.md](phase2-results.md) and
> [detection-notes.md](detection-notes.md). The fine-tune plan below remains
> valid for real-world footage.

Per the plan amendment (2026-08-19), Phase 2 uses the best available researched
model for detecting **small FPV drones** rather than a generic object detector.
A 13-agent research workflow surveyed Hugging Face / Roboflow checkpoints,
anti-UAV datasets, 2024-26 small-object SOTA, and Apple-Silicon practicality;
8 candidates were adversarially verified (weights actually downloadable,
loadable on Python 3.12 + MPS, license checked, small-object evidence checked).

## Primary (composite)

**`sapoepsilon/yolov11s-drone-detector` (Hugging Face) + SAHI tiling + ByteTrack**

- YOLOv11s fine-tune, `best.pt` 19.2 MB — verified ungated download, loads with
  `ultralytics` 8.4.x on Python 3.12 / MPS.
- Only verified drone fine-tune with a held-out eval on a real anti-UAV
  benchmark (Anti-UAV-RGBT — ground-camera, small/distant drones: exactly our
  regime). Trained on 54k images incl. single-digit-pixel-width drone boxes.
- Self-reported recall ~0.76 ⇒ ~1 in 4 distant frames missed if used alone —
  hence the composite:
  - **SAHI v0.12.6** (MIT) 640-px tiled inference as switchable "accuracy mode"
    (+5-7 AP on small objects from slicing alone, per the SAHI paper).
  - **ByteTrack** (`model.track(..., tracker="bytetrack.yaml")`) for temporal
    persistence to bridge per-frame misses.
  - Optional: frame-difference motion gating in front of the detector (fixed
    camera ⇒ cheap; technique from the CVPR2025 4th Anti-UAV Track-1 winner).
- Throughput on this 8 GB M-series: ≥10 FPS full-frame at imgsz=1280;
  ~4-6 FPS in SAHI mode. Both acceptable.

Install/inference recipe:

```python
# pip install ultralytics huggingface_hub sahi
from huggingface_hub import hf_hub_download
from ultralytics import YOLO

weights = hf_hub_download("sapoepsilon/yolov11s-drone-detector", "best.pt")
model = YOLO(weights)
res = model.predict(frame, imgsz=1280, conf=0.15, device="mps")
# NOTE: YOLO("sapoepsilon/...") does NOT resolve HF repo ids — download first.
# Once at setup: compare MPS vs CPU boxes on one frame (known MPS coord issue).
```

## Backup

**`Hibou-Foundation/rtdetr-drone-detection`** — HF transformers RT-DETR 2-class
drone fine-tune (171.5 MB safetensors, verified real drone head, loads via
`AutoModelForObjectDetection` on MPS). Genuinely small training boxes
(9-107 px). Caveats: no license on the weights (private R&D evaluation only),
empty model card, no metrics. Benchmark head-to-head vs primary in Phase 2.

Escalation path if both underperform / license becomes a problem: self
fine-tune D-FINE-small or RT-DETRv2 (Apache-2.0) on DUT Anti-UAV + Seraphim
via free Colab — the fully license-clean route.

## License note

Treat the primary as **AGPL-3.0** (fine-tune of AGPL yolo11s.pt; the
`ultralytics` runtime is AGPL regardless of the repo's Apache tag). Zero
practical obligation for this personal, non-distributed R&D simulator; if ever
published/served: AGPL the app, buy an Ultralytics license, or migrate to the
Apache-clean stack. SAHI is MIT.

## Fine-tune plan (likely needed, after Phase-2 baseline)

Two stacked domain gaps: synthetic-vs-real (Gazebo renders ≠ real photos) and
airframe gap (training data skews DJI-class, not 5-inch racers). Decision rule:
run the Phase-2 baseline first; if <32 px recall on sim clips is within ~10
points of the Anti-UAV-RGBT figure, skip the fine-tune. Otherwise: fine-tune
from sapoepsilon's checkpoint (not scratch) on free Colab T4, 20-50 epochs, on
(a) 2-5k auto-labeled Gazebo frames (sim gives free ground-truth bboxes),
(b) real-data anchor: Seraphim (CC-BY-4.0) and/or DUT Anti-UAV,
(c) Cranfield synthetic (CC-BY-4.0) as bridge domain + bird hard-negatives.
Fine-tune on SLICED 640 crops matching SAHI inference tiling (+13-15 AP vs
slicing alone). 8 GB Mac = smoke tests only; real run on Colab.

## Phase-2 test plan (what "reliable" means here)

1. Recall/precision bucketed by target size: <16 px, 16-32 px, 32-96 px,
   >96 px box width at 1280x720 — the <32 px buckets are the decision
   criterion, not overall mAP.
2. Head-to-head: primary vs backup vs one sanity baseline, at imgsz 640, 1280,
   and SAHI tile mode, on identical Gazebo clips.
3. FPS + peak unified-memory on the 8 GB Mac with the simulator running.
4. Detection-range curve: max distance at which per-frame recall ≥0.5 / ≥0.9,
   per background (sky, terrain, mixed horizon).
5. False positives per minute on drone-free footage with distractors.
6. Track-level: time-to-first-detection, ID switches, longest gap bridged;
   does ByteTrack lift effective recall on the <32 px bucket?
7. Confidence-threshold sweep (0.05-0.5) to pick the operating point.
8. MPS-vs-CPU parity check once at setup.
9. Sim-to-real spot check on a handful of real FPV clips.

## Rejected candidates (verified but outscored)

- `doguilmak/Drone-Detection-YOLOv11x`, `FilippTrigub/yolov11x-drone-finetuned`:
  close-up training data (avg object ~35% of frame), no small-object evidence,
  x-size too slow on this machine; FilippTrigub's own README shows recall 0.606.
- `TomSmail/drone-yolo-v1`, `Javvanny/yolov8m_flying_objects_detection`:
  unbenchmarked, training data mostly medium/large objects, no FPV airframes.
- `IRIS-Computer-Vision/YOLOv8s_EO_Drone_Detection`: reported perfect metrics
  indicate train/val leakage; card admits long-range weakness; CC-BY-NC.
- Ultralytics YOLO11 COCO weights: healthiest stack but no drone class.
- YOLO-World / open-vocab: weak at few-dozen-pixel targets; useful later as an
  auto-labeler only.
