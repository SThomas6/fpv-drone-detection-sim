#!/usr/bin/env python3
"""Visual drone detection for the ground station camera (Phase 2).

Wraps the model chosen in docs/model-selection.md:
  · YOLOv11s drone fine-tune (sapoepsilon/yolov11s-drone-detector)
  · optional SAHI sliced inference as an accuracy mode for tiny targets
  · ByteTrack for temporal persistence and velocity

Non-obvious details, all verified against the installed ultralytics 8.4.130
(see docs/detection-notes.md for the reasoning):

  · Colour order differs by path. Ultralytics reads a numpy array as BGR, so
    predict/track get `frame[:, :, ::-1]`. SAHI documents RGB input and flips
    internally, so it gets the frame unconverted. Getting this wrong costs
    accuracy silently.
  · imgsz is the tuple (736, 1280), not 1280. An int can letterbox to
    1280x1280 depending on flags; the tuple pins native scale, so a 6 px drone
    is never downscaled.
  · The tracker YAML is passed by absolute path. This build's default is
    tracktrack.yaml, so omitting it would silently not be ByteTrack.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# Must be set before torch is imported (ultralytics imports it at module load).
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np  # noqa: E402

WEIGHTS_REPO = "sapoepsilon/yolov11s-drone-detector"
WEIGHTS_FILE = "best.pt"
# Two-class (drone, bird) fine-tune produced by camera/finetune.py. When it
# exists it becomes the default: same architecture and speed, but the model
# itself knows what a bird is instead of calling everything a drone.
FINETUNED_WEIGHTS = Path(__file__).resolve().parent / "weights" / "drone_bird_v1.pt"
TRACKER_YAML = str(Path(__file__).resolve().parent / "drone_bytetrack.yaml")

# Native-scale inference size for 1280x720 input (both stride-32 multiples).
IMGSZ = (736, 1280)
# 0.10 measured as the best operating point across the test matrix: full recall
# inside 130 m with zero false positives on 300 frames of empty sky. Dropping to
# 0.05 buys a little recall past 200 m but starts admitting false positives.
DEFAULT_CONF = 0.10
# Tracking is fed weaker detections on purpose and lets the tracker reject them.
DEFAULT_TRACK_CONF = 0.05
DEFAULT_IOU = 0.7


@dataclass
class Detection:
    """One detection in one frame."""
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    cls_name: str = "drone"
    track_id: Optional[int] = None

    @property
    def centre(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def xyxy(self) -> list[float]:
        return [self.x1, self.y1, self.x2, self.y2]

    def report(self) -> str:
        cx, cy = self.centre
        return (f"{self.cls_name.capitalize()} detected\n"
                f"Confidence: {self.confidence:.2f}\n"
                f"Bounding box: {self.x1:.0f},{self.y1:.0f},{self.x2:.0f},{self.y2:.0f}\n"
                f"Centre: {cx:.0f},{cy:.0f}")


@dataclass
class Track:
    """A target followed across frames."""
    track_id: int
    detection: Detection
    velocity: tuple[float, float]          # pixels/second
    history: list = field(default_factory=list)

    def report(self) -> str:
        cx, cy = self.detection.centre
        vx, vy = self.velocity
        return (f"Target ID: {self.track_id}\n"
                f"Position: ({cx:.0f}, {cy:.0f})\n"
                f"Velocity: ({vx:.1f}, {vy:.1f}) px/s\n"
                f"Confidence: {self.detection.confidence:.2f}")


def pick_device() -> str:
    """Best available inference device: CUDA (PC) > MPS (Mac) > CPU."""
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def default_weights() -> str:
    """Fine-tuned two-class weights if built, else the original HF checkpoint."""
    if FINETUNED_WEIGHTS.exists():
        return str(FINETUNED_WEIGHTS)
    from huggingface_hub import hf_hub_download
    return hf_hub_download(WEIGHTS_REPO, WEIGHTS_FILE)


def _bgr(frame: np.ndarray) -> np.ndarray:
    """RGB -> BGR for ultralytics' numpy path."""
    return np.ascontiguousarray(frame[:, :, ::-1])


def merge_close(dets: list["Detection"], min_px: float = 6.0,
                size_factor: float = 1.2) -> list["Detection"]:
    """Merge detections that are really the same tiny target.

    NMS suppresses by IoU, which breaks down at this scale: two boxes 3 px
    apart on a 4 px drone barely overlap, so both survive and the tracker sees
    two targets. Merging by centre distance is the scale-appropriate fix.
    Keeps the highest-confidence box of each cluster.
    """
    out: list[Detection] = []
    for det in sorted(dets, key=lambda d: -d.confidence):
        keep = True
        for chosen in out:
            if chosen.cls_name != det.cls_name:
                continue   # never let one class swallow the other's detection
            cx, cy = det.centre
            ox, oy = chosen.centre
            limit = max(min_px, size_factor * max(det.width, chosen.width))
            if (cx - ox) ** 2 + (cy - oy) ** 2 <= limit ** 2:
                keep = False
                break
        if keep:
            out.append(det)
    return out


class DroneDetector:
    """Detects (and optionally tracks) drones in station-camera frames."""

    def __init__(self, weights: str | None = None, device: str | None = None,
                 imgsz: tuple[int, int] | int = IMGSZ, conf: float = DEFAULT_CONF,
                 iou: float = DEFAULT_IOU, use_sahi: bool = False,
                 slice_size: int = 320, sahi_image_size: int = 640,
                 overlap: float = 0.2, track_conf: float = DEFAULT_TRACK_CONF,
                 tracker: str = TRACKER_YAML, verbose: bool = False):
        from ultralytics import YOLO

        self.weights = weights or default_weights()
        self.device = device or pick_device()
        self.imgsz = imgsz
        self.conf = conf
        self.iou = iou
        self.use_sahi = use_sahi
        self.slice_size = slice_size
        self.sahi_image_size = sahi_image_size
        self.overlap = overlap
        self.track_conf = track_conf
        self.tracker = tracker
        self.verbose = verbose

        self.model = YOLO(self.weights)
        self.model.to(device)
        self.names = self.model.names

        try:
            import cv2
            cv2.setNumThreads(2)   # stop OpenCV oversubscribing against torch
        except Exception:          # noqa: BLE001 - cosmetic only
            pass

        # Kwargs must stay byte-identical between calls: ultralytics rebuilds
        # the predictor (destroying tracker state) if `device` differs.
        self._predict_kw = dict(imgsz=self.imgsz, conf=self.conf, iou=self.iou,
                                device=self.device, verbose=self.verbose)
        self._track_kw = dict(self._predict_kw, conf=self.track_conf)

        self._sahi_model = None
        self._track_hist: dict[int, list[tuple[float, float, float]]] = {}
        self._last_infer_ms: float | None = None

    # ---------------- full-frame detection ----------------

    def detect(self, frame: np.ndarray) -> list[Detection]:
        t0 = time.perf_counter()
        res = self.model.predict(_bgr(frame), **self._predict_kw)
        self._last_infer_ms = (time.perf_counter() - t0) * 1000
        return merge_close(self._from_results(res[0]))

    @staticmethod
    def _from_results(r) -> list[Detection]:
        out: list[Detection] = []
        b = r.boxes.cpu().numpy()      # one host transfer, not one per box
        if len(b) == 0:
            return out
        ids = b.id.astype(int) if b.is_track else [None] * len(b)
        for (x1, y1, x2, y2), c, k, tid in zip(b.xyxy, b.conf, b.cls.astype(int), ids):
            out.append(Detection(float(x1), float(y1), float(x2), float(y2),
                                 float(c), r.names.get(int(k), str(k)),
                                 None if tid is None else int(tid)))
        return out

    # ---------------- sliced (SAHI) detection ----------------

    def _ensure_sahi(self):
        if self._sahi_model is not None:
            return self._sahi_model
        from sahi import AutoDetectionModel
        try:
            from sahi.postprocess.backends import set_postprocess_backend
            set_postprocess_backend("numpy")
        except Exception:              # noqa: BLE001 - older sahi without backends
            pass
        self._sahi_model = AutoDetectionModel.from_pretrained(
            model_type="ultralytics", model_path=self.weights,
            confidence_threshold=self.conf, device=self.device,
            image_size=self.sahi_image_size)
        # sahi's select_device falls back to CPU silently rather than raising.
        if self.device != "cpu" and str(self._sahi_model.device) != self.device:
            print(f"WARNING: SAHI fell back to {self._sahi_model.device}, "
                  f"not {self.device}", flush=True)
        return self._sahi_model

    def detect_sliced(self, frame: np.ndarray) -> list[Detection]:
        from sahi.predict import get_sliced_prediction

        model = self._ensure_sahi()
        t0 = time.perf_counter()
        # SAHI takes RGB and flips internally — do NOT convert here.
        result = get_sliced_prediction(
            frame, model,
            slice_height=self.slice_size, slice_width=self.slice_size,
            overlap_height_ratio=self.overlap, overlap_width_ratio=self.overlap,
            auto_slice_resolution=False,
            perform_standard_pred=False,   # a full-frame pass adds cost, not tiny targets
            postprocess_type="NMS", postprocess_match_metric="IOU",
            postprocess_match_threshold=0.5, verbose=0)
        self._last_infer_ms = (time.perf_counter() - t0) * 1000

        out: list[Detection] = []
        for op in result.object_prediction_list:
            box = op.bbox
            out.append(Detection(float(box.minx), float(box.miny),
                                 float(box.maxx), float(box.maxy),
                                 float(op.score.value), op.category.name))
        return merge_close(out)

    # ---------------- tracking ----------------

    def track(self, frame: np.ndarray, timestamp: float | None = None) -> list[Track]:
        """Detect + associate across frames. Call once per frame, in order.

        Note: model.track() returns only CONFIRMED tracks, so this is not a
        raw detection feed — use detect() for that.
        """
        t = time.time() if timestamp is None else timestamp
        t0 = time.perf_counter()
        res = self.model.track(_bgr(frame), tracker=self.tracker, persist=True,
                               **self._track_kw)
        self._last_infer_ms = (time.perf_counter() - t0) * 1000

        tracks: list[Track] = []
        for det in self._from_results(res[0]):
            if det.track_id is None:
                continue
            cx, cy = det.centre
            hist = self._track_hist.setdefault(det.track_id, [])
            hist.append((t, cx, cy))
            del hist[:-10]
            tracks.append(Track(det.track_id, det, self._velocity(hist), list(hist)))
        return tracks

    @staticmethod
    def _velocity(hist) -> tuple[float, float]:
        """Pixels/second across the oldest and newest samples in the window."""
        if len(hist) < 2:
            return (0.0, 0.0)
        (t0, x0, y0), (t1, x1, y1) = hist[0], hist[-1]
        dt = t1 - t0
        if dt <= 1e-6:
            return (0.0, 0.0)
        return ((x1 - x0) / dt, (y1 - y0) / dt)

    def reset_tracks(self) -> None:
        """Clear velocity history and the underlying tracker state."""
        self._track_hist.clear()
        predictor = getattr(self.model, "predictor", None)
        for tr in getattr(predictor, "trackers", None) or []:
            tr.reset()

    # ---------------- utilities ----------------

    @property
    def last_inference_ms(self) -> float | None:
        return self._last_infer_ms

    def infer(self, frame: np.ndarray) -> list[Detection]:
        """detect() or detect_sliced(), per the use_sahi setting."""
        return self.detect_sliced(frame) if self.use_sahi else self.detect(frame)

    def warmup(self, shape=(720, 1280, 3)) -> None:
        self.infer(np.zeros(shape, dtype=np.uint8))

    def check_device_parity(self, frame: np.ndarray, tol: float = 1.0) -> list[str]:
        """Compare this device's boxes against CPU on a frame WITH a drone.

        Guards against MPS strided-tensor bugs in box post-processing, whose
        signature is x-coordinates collapsing to the frame height.
        Returns human-readable discrepancies; empty means agreement.
        """
        from ultralytics import YOLO

        mine = self.detect(frame)
        cpu_kw = dict(self._predict_kw, device="cpu")
        r = YOLO(self.weights).predict(_bgr(frame), **cpu_kw)
        # Same post-processing on both sides, or the comparison is meaningless.
        theirs = merge_close(self._from_results(r[0]))

        problems = []
        if len(mine) != len(theirs):
            return [f"{self.device} found {len(mine)} boxes, cpu found {len(theirs)}"]
        if not mine:
            return ["parity check inconclusive: no detections on this frame"]
        h = frame.shape[0]
        for a, b in zip(sorted(mine, key=lambda d: d.x1),
                        sorted(theirs, key=lambda d: d.x1)):
            if abs(a.x1 - h) < 1.0 or abs(a.x2 - h) < 1.0:
                problems.append(
                    f"{self.device} x-coordinate collapsed to frame height ({h}) — "
                    "known strided-tensor corruption; fall back to cpu")
            drift = max(abs(a.x1 - b.x1), abs(a.y1 - b.y1),
                        abs(a.x2 - b.x2), abs(a.y2 - b.y2))
            if drift > tol:
                problems.append(f"box drift {drift:.1f}px between {self.device} and cpu")
            if abs(a.confidence - b.confidence) > 0.05:
                problems.append(f"confidence drift {abs(a.confidence - b.confidence):.3f}")
        return problems
