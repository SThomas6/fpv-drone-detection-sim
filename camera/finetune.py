#!/usr/bin/env python3
"""Fine-tune the drone detector into a two-class (drone, bird) detector.

Starts from the sapoepsilon YOLOv11s drone checkpoint (so its small-target
drone knowledge is kept) and retrains on simulator tiles built by
build_finetune_dataset.py. One detector with both classes sharing a feature
extractor is the principled fix for bird confusion: the two scores become
comparable by construction, unlike two independently-trained models.

    python camera/finetune.py --data data/finetune/dataset.yaml --epochs 20

Training runs on this Mac's GPU (MPS). Augmentation is tuned for tiny targets:
scale jitter is kept small because default scale=0.5 can shrink a 3 px drone
below the detection grid, silently turning positives into noise.
"""

from __future__ import annotations

import argparse
import shutil
import time
from pathlib import Path

FINAL_WEIGHTS = Path(__file__).resolve().parent / "weights" / "drone_bird_v1.pt"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/finetune/dataset.yaml")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default=None, help="cuda|mps|cpu (default: auto)")
    ap.add_argument("--project", default="data/finetune/runs")
    args = ap.parse_args()

    from camera.detector import WEIGHTS_FILE, WEIGHTS_REPO, pick_device
    if args.device is None:
        args.device = pick_device()
        print(f"auto-selected device: {args.device}")
    from huggingface_hub import hf_hub_download
    from ultralytics import YOLO

    # Normalise the dataset path: the committed yaml says "path: ." so it works
    # on any machine; ultralytics wants it absolute.
    data_yaml = Path(args.data).resolve()
    text = data_yaml.read_text()
    if "path: ." in text:
        data_yaml.write_text(text.replace("path: .", f"path: {data_yaml.parent}"))
        print(f"resolved dataset path -> {data_yaml.parent}")
    args.data = str(data_yaml)

    base = hf_hub_download(WEIGHTS_REPO, WEIGHTS_FILE)
    print(f"starting from {base}")
    model = YOLO(base)

    t0 = time.time()
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        project=args.project,
        name="drone_bird",
        exist_ok=True,
        seed=0,
        deterministic=True,
        workers=4,
        patience=6,
        # tiny-target-safe augmentation
        scale=0.2,
        mosaic=0.6,
        flipud=0.0,
        fliplr=0.5,
        translate=0.1,
        close_mosaic=5,
        plots=False,
        verbose=True,
    )
    mins = (time.time() - t0) / 60
    print(f"\ntraining took {mins:.1f} min")

    best = Path(args.project) / "drone_bird" / "weights" / "best.pt"
    if not best.exists():
        raise SystemExit(f"training produced no {best}")
    FINAL_WEIGHTS.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best, FINAL_WEIGHTS)
    print(f"installed {FINAL_WEIGHTS}")

    if results is not None and getattr(results, "results_dict", None):
        for k, v in results.results_dict.items():
            print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    main()
