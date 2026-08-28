#!/usr/bin/env python3
"""Run one fine-tune experiment with kwarg overrides; save every epoch.

Sidesteps two traps hit repeatedly on this project:
  - ultralytics' project=/name= path resolution is unreliable when the
    working directory changes (background processes, WSL) — this reads
    model.trainer.save_dir back after training instead of assuming it.
  - a training-list txt with "../"-relative lines gets silently dropped as
    "corrupt" by ultralytics' scanner (it only resolves "./"-prefixed lines
    against the dataset root) — this is why the 2026-08-28 mixed-domain
    dataset builders write ABSOLUTE paths into their txt files. If you write
    a new combined dataset list, do the same: absolute paths only.

    python scripts/train_experiment.py --name my_exp --base drone \
        --kwargs '{"epochs": 25, "data": "data/finetune/dataset_mixed_v2.yaml"}' \
        --out-dir runs/experiments
"""
import argparse
import json
import shutil
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--base", required=True,
                    help="checkpoint path, or 'drone' (sapoepsilon) / 'coco' "
                         "(yolo11s.pt), or a model .yaml (then --load supplies "
                         "the checkpoint)")
    ap.add_argument("--load", default=None,
                    help="with a .yaml base: checkpoint to transfer weights "
                         "from ('drone' or a path)")
    ap.add_argument("--kwargs", default="{}",
                    help="JSON of train() overrides, or @path to a JSON file")
    ap.add_argument("--out-dir", default="runs/experiments",
                    help="where per-experiment weights/logs land")
    args = ap.parse_args()
    if args.kwargs.startswith("@"):
        args.kwargs = Path(args.kwargs[1:]).read_text()

    from huggingface_hub import hf_hub_download
    from ultralytics import YOLO

    def resolve(name):
        if name == "drone":
            return hf_hub_download("sapoepsilon/yolov11s-drone-detector", "best.pt")
        if name == "coco":
            return "yolo11s.pt"
        return name

    base = resolve(args.base)
    overrides = json.loads(args.kwargs)
    defaults = dict(
        data=str(REPO / "data/finetune/dataset.yaml"),
        epochs=25, batch=32, imgsz=640, device="cuda",
        project=str(REPO / args.out_dir), name=args.name, exist_ok=True,
        seed=0, deterministic=True, workers=4, patience=0,
        optimizer="AdamW", lr0=0.001667, lrf=0.01, warmup_bias_lr=0.0,
        save_period=1,  # per-epoch checkpoints — see sweep_checkpoints.py
        # tiny-target-safe augmentation, carried from the original recipe
        scale=0.2, mosaic=0.6, flipud=0.0, fliplr=0.5, translate=0.1,
        close_mosaic=5, plots=False, verbose=True,
    )
    defaults.update(overrides)
    print(f"EXP {args.name} base={base}", flush=True)
    print("kwargs:", json.dumps(defaults), flush=True)

    model = YOLO(base)
    if str(base).endswith(".yaml") and args.load:
        ckpt = resolve(args.load)
        model = model.load(ckpt)
        print(f"EXP loaded weights from {ckpt} into {base}", flush=True)

    t0 = time.time()
    model.train(**defaults)
    print(f"EXP TRAIN MINUTES {(time.time() - t0) / 60:.1f}", flush=True)

    trainer_dir = Path(model.trainer.save_dir)
    out_dir = REPO / args.out_dir / "weights"
    out_dir.mkdir(parents=True, exist_ok=True)
    outs = {}
    for kind in ("best", "last"):
        src = trainer_dir / "weights" / f"{kind}.pt"
        if src.exists():
            dst = out_dir / f"{args.name}_{kind}.pt"
            shutil.copy2(src, dst)
            outs[kind] = str(dst)
    print(f"EXP DONE name={args.name} run_dir={trainer_dir} " +
          " ".join(f"{k}={v}" for k, v in outs.items()), flush=True)


if __name__ == "__main__":
    main()
