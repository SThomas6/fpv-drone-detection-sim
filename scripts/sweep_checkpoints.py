#!/usr/bin/env python3
"""Pick the best training checkpoint by the PROJECT's real metric, not
ultralytics' fitness.

Why this exists: in the installed ultralytics build, best.pt is selected by
mAP50-95, which is measured noise on the 2-13 px targets this project cares
about (a 1 px shift on a 3 px box swings IoU across several thresholds).
Every checkpoint gets scored with the held-out evaluator's own centre-distance
matching instead.

Winner rule: among checkpoints reaching the recall gate on the given clip,
take the one with the lowest FP/min at its best-scoring threshold; if none
reaches the gate, take max recall (tie-break: lower FP/min).

    python scripts/sweep_checkpoints.py --run-dir runs/experiments/my_exp \
        --clip data/clips/eval_birds --stride 4 --prefix my_exp
"""
import argparse
import json
import re
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PY = REPO / ".venv" / "Scripts" / "python.exe"
RECALL_GATE = 0.99


def eval_ckpt(clip: str, weights: Path, tag: str) -> dict:
    subprocess.run([str(PY), "camera/evaluate.py", "run", "--clip", clip,
                    "--mode", "full", "--tag", tag, "--weights", str(weights)],
                   cwd=REPO, check=True, capture_output=True)
    subprocess.run([str(PY), "camera/evaluate.py", "analyze", "--clip", clip,
                    "--mode", "full", "--tag", tag],
                   cwd=REPO, check=True, capture_output=True)
    return json.loads((REPO / clip / f"summary_full_{tag}.json").read_text())


def score(summary: dict, recall_gate: float):
    rows = summary["sweep"]
    gated = [r for r in rows if r["recall"] >= recall_gate]
    if gated:
        best = min(gated, key=lambda r: (r["fp_per_min"], -r["conf"]))
        return (True, best["conf"], best["recall"], best["fp_per_min"],
                best.get("fp_on_birds"))
    best = max(rows, key=lambda r: (r["recall"], -r["fp_per_min"]))
    return (False, best["conf"], best["recall"], best["fp_per_min"],
            best.get("fp_on_birds"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True,
                    help="experiment dir containing weights/epoch*.pt")
    ap.add_argument("--clip", required=True, help="held-out clip to sweep against")
    ap.add_argument("--stride", type=int, default=4,
                    help="evaluate every Nth epoch checkpoint")
    ap.add_argument("--prefix", required=True, help="tag prefix for output files")
    ap.add_argument("--recall-gate", type=float, default=RECALL_GATE)
    args = ap.parse_args()

    wdir = Path(args.run_dir) / "weights"
    epochs = sorted(
        ((int(re.search(r"epoch(\d+)", p.name).group(1)), p)
         for p in wdir.glob("epoch*.pt")), key=lambda t: t[0])
    chosen = [(n, p) for n, p in epochs if n % args.stride == 0 or n == epochs[-1][0]]
    for name in ("last", "best"):
        p = wdir / f"{name}.pt"
        if p.exists():
            chosen.append((name, p))

    results = []
    print(f"sweeping {len(chosen)} checkpoints from {wdir} against {args.clip}",
          flush=True)
    for n, p in chosen:
        tag = f"{args.prefix}_ep{n}" if isinstance(n, int) else f"{args.prefix}_{n}"
        s = eval_ckpt(args.clip, p, tag)
        gate, conf, rec, fpm, fpb = score(s, args.recall_gate)
        results.append((n, p, gate, conf, rec, fpm, fpb))
        print(f"  ckpt {str(n):>5}: gate={'Y' if gate else 'n'} conf={conf:.2f} "
              f"recall={rec:.3f} fpmin={fpm:.1f} fp_birds={fpb}", flush=True)

    def key(r):
        _, _, gate, _, rec, fpm, _ = r
        return (1, -fpm) if gate else (0, rec)
    winner = max(results, key=key)
    n, p, gate, conf, rec, fpm, fpb = winner
    print(f"\nWINNER ckpt={n} weights={p}")
    print(f"WINNER gate_met={gate} conf={conf} recall={rec} fpmin={fpm} fp_birds={fpb}")


if __name__ == "__main__":
    main()
