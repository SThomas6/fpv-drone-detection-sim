#!/usr/bin/env bash
# The deployed configuration, per scenario, exactly as docs/handoff.md
# records it. This is the ONLY like-for-like way to see whether a change to
# the detector, the acoustic path or the radar moved the headline numbers -
# bare defaults are a different system.
set -u
cd "$(dirname "$0")/.."
P=.venv/Scripts/python.exe
EXTRA="${EXTRA:-}"
run() { printf "%-22s " "$1"; shift; $P camera/fuse_eval.py "$@" $EXTRA 2>&1 \
  | grep -E "^ *($POLICY)" | tr -s ' '; }

POLICY="and-confirm"
run "long-range sweep" --clip data/clips/terrain_ir_sweep --motion-thr 0.3 \
    --young-tracks pass --min-travel 4
POLICY="rgb-only"
run "terrain + birds" --clip data/clips/terrain_ir_birds --motion-thr 0.5 \
    --coast-alarms none --min-travel 0 --young-tracks pass
POLICY="or-fusion"
run "canopy" --clip data/clips/terrain_ir_canopy --rgb-mode both \
    --motion-thr 0.2 --young-tracks pass --coast-alarms none --clutter \
    --min-travel 4
POLICY="rgb-only"
run "sky" --clip data/clips/eval_birds --motion-thr 0.05 --young-tracks pass \
    --coast-alarms none --clutter --no-class-consistent
