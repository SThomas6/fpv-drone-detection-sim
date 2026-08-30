#!/usr/bin/env bash
# Run FERS across a whole trajectory, one dwell per output frame, so the
# passive radar becomes a per-frame SENSOR instead of a single demonstration.
#
# Design, and why it is not "one FERS run per video frame":
#   * A passive radar does not update at 15 Hz. It integrates a dwell
#     (~20 ms here), processes it, then takes the next. So dwells are placed
#     at intervals along the trajectory, which is what the real thing does.
#   * The REFERENCE channel is run ONCE. Transmitter and receivers are
#     static and the waveform is seeded, so the direct path is identical
#     every dwell - exactly as a real reference antenna would see it.
#   * FIVE receivers at HALF-WAVELENGTH spacing (0.25 m at 600 MHz). One
#     receiver gives range and Doppler but no direction. Spacing wider than
#     lambda/2 aliases: a 0.335 m baseline was tried first and gave 36 deg
#     median bearing error, too coarse to associate with a camera track.
#     At lambda/2 the array is unambiguous across the whole forward
#     hemisphere, and a 1.0 m aperture narrows the beam to ~28 deg.
#
# Everything runs in ONE wsl invocation and writes into the repo: WSL's /tmp
# is cleared between invocations, a trap this project has hit three times.
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
FERS=/opt/fers-build/FERS/build/packages/fers-cli/fers-cli
DTD=$(find /opt/fers-build/FERS -name fers-xml.dtd | head -1)
CLIP=${CLIP:-data/clips/terrain_ir_sweep}
OUT="$REPO/radar/fers_track/$(basename "$CLIP")"
WORK=/opt/fers-track
DWELLS=${DWELLS:-90}
DUR=${DUR:-0.02}
RATE=${RATE:-2.0e6}
RCS=${RCS:-0.01}
TX_X=${TX_X:--15000}; TX_Y=${TX_Y:-12000}
mkdir -p "$WORK" "$OUT"; cd "$WORK"; cp "$DTD" .

# Pick the dwell moments and the target state at each, from the real labels.
python3 - "$REPO/$CLIP" "$DWELLS" > dwells.txt <<'PY'
import json, sys
clip, n = sys.argv[1], int(sys.argv[2])
recs = [json.loads(l) for l in open(f"{clip}/labels.jsonl")]
vis = [r for r in recs if r.get("visible") and r.get("pos")]
step = max(1, len(vis) // n)
sel = vis[::step][:n]
for i, r in enumerate(sel):
    j = min(len(recs) - 1, recs.index(r) + 1)
    nxt = recs[j]
    dt = max(nxt["t"] - r["t"], 1e-3)
    vx = (nxt["pos"][0] - r["pos"][0]) / dt if nxt.get("pos") else 0.0
    vy = (nxt["pos"][1] - r["pos"][1]) / dt if nxt.get("pos") else 0.0
    print(r["frame"], r["pos"][0], r["pos"][1], r["pos"][2], vx, vy)
PY
echo "$(wc -l < dwells.txt) dwells selected"

emit_xml() {  # name x y z vx vy include_target
python3 - "$@" <<PY
import sys
name, x, y, z, vx, vy, inc = sys.argv[1:8]
x, y, z, vx, vy = map(float, (x, y, z, vx, vy))
dur = float("$DUR"); inc = inc == "1"
tgt = "" if not inc else f'''
  <platform name="Target">
    <motionpath interpolation="linear">
      <positionwaypoint><x>{x}</x><y>{y}</y><altitude>{z}</altitude><time>0.0</time></positionwaypoint>
      <positionwaypoint><x>{x+vx*dur}</x><y>{y+vy*dur}</y><altitude>{z}</altitude><time>{dur}</time></positionwaypoint>
    </motionpath>
    <fixedrotation><startazimuth>0</startazimuth><startelevation>0</startelevation>
      <azimuthrate>0</azimuthrate><elevationrate>0</elevationrate></fixedrotation>
    <target name="Drone"><rcs type="isotropic"><value>$RCS</value></rcs></target>
  </platform>'''
rx = "".join(f'''
  <platform name="Rx{k}">
    <motionpath interpolation="static"><positionwaypoint>
      <x>0.0</x><y>{off}</y><altitude>2.5</altitude><time>0.0</time></positionwaypoint></motionpath>
    <fixedrotation><startazimuth>0</startazimuth><startelevation>0</startelevation>
      <azimuthrate>0</azimuthrate><elevationrate>0</elevationrate></fixedrotation>
    <receiver name="RxA{k}" antenna="Iso" timing="Clk" nodirect="false">
      <fmcw_mode dechirp_mode="none"/><noise_temp>290.0</noise_temp></receiver>
  </platform>''' for k, off in enumerate((-0.5, -0.25, 0.0, 0.25, 0.5)))
print(f'''<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE simulation SYSTEM "fers-xml.dtd">
<simulation name="{name}">
  <parameters><starttime>0.0</starttime><endtime>{dur}</endtime>
    <rate>$RATE</rate><simSamplingRate>$RATE</simSamplingRate>
    <randomseed>7</randomseed></parameters>
  <antenna name="Iso" pattern="isotropic"><efficiency>1.0</efficiency></antenna>
  <waveform name="Illum"><power>50000.0</power><carrier_frequency>6.0e8</carrier_frequency>
    <fmcw_linear_chirp direction="up"><chirp_bandwidth>8.0e5</chirp_bandwidth>
      <chirp_duration>2.0e-4</chirp_duration><chirp_period>2.0e-4</chirp_period>
      <start_frequency_offset>-4.0e5</start_frequency_offset>
      <chirp_count>{max(1,int(dur/2.0e-4))}</chirp_count></fmcw_linear_chirp></waveform>
  <timing name="Clk"><frequency>6.0e8</frequency></timing>
  <platform name="Tx">
    <motionpath interpolation="static"><positionwaypoint>
      <x>$TX_X</x><y>$TX_Y</y><altitude>150.0</altitude><time>0.0</time></positionwaypoint></motionpath>
    <fixedrotation><startazimuth>0</startazimuth><startelevation>0</startelevation>
      <azimuthrate>0</azimuthrate><elevationrate>0</elevationrate></fixedrotation>
    <transmitter name="TxA" antenna="Iso" waveform="Illum" timing="Clk"><fmcw_mode/></transmitter>
  </platform>{rx}{tgt}
</simulation>''')
PY
}

save_iq() {  # tag
python3 - "$1" "$OUT" <<'PY'
import sys, glob, h5py, numpy as np
tag, out = sys.argv[1], sys.argv[2]
chans = []
for k in range(5):
    fn = f"RxA{k}_results.h5"
    f = h5py.File(fn, "r")
    got = {}
    def grab(n, o):
        if isinstance(o, h5py.Dataset) and o.size > 16:
            key = n.split("/")[-1]
            if key in ("I_data", "Q_data"):
                got[key] = np.asarray(o, dtype=np.float64)
    f.visititems(grab)
    i = got["I_data"]
    q = got.get("Q_data", np.zeros_like(i))
    chans.append((i + 1j * q).astype(np.complex64))
n = min(len(c) for c in chans)
np.save(f"{out}/{tag}.npy", np.stack([c[:n] for c in chans]))
PY
}

# --- reference: static geometry, so ONE run serves every dwell ---
emit_xml reference 0 0 0 0 0 0 > reference.fersxml
rm -f RxA*_results.h5
"$FERS" reference.fersxml > /opt/fers_ref.log 2>&1 || {
  echo "reference FAILED"; tail -12 /opt/fers_ref.log; exit 1; }
save_iq reference
echo "reference captured"

i=0
while read -r frame x y z vx vy; do
  emit_xml "d$i" "$x" "$y" "$z" "$vx" "$vy" 1 > dwell.fersxml
  rm -f RxA*_results.h5
  "$FERS" dwell.fersxml > /opt/fers_d.log 2>&1 || {
    echo "dwell $i FAILED"; tail -12 /opt/fers_d.log; exit 1; }
  save_iq "surv_$frame"
  i=$((i+1))
  [ $((i % 20)) -eq 0 ] && echo "  $i dwells"
done < dwells.txt
cp dwells.txt "$OUT/dwells.txt"
echo "FERSTRACKDONE $i dwells -> $OUT"
