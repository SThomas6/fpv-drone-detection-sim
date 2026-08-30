#!/usr/bin/env bash
# Passive-radar I/Q from FERS, for one target geometry.
#
# Everything happens in ONE wsl invocation and lands in the repo, because
# WSL's /tmp is cleared between invocations - a trap this project has now
# hit three times.
#
# Two runs per geometry, which is how a passive radar actually works:
#   reference   - transmitter + receiver only. This is the direct path, the
#                 clean copy of what the illuminator is sending.
#   surveillance- the same scene WITH the drone in it.
# The echo is the difference, and it is ~10^9 times weaker than the direct
# path, so the cross-ambiguity function has to dig it out of the reference's
# own sidelobes. That is the real problem, and running it on real I/Q is the
# point of using FERS rather than asserting a detection probability.
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
FERS=/opt/fers-build/FERS/build/packages/fers-cli/fers-cli
DTD=$(find /opt/fers-build/FERS -name fers-xml.dtd | head -1)
OUT="$REPO/radar/fers_out"
WORK=/opt/fers-run
TX_X=${TX_X:--15000}; TX_Y=${TX_Y:-12000}
TGT_X=${TGT_X:-600}; TGT_Y=${TGT_Y:-40}; TGT_Z=${TGT_Z:-90}
TGT_VX=${TGT_VX:--40}; TGT_VY=${TGT_VY:-8}
RCS=${RCS:-0.01}; DUR=${DUR:-0.02}; RATE=${RATE:-2.0e6}
mkdir -p "$WORK" "$OUT"; cd "$WORK"; cp "$DTD" .

scenario() { # name include_target
python3 - "$1" "$2" <<PY
import sys
name, with_tgt = sys.argv[1], sys.argv[2] == "1"
dur=float("$DUR"); vx=float("$TGT_VX"); vy=float("$TGT_VY")
x=float("$TGT_X"); y=float("$TGT_Y"); z=float("$TGT_Z")
tgt = "" if not with_tgt else f'''
    <platform name="Target">
        <motionpath interpolation="linear">
            <positionwaypoint><x>{x}</x><y>{y}</y><altitude>{z}</altitude><time>0.0</time></positionwaypoint>
            <positionwaypoint><x>{x+vx*dur}</x><y>{y+vy*dur}</y><altitude>{z}</altitude><time>{dur}</time></positionwaypoint>
        </motionpath>
        <fixedrotation><startazimuth>0</startazimuth><startelevation>0</startelevation>
          <azimuthrate>0</azimuthrate><elevationrate>0</elevationrate></fixedrotation>
        <target name="Drone"><rcs type="isotropic"><value>$RCS</value></rcs></target>
    </platform>'''
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
  </platform>
  <platform name="Rx">
    <motionpath interpolation="static"><positionwaypoint>
      <x>0.0</x><y>0.0</y><altitude>2.5</altitude><time>0.0</time></positionwaypoint></motionpath>
    <fixedrotation><startazimuth>0</startazimuth><startelevation>0</startelevation>
      <azimuthrate>0</azimuthrate><elevationrate>0</elevationrate></fixedrotation>
    <receiver name="RxA" antenna="Iso" timing="Clk" nodirect="false">
      <fmcw_mode dechirp_mode="none"/><noise_temp>290.0</noise_temp></receiver>
  </platform>{tgt}
</simulation>''')
PY
}

for tag in reference surveillance; do
  [ "$tag" = surveillance ] && inc=1 || inc=0
  scenario "$tag" "$inc" > "$tag.fersxml"
  rm -f RxA_results.h5
  "$FERS" "$tag.fersxml" > "/opt/fers_$tag.log" 2>&1 || {
    echo "FERS failed on $tag:"; tail -12 "/opt/fers_$tag.log"; exit 1; }
  python3 - "$tag" "$OUT" <<'PY'
import sys, h5py, numpy as np
tag, out = sys.argv[1], sys.argv[2]
f = h5py.File("RxA_results.h5", "r")
found = {}
def grab(n, o):
    if isinstance(o, h5py.Dataset) and o.size > 16:
        key = n.split("/")[-1]
        if key in ("I_data", "Q_data"):
            found[key] = np.asarray(o, dtype=np.float64)
f.visititems(grab)
# I and Q together or the phase is lost, and phase is the whole basis of
# both the Doppler axis and the direct-signal cancellation
i = found["I_data"]
q = found.get("Q_data", np.zeros_like(i))
z = i + 1j * q
np.save(f"{out}/{tag}.npy", z)
print(f"  {tag}: {z.shape} complex, |z| mean {np.abs(z).mean():.3e}")
PY
done
echo "FERSPCLDONE"
