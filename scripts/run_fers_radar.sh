#!/usr/bin/env bash
# MONOSTATIC radar with genuine micro-Doppler, from FERS.
#
# scripts/radar_sim.py asserts the thing that matters: "a multirotor is
# called a drone 93% of a dwell, a bird 92%", taken from the literature.
# That is the number the whole cued-radar architecture rests on, and nothing
# in this project has ever measured it.
#
# FERS models point scatterers, not articulated bodies - so a drone is built
# as several scatterers instead of one: a body, plus BLADE TIPS on circular
# motion paths at the rotor rate. Their radial velocity then swings
# sinusoidally at hundreds of Hz, which is exactly the micro-Doppler
# sideband structure a real counter-UAS radar classifies on. It is not
# imposed on the signal; it falls out of the geometry.
#
# A bird is built the same way with the physics that actually differs: wings
# that flap at 3-8 Hz over a much smaller radius, and no rigid rotor. If the
# classifier can separate these two from the returned I/Q, the separation is
# earned.
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
FERS=/opt/fers-build/FERS/build/packages/fers-cli/fers-cli
DTD=$(find /opt/fers-build/FERS -name fers-xml.dtd | head -1)
OUT="$REPO/radar/fers_md"
WORK=/opt/fers-md
DUR=${DUR:-0.08}
RATE=${RATE:-2.0e6}
PRF=${PRF:-20000}
mkdir -p "$WORK" "$OUT"; cd "$WORK"; cp "$DTD" .

emit() {  # kind range_m
python3 - "$1" "$2" "$DUR" "$RATE" "$PRF" <<'PY'
import sys, math
kind, rng, dur, rate, prf = sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), sys.argv[4], float(sys.argv[5])

def circle(name, cx, cy, cz, radius, rate_hz, rcs, n=48, phase=0.0, plane="rotor"):
    """A scatterer on a closed path, sampled finely enough to be smooth.

    FERS interpolates linearly between waypoints, so the path has to be
    sampled well above the rotation rate or the micro-Doppler is aliased
    into a staircase.
    """
    pts = []
    for k in range(n + 1):
        t = dur * k / n
        a = 2 * math.pi * rate_hz * t + phase
        if plane == "rotor":          # blade tip sweeps a horizontal disc
            x, y, z = cx + radius * math.cos(a), cy + radius * math.sin(a), cz
        else:                          # wing flaps mostly vertically
            x, y, z = cx, cy + 0.35 * radius * math.sin(a), cz + radius * math.sin(a)
        pts.append(f'<positionwaypoint><x>{x:.4f}</x><y>{y:.4f}</y>'
                   f'<altitude>{z:.4f}</altitude><time>{t:.6f}</time></positionwaypoint>')
    return f'''
  <platform name="{name}">
    <motionpath interpolation="linear">{"".join(pts)}</motionpath>
    <fixedrotation><startazimuth>0</startazimuth><startelevation>0</startelevation>
      <azimuthrate>0</azimuthrate><elevationrate>0</elevationrate></fixedrotation>
    <target name="{name}T"><rcs type="isotropic"><value>{rcs}</value></rcs></target>
  </platform>'''

# target closes slowly, so the body sits at a steady bulk Doppler
cx, cy, cz = rng, 0.0, 60.0
parts = [circle("Body", cx, cy, cz, 0.0, 0.0, 0.02 if kind == "drone" else 0.01)]
if kind == "drone":
    # four rotors, 0.12 m blade tips, ~6000 rpm = 100 rev/s
    for i, (ox, oy) in enumerate([(0.17, 0.17), (0.17, -0.17), (-0.17, 0.17), (-0.17, -0.17)]):
        parts.append(circle(f"Blade{i}", cx + ox, cy + oy, cz, 0.12, 100.0,
                            0.0015, phase=i * 1.4))
else:
    # a bird: two wings flapping at 5 Hz over 0.25 m, no rotor anywhere
    for i, oy in enumerate((0.12, -0.12)):
        parts.append(circle(f"Wing{i}", cx, cy + oy, cz, 0.25, 5.0, 0.004,
                            phase=i * math.pi, plane="wing"))

print(f'''<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE simulation SYSTEM "fers-xml.dtd">
<simulation name="{kind}">
  <parameters><starttime>0.0</starttime><endtime>{dur}</endtime>
    <rate>{rate}</rate><simSamplingRate>{rate}</simSamplingRate>
    <randomseed>11</randomseed></parameters>
  <antenna name="Iso" pattern="isotropic"><efficiency>1.0</efficiency></antenna>
  <waveform name="Pulse"><power>20000.0</power><carrier_frequency>1.0e10</carrier_frequency>
    <fmcw_linear_chirp direction="up"><chirp_bandwidth>1.0e6</chirp_bandwidth>
      <chirp_duration>{1.0/prf}</chirp_duration><chirp_period>{1.0/prf}</chirp_period>
      <start_frequency_offset>-5.0e5</start_frequency_offset>
      <chirp_count>{int(dur*prf)}</chirp_count></fmcw_linear_chirp></waveform>
  <timing name="Clk"><frequency>1.0e10</frequency></timing>
  <platform name="Station">
    <motionpath interpolation="static"><positionwaypoint>
      <x>0.0</x><y>0.0</y><altitude>2.5</altitude><time>0.0</time></positionwaypoint></motionpath>
    <fixedrotation><startazimuth>0</startazimuth><startelevation>0</startelevation>
      <azimuthrate>0</azimuthrate><elevationrate>0</elevationrate></fixedrotation>
    <transmitter name="Tx" antenna="Iso" waveform="Pulse" timing="Clk"><fmcw_mode/></transmitter>
    <receiver name="Rx" antenna="Iso" timing="Clk" nodirect="true">
      <fmcw_mode dechirp_mode="none"/><noise_temp>290.0</noise_temp></receiver>
  </platform>{"".join(parts)}
</simulation>''')
PY
}

save() {  # tag
python3 - "$1" "$OUT" <<'PY'
import sys, h5py, numpy as np
tag, out = sys.argv[1], sys.argv[2]
f = h5py.File("Rx_results.h5", "r")
got = {}
def grab(n, o):
    if isinstance(o, h5py.Dataset) and o.size > 16:
        k = n.split("/")[-1]
        if k in ("I_data", "Q_data"):
            got[k] = np.asarray(o, dtype=np.float64)
f.visititems(grab)
i = got["I_data"]; q = got.get("Q_data", np.zeros_like(i))
np.save(f"{out}/{tag}.npy", (i + 1j * q).astype(np.complex64))
print(f"  {tag}: {len(i)} samples")
PY
}

for kind in drone bird; do
  for rng in 100 200 300 450 600 800 1000; do
    emit "$kind" "$rng" > s.fersxml
    rm -f Rx_results.h5
    "$FERS" s.fersxml > /opt/fers_md.log 2>&1 || {
      echo "FERS failed ($kind @ $rng):"; tail -15 /opt/fers_md.log; exit 1; }
    save "${kind}_${rng}"
  done
done
echo "FERSMDDONE"
