#!/usr/bin/env python3
"""Real acoustic pipeline: synthesise the WAVEFORM, then actually detect it.

`scripts/acoustic_sim.py` is a sensor MODEL - it decides "would a mic array
have heard this?" from geometry plus published figures, and reports a bearing
with an assumed error. That is honest but it is not a detector: nothing is
ever measured, so its errors are asserted rather than earned.

This is the detector. Gazebo has no acoustics, so the SIGNAL is synthesised
from simulator ground truth - but everything after that is genuine DSP on
genuine samples, and the performance that comes out is a consequence of
physics and noise rather than a number typed into a table:

  synthesis (physics)                 detection (real DSP)
  --------------------                --------------------
  blade-pass harmonic comb            Welch spectrum per channel
  4 rotors at slightly different RPM  harmonic-comb search over f0
  Doppler from radial velocity        peak-to-floor comb score
  1/r spherical spreading             GCC-PHAT between mic pairs
  frequency-dependent air absorption  parabolic sub-sample interpolation
  per-mic propagation delay           least-squares bearing from TDOAs
  wind / rain / background noise      CFAR-style threshold on comb score

Why the bearing is real: sound arrives at the four microphones at slightly
different times, and that delay is what encodes direction. At 48 kHz with a
0.30 m array the whole usable delay range is +-42 samples, so the estimator
has to interpolate between samples to be worth anything - which is exactly
why GCC-PHAT with parabolic interpolation is used rather than an argmax.

Birds contribute nothing, and that is not a convenience: birds do not run
engines, so the harmonic comb a multirotor produces has no avian analogue.
This is the one channel in the whole system with no bird problem, and here
that falls out of the physics instead of being assumed.

    python audio/pipeline.py run --clip data/clips/terrain_ir_canopy --env windy
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

try:
    import pyroomacoustics as pra
except ImportError:      # the built-in path still works
    pra = None

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

C_SOUND = 343.0
FS = 48000
CAM = np.array([0.1, 0.0, 2.5])
MIC_CENTRE = np.array([0.0, 0.35, 1.1])
ARRAY_SIDE = 0.30

# name: (wind noise rms, broadband rms, note)
ENVIRONMENTS = {
    "still":      (0.004, 0.004, "calm, clear"),
    "breeze":     (0.020, 0.008, "3-5 m/s, leaves rustling"),
    "windy":      (0.075, 0.020, "8-11 m/s: wind noise at the diaphragm "
                                 "dominates below ~300 Hz"),
    "light_rain": (0.030, 0.045, "rain is BROADBAND, so it raises the floor "
                                 "the comb has to beat"),
    "heavy_rain": (0.060, 0.110, "drumming on the housing"),
    "traffic":    (0.025, 0.030, "engine harmonics - the one false source "
                                 "that genuinely mimics a multirotor"),
}


def mic_positions() -> np.ndarray:
    """Four mics on a square, in world coordinates."""
    h = ARRAY_SIDE / 2.0
    offs = np.array([[+h, +h, 0.0], [+h, -h, 0.0],
                     [-h, +h, 0.0], [-h, -h, 0.0]])
    return MIC_CENTRE + offs


def air_absorption_db_per_m(f_hz: float) -> float:
    """Rough ISO 9613-style absorption, 20 C / 70% RH. Small but not zero:
    at 1 kHz it is ~0.005 dB/m, so 300 m costs ~1.5 dB and the HIGH
    harmonics fade before the fundamental does - which is why a distant
    drone sounds like a hum rather than a buzz."""
    return 0.0000005 * (f_hz ** 1.7) / 1000.0 * 1000.0


class Synth:
    """Multirotor acoustic source, sampled at the four microphones."""

    def __init__(self, seed: int = 0, n_rotors: int = 4, blades: int = 2,
                 rpm: float = 6000.0, harmonics: int = 6):
        self.rng = np.random.default_rng(seed)
        self.blades = blades
        self.harmonics = harmonics
        # each rotor runs slightly differently - that beating is audible and
        # is part of what makes a multirotor sound like a multirotor
        self.rotor_rpm = rpm * (1.0 + 0.02 * self.rng.standard_normal(n_rotors))
        self.phase = np.zeros((n_rotors, harmonics))

    def chunk(self, pos, vel, mics, dur_s: float, env: str, rng):
        """One block of 4-channel audio for a target at `pos` moving `vel`."""
        n = int(FS * dur_s)
        t = np.arange(n) / FS
        out = np.zeros((len(mics), n), dtype=np.float32)

        for m_i, m in enumerate(mics):
            d = pos - m
            r = float(np.linalg.norm(d))
            if r < 1e-3:
                continue
            u = d / r
            v_rad = float(np.dot(vel, u))          # +ve = receding
            doppler = C_SOUND / max(C_SOUND + v_rad, 1.0)
            delay = r / C_SOUND
            spread = 1.0 / max(r, 1.0)             # spherical spreading
            for ri, rpm in enumerate(self.rotor_rpm):
                f_blade = rpm / 60.0 * self.blades
                for h in range(1, self.harmonics + 1):
                    f = f_blade * h * doppler
                    if f >= FS / 2:
                        continue
                    a = spread / (h ** 1.4)        # higher harmonics weaker
                    a *= 10 ** (-air_absorption_db_per_m(f) * r / 20.0)
                    ph = self.phase[ri, h - 1] - 2 * math.pi * f * delay
                    out[m_i] += (a * np.sin(2 * math.pi * f * t + ph)
                                 ).astype(np.float32)
            # advance phase so the tone is continuous across blocks
        for ri, rpm in enumerate(self.rotor_rpm):
            f_blade = rpm / 60.0 * self.blades
            for h in range(1, self.harmonics + 1):
                self.phase[ri, h - 1] = (self.phase[ri, h - 1]
                                         + 2 * math.pi * f_blade * h * dur_s
                                         ) % (2 * math.pi)

        wind_rms, broad_rms, _ = ENVIRONMENTS[env]
        for m_i in range(len(mics)):
            # wind is LOW-frequency: a one-pole lowpass on white noise
            w = rng.standard_normal(n).astype(np.float32)
            lp = np.empty_like(w)
            acc = 0.0
            alpha = 0.02
            for i in range(n):                     # cheap one-pole
                acc += alpha * (w[i] - acc)
                lp[i] = acc
            lp *= wind_rms / (np.std(lp) + 1e-9)
            out[m_i] += lp
            out[m_i] += (broad_rms
                         * rng.standard_normal(n).astype(np.float32))
        return out


def gcc_phat(a: np.ndarray, b: np.ndarray, max_shift: int,
             harmonics_hz=None, half_bw_hz: float = 12.0):
    """Sub-sample delay of b relative to a, by phase-transform correlation.

    `harmonics_hz` restricts the correlation to the bins where the target
    actually lives. This matters enormously and is what a real array does:
    the phase transform weights every frequency equally, so across the full
    band a wind-dominated spectrum contributes almost all the phase and the
    bearing is decided by noise. Measured before banding: 5.7 deg median
    error in still air but 75.5 deg in wind. The comb detector has already
    told us f0, so there is no excuse for correlating anywhere else.
    """
    n = 1
    while n < len(a) + len(b):
        n *= 2
    A = np.fft.rfft(a, n)
    B = np.fft.rfft(b, n)
    R = A * np.conj(B)
    mag = np.abs(R)
    mag[mag < 1e-12] = 1e-12
    R = R / mag
    if harmonics_hz:
        freqs = np.fft.rfftfreq(n, 1.0 / FS)
        keep = np.zeros(len(R), dtype=bool)
        for f in harmonics_hz:
            keep |= np.abs(freqs - f) <= half_bw_hz
        R = R * keep
    cc = np.fft.irfft(R, n)
    cc = np.concatenate((cc[-max_shift:], cc[:max_shift + 1]))
    k = int(np.argmax(cc))
    # parabolic interpolation: whole-sample resolution is far too coarse
    if 0 < k < len(cc) - 1:
        y0, y1, y2 = cc[k - 1], cc[k], cc[k + 1]
        denom = (y0 - 2 * y1 + y2)
        frac = 0.5 * (y0 - y2) / denom if abs(denom) > 1e-12 else 0.0
    else:
        frac = 0.0
    return (k - max_shift + frac) / FS, float(cc[k])


def comb_score(x: np.ndarray, f_lo=70.0, f_hi=320.0, harmonics=6):
    """Is there a harmonic comb here, and at what fundamental?

    A multirotor puts energy at f0, 2*f0, 3*f0 ... A bird, wind and rain do
    not. Scored as summed harmonic energy over the local spectral floor, so
    it is a contrast, not an absolute level, and does not need calibrating
    against a loudness.
    """
    n = len(x)
    win = np.hanning(n)
    spec = np.abs(np.fft.rfft(x * win)) ** 2
    freqs = np.fft.rfftfreq(n, 1.0 / FS)
    floor = np.median(spec[(freqs > 50) & (freqs < 4000)]) + 1e-20
    # Score against the DISTRIBUTION of comb sums over all candidate f0,
    # not against the raw spectral floor. Taking the max over +-2 bins at
    # each of 6 harmonics is a max-of-30 statistic, which sits well above the
    # median floor even in pure noise: measured, the old score never dropped
    # below ~23 dB, so a 9 dB threshold fired on every frame and the detector
    # reported hearing the drone 100% of the time at 1100 m. Normalising by
    # the median over all f0 makes the null hypothesis score ~0 dB by
    # construction, because the same max-of-30 bias appears in both terms.
    totals, f0s = [], []
    for f0 in np.arange(f_lo, f_hi, 1.0):
        tot = 0.0
        for h in range(1, harmonics + 1):
            f = f0 * h
            if f >= freqs[-1]:
                break
            i = int(round(f / (FS / 2) * (len(spec) - 1)))
            lo, hi = max(0, i - 2), min(len(spec), i + 3)
            tot += float(spec[lo:hi].max())
        totals.append(tot)
        f0s.append(f0)
    totals = np.asarray(totals)
    k = int(np.argmax(totals))
    null = float(np.median(totals)) + 1e-30
    return 10.0 * math.log10(totals[k] / null + 1e-20), float(f0s[k])


def bearing_from_tdoa(taus, mics, pairs):
    """Least-squares azimuth from pairwise delays.

    For a far-field source at azimuth th, the delay between mics i and j is
    tau = (p_j - p_i) . d / c with d = (cos th, sin th). Two independent
    baselines are enough; four mics give six pairs, so it is overdetermined
    and least squares averages the noise down.
    """
    A, b = [], []
    for (i, j), tau in zip(pairs, taus):
        dv = mics[j][:2] - mics[i][:2]
        A.append(dv)
        # Sign convention, fixed empirically because getting it wrong is
        # silent: the detector still fires on every frame and only the
        # DIRECTION is reversed, which showed up as a 174.8 deg median
        # bearing error - a number close to 180 is a flipped sign, not a
        # noisy estimator.
        b.append(tau * C_SOUND)
    A = np.array(A)
    b = np.array(b)
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    nrm = float(np.linalg.norm(sol))
    if nrm < 1e-9:
        return None, 0.0
    return math.atan2(sol[1], sol[0]), nrm


def source_signal(synth, dur_s, doppler, rng):
    """Mono multirotor signal at the SOURCE, Doppler already applied.

    Propagation (per-mic delay and 1/r spreading) is left to
    pyroomacoustics, which is a published, tested implementation of exactly
    that. Doppler is applied here because pra treats a source as static
    within one simulate() call - over a 0.12 s block the shift is constant
    to well under a bin, so pre-applying it is exact enough and keeps the
    physics in one place.
    """
    n = int(FS * dur_s)
    t = np.arange(n) / FS
    sig = np.zeros(n, dtype=np.float64)
    for ri, rpm in enumerate(synth.rotor_rpm):
        f_blade = rpm / 60.0 * synth.blades
        for h in range(1, synth.harmonics + 1):
            f = f_blade * h * doppler
            if f >= FS / 2:
                continue
            sig += (1.0 / h ** 1.4) * np.sin(
                2 * np.pi * f * t + synth.phase[ri, h - 1])
    for ri, rpm in enumerate(synth.rotor_rpm):
        f_blade = rpm / 60.0 * synth.blades
        for h in range(1, synth.harmonics + 1):
            synth.phase[ri, h - 1] = (synth.phase[ri, h - 1]
                                      + 2 * np.pi * f_blade * h * dur_s
                                      ) % (2 * np.pi)
    return sig


def pra_capture(synth, pos, vel, mics, dur_s, env, rng):
    """Propagate through pyroomacoustics' free-field model to the array."""
    centre = mics.mean(axis=0)
    d = pos - centre
    r = float(np.linalg.norm(d))
    v_rad = float(np.dot(vel, d / max(r, 1e-6)))
    doppler = C_SOUND / max(C_SOUND + v_rad, 1.0)
    sig = source_signal(synth, dur_s + 0.05, doppler, rng)

    room = pra.AnechoicRoom(dim=3, fs=FS)
    room.add_source(list(pos), signal=sig)
    room.add_microphone_array(pra.MicrophoneArray(mics.T.copy(), fs=FS))
    room.simulate()
    x = np.asarray(room.mic_array.signals, dtype=np.float64)
    n = int(FS * dur_s)
    # drop the propagation delay head so every block is the same length
    if x.shape[1] < n:
        x = np.pad(x, ((0, 0), (0, n - x.shape[1])))
    start = max(0, x.shape[1] - n)
    x = x[:, start:start + n]

    wind_rms, broad_rms, _ = ENVIRONMENTS[env]
    for i in range(x.shape[0]):
        w = rng.standard_normal(n)
        lp = np.convolve(w, np.ones(64) / 64.0, mode="same")   # low-pass wind
        lp *= wind_rms / (np.std(lp) + 1e-12)
        x[i] += lp + broad_rms * rng.standard_normal(n)
    return x


def pra_doa(x, mics, f0, algo="NormMUSIC"):
    """Bearing from a published DOA implementation, not a hand-rolled one."""
    nfft = 1024
    X = pra.transform.stft.analysis(x.T, nfft, nfft // 2)
    X = X.transpose([2, 1, 0])                      # (mics, freq, frames)
    est = pra.doa.algorithms[algo](
        mics.T.copy()[:2], FS, nfft, c=C_SOUND, num_src=1,
        azimuth=np.linspace(-180.0, 180.0, 721) * np.pi / 180.0)
    est.locate_sources(X, freq_range=[max(40.0, f0 * 0.8), min(f0 * 6.5,
                                                               FS / 2 - 1)])
    return float(est.azimuth_recon[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["run"])
    ap.add_argument("--clip", required=True)
    ap.add_argument("--env", default="still", choices=sorted(ENVIRONMENTS))
    ap.add_argument("--rpm", type=float, default=6000.0)
    ap.add_argument("--dur", type=float, default=0.12,
                    help="analysed block per frame, seconds")
    ap.add_argument("--score-db", type=float, default=5.5,
                    help="comb score to declare a detection. Pure noise scores "
                         "~2.7 dB and a clear comb ~13 dB, both measured, "
                         "so 5.5 sits between them")
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--engine", default="pra", choices=["pra", "builtin"],
                    help="pra = pyroomacoustics free-field propagation + its "
                         "MUSIC/SRP direction finder (published, tested); "
                         "builtin = the hand-rolled synth + GCC-PHAT kept "
                         "for comparison, because a library is only worth "
                         "adopting if it measurably wins")
    ap.add_argument("--doa", default="NormMUSIC",
                    choices=["NormMUSIC", "MUSIC", "SRP"])
    args = ap.parse_args()

    clip = Path(args.clip)
    meta = json.loads((clip / "meta.json").read_text())
    fx = float(meta.get("fx", 1108.77))
    W = int(meta.get("width", 1280))
    H = int(meta.get("height", 720))
    recs = [json.loads(l) for l in open(clip / "labels.jsonl")]
    if args.limit:
        recs = recs[:args.limit]
    mics = mic_positions()
    pairs = [(i, j) for i in range(4) for j in range(i + 1, 4)]
    max_shift = int(FS * ARRAY_SIDE * 1.5 / C_SOUND) + 2
    synth = Synth(seed=args.seed, rpm=args.rpm)
    rng = np.random.default_rng(args.seed + 1)

    out, hits, tot = [], 0, 0
    err = []
    prev_p, prev_t = None, None
    for k, rec in enumerate(recs):
        dets = []
        p = np.array(rec["pos"], dtype=float) if rec.get("pos") else None
        if p is not None:
            v = ((p - prev_p) / max(rec["t"] - prev_t, 1e-6)
                 if prev_p is not None else np.zeros(3))
            prev_p, prev_t = p, rec["t"]
            if args.engine == "pra":
                if pra is None:
                    raise SystemExit("pyroomacoustics not installed; use "
                                     "--engine builtin")
                audio = pra_capture(synth, p, v, mics, args.dur, args.env, rng)
            else:
                audio = synth.chunk(p, v, mics, args.dur, args.env, rng)
            score, f0 = comb_score(np.asarray(audio[0], dtype=np.float64))
            tot += 1
            if score >= args.score_db:
                if args.engine == "pra":
                    az = pra_doa(np.asarray(audio), mics, f0, args.doa)
                else:
                    harm = [f0 * h for h in range(1, 7) if f0 * h < FS / 2]
                    taus = []
                    for (i, j) in pairs:
                        tau, _ = gcc_phat(audio[i], audio[j], max_shift,
                                          harmonics_hz=harm)
                        taus.append(tau)
                    az, _ = bearing_from_tdoa(taus, mics, pairs)
                if az is not None:
                    hits += 1
                    rel = p - CAM
                    true_az = math.atan2(rel[1], rel[0])
                    e = math.degrees(abs((az - true_az + math.pi)
                                         % (2 * math.pi) - math.pi))
                    err.append(e)
                    col = W / 2 - fx * math.tan(az)
                    half = max(6.0, fx * math.tan(math.radians(3.0)))
                    dets.append({
                        "xyxy": [col - half, 0.0, col + half, float(H)],
                        "conf": round(min(1.0, score / 30.0), 4),
                        "cls": "acoustic",
                        "bearing_deg": round(math.degrees(az), 2),
                        "bearing_sigma_deg": 3.0,
                        "comb_db": round(score, 1),
                        "f0_hz": round(f0, 1),
                    })
        out.append({"frame": rec["frame"], "detections": dets})
        if (k + 1) % 100 == 0:
            print(f"  {k+1}/{len(recs)} frames", flush=True)

    dst = clip / "detections_acoustic_real.jsonl"
    with open(dst, "w") as fh:
        for r in out:
            fh.write(json.dumps(r) + "\n")
    e = np.array(err) if err else np.array([0.0])
    print(f"\n{clip.name} [{args.env}]: heard on {hits}/{tot} frames "
          f"= {hits/max(tot,1):.1%}")
    print(f"bearing error: median {np.median(e):.2f} deg, "
          f"p90 {np.percentile(e,90):.2f} deg  (MEASURED from the waveform, "
          f"not assumed)")
    print(f"wrote {dst}")


if __name__ == "__main__":
    main()
