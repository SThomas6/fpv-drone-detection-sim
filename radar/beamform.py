#!/usr/bin/env python3
"""Digital beamforming: find the illuminator and null it, in software.

Why this exists. A passive radar needs two things from the antenna: a clean
copy of what the transmitter is sending (the REFERENCE), and a view of the
airspace with that transmitter suppressed (the SURVEILLANCE channel). The
obvious way is two antennas, one of them a Yagi physically aimed at the mast.

On a fixed site that is fine - you aim it once. On a VEHICLE it is not: park
facing a different way and the Yagi points at nothing. Motorising it means a
rotator, a compass, a GPS and a transmitter database, all to solve a problem
that a coherent array already solves without moving.

With N coherent elements both channels are just weighted sums of the same
samples:

    reference     steer the beam AT the transmitter - all elements added in
                  phase for that direction, so the direct path adds
                  coherently and everything else partially cancels
    surveillance  project the snapshot off the transmitter's steering vector
                  - a spatial NULL in that direction, which suppresses the
                  direct path before any temporal cancellation runs

And the transmitter's direction does not need to be configured, which is the
part that makes it automatic. The direct path is by far the strongest thing
the array hears - typically ~90 dB above the echo - so scanning the beam and
taking the peak finds it. Re-park the vehicle and the scan simply returns a
different angle.

It also removes a cheat. `radar/caf.py` currently takes its reference from a
separate FERS run with no target in the scene, which no real receiver can
have. Beamforming derives both channels from the same snapshot, so the chain
becomes self-contained and physically honest.

Aliasing caveat, stated rather than hidden: elements must be spaced at or
under lambda/2 or the scan has grating lobes and the "peak" may be a false
angle. At 600 MHz that is 0.25 m, which is what the rig is drawn to.
"""

from __future__ import annotations

import math

import numpy as np


def steering(angles_rad, n_elem: int, spacing_m: float, lam: float):
    """Steering matrix: rows are angles, columns are elements."""
    pos = (np.arange(n_elem) - (n_elem - 1) / 2.0) * spacing_m
    # Sign fixed against ground truth, because getting it wrong is silent:
    # the scan still finds a sharp peak of the right MAGNITUDE and only the
    # sign is wrong, which reads as a plausible bearing on the wrong side.
    # Measured -38.75 deg where the transmitter was at +38.66 deg.
    return np.exp(2j * math.pi * np.outer(np.sin(angles_rad), pos) / lam)


def scan_power(x: np.ndarray, spacing_m: float, lam: float,
               n_angles: int = 721):
    """Beam power against angle, for a snapshot matrix x of shape (elem, n)."""
    angles = np.linspace(-math.pi / 2, math.pi / 2, n_angles)
    a = steering(angles, x.shape[0], spacing_m, lam)
    # covariance once, then a quadratic form per angle: far cheaper than
    # beamforming the whole time series at every candidate angle
    R = (x @ x.conj().T) / x.shape[1]
    p = np.einsum("ij,jk,ik->i", a.conj(), R, a).real
    return angles, p


def find_illuminator(x: np.ndarray, spacing_m: float, lam: float):
    """Direction of the strongest arrival - the transmitter, automatically.

    No configuration, no compass, no database. The direct path dominates the
    array by ~90 dB, so the beam-power peak IS the illuminator. Returns
    (azimuth, peak-to-median ratio in dB); the ratio is the confidence that
    there is a dominant source at all, so a site with no usable transmitter
    reports a low number rather than a confident wrong angle.
    """
    angles, p = scan_power(x, spacing_m, lam)
    k = int(np.argmax(p))
    ratio = 10 * math.log10(p[k] / (float(np.median(p)) + 1e-30))
    return float(angles[k]), ratio


def reference_beam(x: np.ndarray, az: float, spacing_m: float, lam: float):
    """Sum the elements in phase for `az`: the clean copy of the transmission."""
    a = steering(np.array([az]), x.shape[0], spacing_m, lam)[0]
    return (a.conj() @ x) / x.shape[0]


def null_illuminator(x: np.ndarray, az: float, spacing_m: float, lam: float):
    """Project every snapshot off the transmitter's steering vector.

    This is the spatial half of direct-signal suppression, and it happens
    before the temporal canceller sees anything. It costs one degree of
    freedom out of N, and a target at exactly the transmitter's bearing is
    nulled with it - a real blind direction, and the same one the baseline
    geometry already makes useless.
    """
    a = steering(np.array([az]), x.shape[0], spacing_m, lam)[0]
    a = a / np.linalg.norm(a)
    proj = np.eye(x.shape[0], dtype=np.complex128) - np.outer(a, a.conj())
    return proj @ x


def beam_at(x: np.ndarray, az: float, spacing_m: float, lam: float):
    """Steer a beam at an arbitrary direction (used to sweep the airspace)."""
    return reference_beam(x, az, spacing_m, lam)
