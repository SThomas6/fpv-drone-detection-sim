#!/usr/bin/env python3
"""Labelled 3D drawing of the car-mounted rig, at the sizes the physics sets.

A Gazebo render shows the rig but not the REASONS - everything is grey and
nothing is labelled, so it cannot be reviewed as a design. This draws the
same geometry with each subsystem coloured and dimensioned, so the numbers
that matter for buying parts are visible on the drawing:

  PCL array   5 elements at lambda/2 = 0.25 m for a 600 MHz illuminator.
              Wider spacing ALIASES - measured, 0.335 m gave 36 deg bearing
              error against 4.6 deg at lambda/2.
  Reference   a separate Yagi, NOT one of the array elements. Correlating
              each element against its own reference made the bearing read
              the transmitter instead of the drone.
  Mic array   0.30 m square, the baseline the acoustic pipeline uses.
  Head        pan-tilt: a 6 deg telephoto sees 1% of the sky, so it is
              useless unless something points it.

    python scripts/rig_diagram.py --out docs/rig/rig_diagram.png
"""

from __future__ import annotations

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

C_CAR = "#8d99ae"
C_STRUCT = "#495057"
C_OPTIC = "#1d3557"
C_THERM = "#c1121f"
C_PCL = "#2a9d8f"
C_MIC = "#3a0ca3"
C_REF = "#e07a5f"


def box(ax, c, s, colour, alpha=1.0):
    """Axis-aligned cuboid centred at c with full sizes s."""
    cx, cy, cz = c
    sx, sy, sz = (v / 2 for v in s)
    x = [cx - sx, cx + sx]
    y = [cy - sy, cy + sy]
    z = [cz - sz, cz + sz]
    faces = []
    for i in range(2):
        faces.append([(x[i], y[0], z[0]), (x[i], y[1], z[0]),
                      (x[i], y[1], z[1]), (x[i], y[0], z[1])])
        faces.append([(x[0], y[i], z[0]), (x[1], y[i], z[0]),
                      (x[1], y[i], z[1]), (x[0], y[i], z[1])])
        faces.append([(x[0], y[0], z[i]), (x[1], y[0], z[i]),
                      (x[1], y[1], z[i]), (x[0], y[1], z[i])])
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    ax.add_collection3d(Poly3DCollection(
        faces, facecolors=colour, edgecolors="#00000030", alpha=alpha))


def tube(ax, p0, p1, r, colour, n=14):
    """A cylinder between two points, drawn as a surface."""
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    v = p1 - p0
    L = np.linalg.norm(v)
    if L < 1e-9:
        return
    v = v / L
    notv = np.array([1, 0, 0]) if abs(v[0]) < 0.9 else np.array([0, 1, 0])
    n1 = np.cross(v, notv)
    n1 /= np.linalg.norm(n1)
    n2 = np.cross(v, n1)
    t = np.linspace(0, L, 2)
    th = np.linspace(0, 2 * np.pi, n)
    T, TH = np.meshgrid(t, th)
    X = p0[0] + v[0] * T + r * np.sin(TH) * n1[0] + r * np.cos(TH) * n2[0]
    Y = p0[1] + v[1] * T + r * np.sin(TH) * n1[1] + r * np.cos(TH) * n2[1]
    Z = p0[2] + v[2] * T + r * np.sin(TH) * n1[2] + r * np.cos(TH) * n2[2]
    ax.plot_surface(X, Y, Z, color=colour, shade=True, linewidth=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs/rig/rig_diagram.png")
    args = ap.parse_args()

    fig = plt.figure(figsize=(17, 7.2), facecolor="white")

    for si, (elev, azim, title) in enumerate([
            (16, -58, "three-quarter view"),
            (4, -90, "side elevation  (drone approaches from the right)"),
            (24, 0, "rear view  -  antenna spacing")]):
        ax = fig.add_subplot(1, 3, si + 1, projection="3d")
        ax.set_facecolor("white")

        # --- vehicle, for scale only ---
        # only the roof panel, since that is all that is in frame
        box(ax, (-0.15, 0, 1.36), (2.10, 1.62, 0.16), C_CAR, 0.5)

        # --- roof bars + baseplate ---
        tube(ax, (0.45, -0.65, 1.55), (0.45, 0.65, 1.55), 0.022, C_STRUCT)
        tube(ax, (-0.45, -0.65, 1.55), (-0.45, 0.65, 1.55), 0.022, C_STRUCT)
        box(ax, (0, 0, 1.60), (1.05, 0.62, 0.03), C_STRUCT)
        box(ax, (-0.30, 0, 1.72), (0.36, 0.30, 0.20), "#343a40")

        # --- mast + pan-tilt head ---
        tube(ax, (0.10, 0, 1.63), (0.10, 0, 2.16), 0.028, C_STRUCT)
        tube(ax, (0.10, 0, 2.16), (0.10, 0, 2.25), 0.075, "#212529")
        tube(ax, (0.13, 0, 2.27), (0.40, 0, 2.23), 0.052, C_OPTIC)   # telephoto
        box(ax, (0.16, 0.11, 2.27), (0.09, 0.06, 0.06), C_OPTIC)     # wide
        box(ax, (0.16, -0.11, 2.27), (0.07, 0.05, 0.05), C_THERM)    # thermal

        # --- PCL surveillance array: 5 elements at lambda/2 = 0.25 m ---
        tube(ax, (-0.10, -0.52, 1.68), (-0.10, 0.52, 1.68), 0.018, "#264653")
        for off in (-0.5, -0.25, 0.0, 0.25, 0.5):
            tube(ax, (-0.10, off, 1.69), (-0.10, off, 1.96), 0.008, C_PCL)

        # --- reference Yagi, aimed at the illuminator ---
        tube(ax, (-0.62, 0.30, 1.98), (-0.30, 0.52, 1.98), 0.010, C_REF)
        for k in range(4):
            f = k / 3.0
            bx = -0.62 + 0.32 * f
            by = 0.30 + 0.22 * f
            tube(ax, (bx - 0.06, by + 0.09, 1.98), (bx + 0.06, by - 0.09, 1.98),
                 0.006, C_REF)

        # --- mic array: 0.30 m square ---
        for mx, my in ((0.25, 0.15), (0.25, -0.15), (-0.05, 0.15), (-0.05, -0.15)):
            tube(ax, (mx, my, 1.63), (mx, my, 1.68), 0.012, C_MIC)
        tube(ax, (0.25, -0.15, 1.655), (0.25, 0.15, 1.655), 0.004, C_MIC)
        tube(ax, (-0.05, -0.15, 1.655), (-0.05, 0.15, 1.655), 0.004, C_MIC)
        tube(ax, (-0.05, 0.15, 1.655), (0.25, 0.15, 1.655), 0.004, C_MIC)
        tube(ax, (-0.05, -0.15, 1.655), (0.25, -0.15, 1.655), 0.004, C_MIC)

        if si == 2:      # dimension the antenna spacing on the rear view
            for off in (-0.5, -0.25, 0.0, 0.25, 0.5):
                ax.plot([-0.10, -0.10], [off, off], [2.00, 2.06],
                        color=C_PCL, lw=1)
            ax.text(-0.10, 0.0, 2.16, "5 x lambda/2 = 0.25 m\n(1.0 m aperture)",
                    color=C_PCL, fontsize=8.5, ha="center")
            ax.text(-0.10, -0.62, 1.50, "wider spacing ALIASES",
                    color="#9d0208", fontsize=7.5, ha="center")

        # Frame the RIG, not the vehicle. Showing the whole car makes the
        # sensor head about 60 px tall, which is a picture of a car with
        # something on it rather than a drawing anyone can review.
        ax.set_xlim(-1.15, 1.15); ax.set_ylim(-1.15, 1.15)
        ax.set_zlim(1.30, 2.45)
        ax.set_box_aspect((1, 1, 0.62))
        ax.view_init(elev=elev, azim=azim)
        ax.set_axis_off()
        ax.set_title(title, fontsize=10, color="#333333", pad=0)

    handles = [plt.Line2D([], [], marker="s", ls="", ms=11, color=c, label=l)
               for c, l in [
                   (C_OPTIC, "wide 60 deg + telephoto 6 deg, on pan-tilt"),
                   (C_THERM, "thermal LWIR 24 deg"),
                   (C_PCL, "passive-radar array: 5 el. at 0.25 m"),
                   (C_REF, "reference Yagi -> aimed at the TV/LTE mast"),
                   (C_MIC, "mic array, 0.30 m square"),
                   (C_STRUCT, "roof bars, mast, electronics box")]]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False,
               fontsize=10, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle("Car-roof sensor rig  -  sensor head 2.26 m above ground, "
                 "0.71 m above the roof bars  (car roof shown in grey)",
                 fontsize=12.5, y=0.97)
    fig.tight_layout(rect=(0, 0.09, 1, 0.95))
    fig.savefig(args.out, dpi=125, facecolor="white")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
