#!/usr/bin/env python3
"""Build a car-mountable version of the rig, with cameras to photograph it.

Every earlier picture in this project is the view FROM the station. This
builds the station itself, on a vehicle roof, at the sizes the physics
elsewhere in the project implies - so the drawing cannot quietly disagree
with the numbers:

  PCL array   five elements at lambda/2 = 0.25 m (600 MHz) -> 1.0 m aperture.
              Wider spacing aliases; this was measured, not assumed.
  Reference   a separate small Yagi, NOT part of the array. Correlating each
              array element against its own reference made the bearing read
              the illuminator instead of the drone (36 deg error).
  Mic array   4 mics on a 0.30 m square, the baseline the acoustic pipeline
              actually uses.
  Head        pan-tilt carrying wide + 6 deg telephoto + thermal, because a
              6 deg lens sees 1% of the sky and only works when pointed.

Sized to standard 0.9 m roof-bar spacing and kept low enough that the whole
vehicle still clears a domestic garage door.

    python scripts/gen_rig_showcase.py
"""

from __future__ import annotations

import pathlib

WORLDS = pathlib.Path(__file__).resolve().parents[1] / "simulator" / "worlds"


def link(name, pose, geom, rgb, vis_pose=None):
    vp = f"<pose>{vis_pose}</pose>" if vis_pose else ""
    r, g, b = rgb
    return (f'    <link name="{name}"><pose>{pose}</pose>\n'
            f'      <visual name="v">{vp}<geometry>{geom}</geometry>\n'
            f'        <material><ambient>{r} {g} {b} 1</ambient>'
            f'<diffuse>{r} {g} {b} 1</diffuse></material></visual></link>\n')


def box(x, y, z):
    return f"<box><size>{x} {y} {z}</size></box>"


def cyl(r, l):
    return f"<cylinder><radius>{r}</radius><length>{l}</length></cylinder>"


def main():
    src = (WORLDS / "detection_world.sdf").read_text(encoding="utf-8")

    rig = ['  <model name="car_rig"><static>true</static>\n']
    # roof bars
    rig.append(link("bar_front", "0.45 0 1.55 0 0 1.5708", cyl(0.022, 1.30),
                    (0.15, 0.15, 0.16)))
    rig.append(link("bar_rear", "-0.45 0 1.55 0 0 1.5708", cyl(0.022, 1.30),
                    (0.15, 0.15, 0.16)))
    rig.append(link("plate", "0 0 1.60 0 0 0", box(1.05, 0.62, 0.03),
                    (0.25, 0.26, 0.28)))
    rig.append(link("ebox", "-0.30 0 1.72 0 0 0", box(0.36, 0.30, 0.20),
                    (0.20, 0.21, 0.23)))
    rig.append(link("mast", "0.10 0 1.95 0 0 0", cyl(0.028, 0.42),
                    (0.45, 0.45, 0.47)))
    # pan-tilt head: wide + telephoto + thermal
    rig.append(link("head_yoke", "0.10 0 2.20 0 0 0", cyl(0.075, 0.09),
                    (0.18, 0.18, 0.20)))
    rig.append(link("tele", "0.24 0 2.26 0 -0.15 0", cyl(0.052, 0.30),
                    (0.10, 0.10, 0.12), vis_pose="0 0 0 0 1.5708 0"))
    rig.append(link("wide", "0.20 0.10 2.26 0 -0.15 0", box(0.09, 0.06, 0.06),
                    (0.12, 0.12, 0.14)))
    rig.append(link("thermal", "0.20 -0.10 2.26 0 -0.15 0",
                    box(0.07, 0.05, 0.05), (0.45, 0.15, 0.12)))
    # passive-radar surveillance array
    rig.append(link("pcl_rail", "-0.10 0 1.68 0 0 1.5708", box(0.04, 1.02, 0.03),
                    (0.22, 0.24, 0.26)))
    for i, off in enumerate((-0.5, -0.25, 0.0, 0.25, 0.5)):
        rig.append(link(f"pcl_el{i}", f"-0.10 {off} 1.83 0 0 0",
                        cyl(0.006, 0.27), (0.78, 0.78, 0.80)))
    # reference antenna, aimed at the illuminator, separate from the array
    rig.append(link("ref_boom", "-0.40 0.34 1.95 0 0 2.44", cyl(0.010, 0.50),
                    (0.58, 0.58, 0.60), vis_pose="0 0 0 0 1.5708 0"))
    for k in range(4):
        rig.append(link(f"ref_el{k}", f"{-0.40 + 0.12 * k - 0.18} "
                                      f"{0.34 + 0.10} 1.95 1.5708 0 0",
                        cyl(0.005, 0.20), (0.58, 0.58, 0.60)))
    # microphone array
    rig.append(link("mic_x", "0.10 0 1.66 0 0 0", box(0.30, 0.014, 0.014),
                    (0.30, 0.30, 0.33)))
    rig.append(link("mic_y", "0.10 0 1.66 0 0 0", box(0.014, 0.30, 0.014),
                    (0.30, 0.30, 0.33)))
    for i, (mx, my) in enumerate(((0.25, 0.15), (0.25, -0.15),
                                  (-0.05, 0.15), (-0.05, -0.15))):
        rig.append(link(f"mic{i}", f"{mx} {my} 1.67 0 0 0", cyl(0.011, 0.03),
                        (0.12, 0.12, 0.45)))
    rig.append("  </model>\n")

    car = ['  <model name="car"><static>true</static>\n']
    car.append(link("body", "0 0 0.75 0 0 0", box(4.30, 1.80, 0.80),
                    (0.33, 0.37, 0.46)))
    car.append(link("cabin", "-0.15 0 1.28 0 0 0", box(2.10, 1.62, 0.62),
                    (0.22, 0.25, 0.33)))
    for i, (wx, wy) in enumerate(((1.35, 0.92), (1.35, -0.92),
                                  (-1.35, 0.92), (-1.35, -0.92))):
        car.append(link(f"wheel{i}", f"{wx} {wy} 0.34 1.5708 0 0",
                        cyl(0.34, 0.22), (0.07, 0.07, 0.07)))
    car.append("  </model>\n")

    views = []
    # Framed on the RIG, not the vehicle: the first attempt sat 6-8 m back
    # with a 54 deg lens and the whole sensor head came out 120 px tall,
    # which shows a car with something on the roof rather than showing the
    # rig. These sit ~3 m out at rig height with a longer lens.
    for i, (x, y, z, pitch, yaw) in enumerate([
            (2.9, -2.4, 2.55, 0.10, 2.47),    # three-quarter, close
            (0.10, -3.1, 2.05, -0.02, 1.5708),  # side elevation
            (3.1, 0.05, 2.75, 0.16, 3.1416)]):  # front, down the boresight
        views.append(
            f'  <model name="view{i}"><static>true</static>\n'
            f'    <link name="l"><pose>{x} {y} {z} 0 {pitch} {yaw}</pose>\n'
            f'      <sensor name="c{i}" type="camera">'
            f'<topic>/rig/view{i}</topic>\n'
            f'        <update_rate>4</update_rate><always_on>true</always_on>\n'
            f'        <camera><horizontal_fov>0.75</horizontal_fov>\n'
            f'          <image><width>1280</width><height>800</height>'
            f'<format>R8G8B8</format></image>\n'
            f'          <clip><near>0.05</near><far>200</far></clip></camera>\n'
            f'      </sensor></link></model>\n')

    body = "".join(rig) + "".join(car) + "".join(views)
    out = src.replace("</world>", body + "\n  </world>")
    dst = WORLDS / "rig_showcase.sdf"
    dst.write_text(out, encoding="utf-8")
    print(f"wrote {dst.name}: {out.count('<model name=')} models, "
          f"3 viewpoints")
    print("sensor head sits 2.26 m up on a 1.55 m roof bar; "
          "PCL aperture 1.0 m; mic baseline 0.30 m")


if __name__ == "__main__":
    main()
