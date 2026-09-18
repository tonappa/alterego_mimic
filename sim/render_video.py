#!/usr/bin/env python3
"""Renders frames.pkl (simulate_follower.py) to an MP4: 3D scene with the AlterEgo v2 URDF meshes and the
person, the image seen by the head camera, and the follower state."""
import os
import pickle
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import robot_model  # noqa: E402

RED, BLUE, ORANGE = np.array([0.85, 0.2, 0.2]), np.array([0.2, 0.4, 0.85]), "#e08a1e"
BONES = [(5, 6), (5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16)]
R_ARM, L_ARM = [(6, 8), (8, 10)], [(5, 7), (7, 9)]
LIGHT = np.array([0.35, -0.55, 0.75]) / np.linalg.norm([0.35, -0.55, 0.75])


def recolor(robot):
    """Robot LEFT arm red, RIGHT arm blue (mirror mode: your right arm -> its left arm)."""
    parts = []
    for link, V, F, c in robot.parts:
        if link.startswith("left_") and "hand" not in link:
            c = RED * (0.75 if "cube" in link else 1.0)
        elif link.startswith("right_") and "hand" not in link:
            c = BLUE * (0.75 if "cube" in link else 1.0)
        parts.append((link, V, F, c))
    robot.parts = parts


def draw_scene(ax, robot, f):
    ax.cla()
    poses = robot.link_poses(f["q_r"], f["q_l"], f["yaw"], f["pitch"], (f["base_x"], f.get("base_y", 0.0)),
                             f.get("base_yaw", 0.0))
    tris, cols = robot.triangles(poses)
    n = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
    shade = 0.35 + 0.65 * np.abs(n @ LIGHT)
    ax.add_collection3d(Poly3DCollection(tris, facecolors=np.clip(cols * shade[:, None], 0, 1), edgecolor="none"))

    # floor grid
    for x in np.arange(-1.0, 3.6, 0.5):
        ax.plot([x, x], [-1.3, 1.3], [0, 0], color="0.85", lw=0.6)
    for y in np.arange(-1.0, 1.31, 0.5):
        ax.plot([-1.0, 3.5], [y, y], [0, 0], color="0.85", lw=0.6)

    # person
    k = f["kw"]
    for a, b in BONES:
        ax.plot(*zip(k[a], k[b]), color=ORANGE, lw=4, solid_capstyle="round")
    for bones, col in ((R_ARM, RED), (L_ARM, BLUE)):
        for a, b in bones:
            ax.plot(*zip(k[a], k[b]), color=col, lw=5, solid_capstyle="round")
    neck_top = (k[3] + k[4]) / 2
    ax.plot(*zip((k[5] + k[6]) / 2, neck_top - np.array([0, 0, 0.09])), color=ORANGE, lw=4)
    ax.scatter(*neck_top, s=380, color=ORANGE, depthshade=False)
    ax.scatter(*k[0], s=25, color="k", depthshade=False)

    # camera: optical axis and field of view
    T = f["T_cam"]
    o = T[:3, 3]
    z = T[:3, 2]
    ax.plot(*zip(o, o + z * 1.2), color="green", lw=1.2)
    for u, v in ((0, 0), (640, 0), (640, 480), (0, 480)):
        ray = T[:3, :3] @ np.array([(u - 320) / 615, (v - 240) / 615, 1.0])
        ax.plot(*zip(o, o + ray * 0.9), color="green", lw=0.5, alpha=0.6)

    # distance on the floor
    tx = (k[5] + k[6]) / 2
    by = f.get("base_y", 0.0)
    ax.plot([f["base_x"], tx[0]], [by, tx[1]], [0.01, 0.01], "k--", lw=1)
    if f["d"] is not None:
        mid = ((f["base_x"] + tx[0]) / 2, (by + tx[1]) / 2, 0.05)
        ax.text(*mid, "%.2f m" % f["d"], fontsize=9, ha="center")

    ax.set_xlim(-1.0, 3.3); ax.set_ylim(-1.2, 1.2); ax.set_zlim(0, 1.9)
    ax.set_box_aspect((4.3, 2.4, 1.9), zoom=1.45)
    ax.view_init(elev=14, azim=-62)
    ax.set_axis_off()


def draw_camera(ax, f):
    ax.cla()
    view = f["view"]
    ax.set_xlim(0, 640); ax.set_ylim(480, 0); ax.set_aspect("equal")
    ax.set_facecolor("#1d1f24")
    ax.set_xticks([]); ax.set_yticks([])
    u, v, vis = view["u"], view["v"], view["vis"]
    for bones, col in ((BONES, ORANGE), (R_ARM, RED), (L_ARM, BLUE)):
        for a, b in bones:
            if vis[a] and vis[b]:
                ax.plot([u[a], u[b]], [v[a], v[b]], color=col, lw=3)
    ax.scatter(u[vis], v[vis], s=12, color="w", zorder=3)
    ax.plot([310, 330], [240, 240], color="w", lw=1); ax.plot([320, 320], [230, 250], color="w", lw=1)
    fu = view.get("face_uv")
    if fu is not None:
        est = view.get("face_est")
        ax.scatter([fu[0]], [fu[1]], marker="+", s=260, color="yellow" if not est else "magenta", lw=2, zorder=4)
        ax.text(8, 470, "face estimated above the shoulders (out of view)" if est else "face",
                color="magenta" if est else "yellow", fontsize=8)
    ax.set_title("What the head camera sees (640x480)", fontsize=10)


def draw_info(ax, f):
    ax.cla(); ax.set_axis_off()
    state = "FOLLOWING" if f["active"] else "IDLE"
    lines = [
        ("State", state),
        ("Distance", "%s  (start %s)" % ("%.2f m" % f["d"] if f["d"] is not None else "--",
                                                          "%.2f m" % f["d_ref"] if f["d_ref"] else "--")),
        ("Base velocity", "%+.2f m/s" % f["v"]),
        ("Neck yaw / pitch", "%+.0f / %+.0f deg  (pitch < 0 = up)" % (np.degrees(f["yaw"]), np.degrees(f["pitch"]))),
        ("Right arm q0 q1 q3", "%4.0f %4.0f %4.0f deg" % tuple(np.degrees(f["q_r"][[0, 1, 3]]))),
        ("Left arm  q0 q1 q3", "%4.0f %4.0f %4.0f deg" % tuple(np.degrees(f["q_l"][[0, 1, 3]]))),
    ]
    y = 0.95
    for k_, v_ in lines:
        ax.text(0.0, y, k_, fontsize=9.5, color="0.35", transform=ax.transAxes, family="monospace")
        ax.text(0.42, y, v_, fontsize=9.5, transform=ax.transAxes, family="monospace",
                color=("green" if v_ == "FOLLOWING" else "k"), weight=("bold" if k_ == "State" else "normal"))
        y -= 0.12
    ax.text(0.0, 0.12, "Your RIGHT arm (red) moves its LEFT arm (red): mirror mode", fontsize=8.5, transform=ax.transAxes)
    ax.text(0.0, 0.0, "Kinematic simulation: real person_follower.py code, simulated camera\n"
                      "(15 Hz, 0.15 s delay, depth noise), ideal actuators, no balance dynamics",
            fontsize=7.5, color="0.4", transform=ax.transAxes, va="bottom")


def main(src=os.path.join(HERE, "frames.pkl"), out=os.path.join(HERE, "alterego_person_follower_sim.mp4"),
         only=None):
    D = pickle.load(open(src, "rb"))
    frames = D["frames"] if only is None else [D["frames"][i] for i in only]
    robot = robot_model.Robot()
    recolor(robot)
    fig = plt.figure(figsize=(12.8, 7.2), dpi=100)
    ax3d = fig.add_axes([0.0, 0.0, 0.62, 0.9], projection="3d")
    axcam = fig.add_axes([0.63, 0.40, 0.35, 0.47])
    axinfo = fig.add_axes([0.64, 0.02, 0.35, 0.36])
    title = fig.text(0.02, 0.95, "", fontsize=15, weight="bold")
    clock = fig.text(0.98, 0.95, "", fontsize=12, ha="right", family="monospace")
    writer = FFMpegWriter(fps=D["fps"], bitrate=2500, codec="libx264", extra_args=["-pix_fmt", "yuv420p"])
    with writer.saving(fig, out, dpi=100):
        for i, f in enumerate(frames):
            draw_scene(ax3d, robot, f)
            draw_camera(axcam, f)
            draw_info(axinfo, f)
            title.set_text(f["caption"])
            clock.set_text("t = %5.1f s" % f["t"])
            writer.grab_frame()
            if i % 50 == 0:
                print("frame %d / %d" % (i, len(frames)), flush=True)
    print("written", out)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        main(out=os.path.join(HERE, "test.mp4"), only=[0, 100, 280, 300, 390, 450])
    else:
        main()
