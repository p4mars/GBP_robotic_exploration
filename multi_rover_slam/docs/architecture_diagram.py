"""Draw the offline/online multi-rover SLAM architecture as a PNG for slides.

    python3 multi_rover_slam/docs/architecture_diagram.py   ->  multi_rover_slam/docs/slam_architecture.png
"""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = Path(__file__).with_name("slam_architecture.png")
W, H = 34, 18

STYLE = {
    "ours": dict(fc="#dbe8fb", ec="#2f6fc9"),
    "upstream": dict(fc="#eceef1", ec="#7c838c"),
    "modified": dict(fc="#fde3c8", ec="#e07b12"),
    "planned": dict(fc="#ffffff", ec="#2f6fc9", ls=(0, (5, 3))),
    "data": dict(fc="#fbf5d9", ec="#b59a2a"),
    "contract": dict(fc="#e6f4ea", ec="#2e8b57"),
}
TEXT = "#1f2328"

fig = plt.figure(figsize=(W / 2, H / 2), dpi=200)
ax = fig.add_axes((0, 0, 1, 1))
ax.set_xlim(0, W)
ax.set_ylim(0, H)
ax.axis("off")
fig.patch.set_facecolor("white")


def frame(x0, y0, x1, y1, label, ec="#9aa1a9", fc="none", ls="-", size=13, lw=1.6, z=0):
    ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0, boxstyle="round,pad=0,rounding_size=0.35",
                                fc=fc, ec=ec, lw=lw, ls=ls, zorder=z))
    ax.text(x0 + 0.3, y1 - 0.3, label, ha="left", va="top", fontsize=size, weight="bold", color=ec, zorder=z + 0.1)


def box(x0, y0, x1, y1, title, sub="", kind="ours", size=11, z=2):
    s = STYLE[kind]
    ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0, boxstyle="round,pad=0,rounding_size=0.2",
                                fc=s["fc"], ec=s["ec"], lw=1.8, ls=s.get("ls", "-"), zorder=z))
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    if sub:
        ax.text(cx, cy + 0.12, title, ha="center", va="bottom", fontsize=size, weight="bold", color=TEXT, zorder=z + 1)
        ax.text(cx, cy - 0.05, sub, ha="center", va="top", fontsize=size - 2, color=TEXT, zorder=z + 1,
                linespacing=1.35)
    else:
        ax.text(cx, cy, title, ha="center", va="center", fontsize=size, weight="bold", color=TEXT, zorder=z + 1)


def arrow(p, q, label="", rad=0.0, ls="-", color="#4a5058", both=False, lpos=0.5, loff=(0, 0.22), lsize=9, rot=0):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="<|-|>" if both else "-|>", mutation_scale=14, lw=1.6,
                                 color=color, ls=ls, connectionstyle=f"arc3,rad={rad}", zorder=5,
                                 shrinkA=2, shrinkB=2))
    if label:
        x = p[0] + (q[0] - p[0]) * lpos + loff[0]
        y = p[1] + (q[1] - p[1]) * lpos + loff[1]
        ax.text(x, y, label, ha="center", va="center", fontsize=lsize, color=color, style="italic", rotation=rot,
                zorder=6, bbox=dict(fc="white", ec="none", pad=1.0, alpha=0.9))


def polyline(points, ls="-", color="#4a5058", label="", lxy=None):
    xs, ys = zip(*points[:-1])
    ax.plot(xs, ys, color=color, lw=1.6, ls=ls, zorder=5, solid_capstyle="round")
    arrow(points[-2], points[-1], ls=ls, color=color)
    if label:
        ax.text(*lxy, label, ha="center", va="center", fontsize=9, color=color, style="italic", zorder=6,
                bbox=dict(fc="white", ec="none", pad=1.0, alpha=0.9))


# ---------------------------------------------------------------- title + legend
ax.text(0.4, 17.55, "Multi-rover collaborative SLAM: offline replay (today) and online mode (planned)",
        ha="left", va="center", fontsize=17, weight="bold", color=TEXT)
lx = 0.4
for kind, label in [("ours", "our code"), ("upstream", "upstream, unchanged"), ("modified", "upstream, we modify ★"),
                    ("planned", "planned"), ("data", "data on disk"), ("contract", "ROS 2 topic interface")]:
    s = STYLE[kind]
    ax.add_patch(FancyBboxPatch((lx, 16.75), 0.8, 0.42, boxstyle="round,pad=0,rounding_size=0.08",
                                fc=s["fc"], ec=s["ec"], lw=1.5, ls=s.get("ls", "-")))
    ax.text(lx + 1.0, 16.96, label, ha="left", va="center", fontsize=10, color=TEXT)
    lx += 1.3 + 0.17 * len(label) + 0.6

# ---------------------------------------------------------------- SRB container
frame(0.4, 5.0, 9.6, 16.3, "SRB container  (Isaac Sim)")
box(0.9, 14.0, 9.1, 15.4, "Isaac Sim + Space Robotics Bench", "physics · rendering · lunar terrain", "upstream")
box(0.9, 8.9, 9.1, 13.4, "multi_rover_setup",
    "config presets (YAML)\nterrain + rover layout\nMotionController (orbit / follow / wander)\nRGB-D camera per rover\nrecorder", "ours")
box(5.0, 5.6, 9.1, 7.6, "live publisher", "rclpy, same topics", "planned", size=10)
arrow((5.0, 13.95), (5.0, 13.45))

# run folder (on disk, shared by both containers through the mounted repo)
box(0.9, 1.0, 4.4, 3.6, "run folder", "RGB · depth (png16)\nintrinsics K\nground-truth poses", "data", size=10)
arrow((2.65, 8.85), (2.65, 3.65), "OFFLINE", loff=(0, 0), lpos=0.62)
arrow((7.05, 8.85), (7.05, 7.65), "ONLINE", ls="--", color="#2f6fc9", loff=(0, 0), lpos=0.5)

# ---------------------------------------------------------------- Swarm-SLAM container
frame(10.4, 0.6, 33.6, 16.3, "Swarm-SLAM container  (ROS 2 Jazzy)")

# offline replay column
box(11.0, 1.2, 15.9, 2.8, "run_to_bags", "one rosbag per rover", "ours", size=10)
box(11.0, 3.6, 15.9, 5.0, "rosbags", "bags/r<id>  (MCAP)", "data", size=10)
box(11.0, 5.8, 15.9, 7.2, "ros2 bag play", "all rovers in sync", "upstream", size=10)
arrow((4.45, 2.0), (10.95, 2.0))
arrow((13.45, 2.85), (13.45, 3.55))
arrow((13.45, 5.05), (13.45, 5.75))
arrow((13.45, 7.25), (13.45, 8.65))

# topic interface
box(11.0, 8.7, 15.9, 12.3, "topic interface", "per rover /r<id>/\ncolor/image\ndepth/image\ncolor/camera_info\nodom (optional)",
    "contract", size=10)
polyline([(9.1, 6.6), (10.0, 6.6), (10.0, 10.5), (10.95, 10.5)], ls="--", color="#2f6fc9")

# rover r0 stack (peers drawn behind as cards)
frame(17.0, 4.3, 29.0, 15.5, "rover r0   (namespace /r0)", ec="#5b6470", fc="#fafbfc", size=12)
box(17.5, 12.7, 22.5, 14.4, "rgbd_odometry", "RTAB-Map visual odometry", "upstream", size=10)
box(23.5, 12.7, 28.5, 14.4, "CosPlace", "ResNet-18 → 64-d place descriptor", "upstream", size=10)
ax.add_patch(FancyBboxPatch((17.3, 9.1), 11.4, 3.2, boxstyle="round,pad=0,rounding_size=0.2",
                            fc="none", ec="#9aa1a9", lw=1.0, ls=(0, (2, 2)), zorder=1))
ax.text(17.45, 9.25, "FRONT END", ha="left", va="bottom", fontsize=8.5, weight="bold", color="#7c838c")
box(17.5, 9.8, 22.5, 11.9, "map_manager", "C++ · keyframes\nPnP geometric verification", "upstream", size=10)
box(23.5, 9.8, 28.5, 11.9, "loop_closure_detection", "Python · descriptor matching\ncandidate selection (budget)",
    "upstream", size=10)
box(17.5, 5.0, 28.5, 8.2, "BACK END ★   pose_graph_manager",
    "decentralized pose-graph optimization (GTSAM)\nmerges the rovers' maps · to be modified", "modified", size=11)

# data flow inside the rover
arrow((15.95, 11.2), (17.45, 13.3), "images", loff=(-0.2, 0.25))
arrow((15.95, 10.6), (17.45, 10.6))
ax.text(16.7, 10.05, "images\n+ odom", ha="center", va="center", fontsize=8, color="#4a5058", style="italic")
arrow((20.0, 12.65), (20.0, 11.95), "odom", loff=(0.55, 0))
arrow((22.55, 11.25), (23.45, 11.25), both=True)
ax.text(23.0, 11.5, "match", ha="center", va="bottom", fontsize=8, color="#4a5058", style="italic")
ax.text(23.0, 11.0, "verify", ha="center", va="top", fontsize=8, color="#4a5058", style="italic")
arrow((26.0, 12.65), (26.0, 11.95), both=True, label="descriptors", loff=(0.95, 0))
arrow((20.0, 9.75), (20.0, 8.25), "odometry + keyframes", lpos=0.72, loff=(0, 0))
arrow((26.0, 9.75), (26.0, 8.25), "loop closures", lpos=0.72, loff=(0, 0))

# peers r1 … rN
for i, dx in enumerate((0.5, 0.25, 0.0)):
    ax.add_patch(FancyBboxPatch((31.0 + dx, 4.3 + dx), 2.0, 11.2 - 0.5, boxstyle="round,pad=0,rounding_size=0.3",
                                fc="#fafbfc", ec="#5b6470", lw=1.4, zorder=2 + i))
ax.text(32.0, 9.8, "rovers r1 … rN\n(same stack)", ha="center", va="center", rotation=90, fontsize=11.5,
        weight="bold", color="#5b6470", zorder=6)
arrow((29.05, 9.8), (30.95, 9.8), both=True, color="#e07b12")
ax.text(30.0, 12.9, "inter-robot\nexchange", ha="center", va="center", fontsize=9, weight="bold", color="#e07b12")
ax.text(30.0, 7.0, "descriptors\nmatches\nloop closures\npose graphs\nheartbeats", ha="center", va="center",
        fontsize=8, color="#e07b12", style="italic", linespacing=1.4)

# outputs + evaluation
box(17.5, 1.2, 22.5, 3.2, "cslam logs", "pose graphs (g2o)\nkeyframe timestamps", "data", size=10)
box(23.5, 1.2, 28.5, 3.2, "eval_slam", "ATE vs. ground truth\ntrajectory + loop-closure plots", "ours", size=10)
arrow((20.0, 4.95), (20.0, 3.25))
arrow((22.55, 2.2), (23.45, 2.2))
polyline([(2.65, 0.95), (2.65, 0.3), (26.0, 0.3), (26.0, 1.15)], label="ground truth", lxy=(8.0, 0.3))

fig.savefig(OUT, dpi=200, facecolor="white")
print(OUT)
