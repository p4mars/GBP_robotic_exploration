"""Read cslam's logs back and compare them with the recorded ground truth.

cslam (``evaluation.enable_logs``) writes, each time a robot receives optimized
estimates, ``<log_folder>/<date>_experiment_robot_<id>/<date>/`` containing
``pose_timestamps<id>.csv`` (vertex key -> odometry stamp) and, on the robot
that ran the optimization, ``initial_global_pose_graph.g2o`` (odometry only)
and ``optimized_global_pose_graph.g2o``. Vertex keys are GTSAM
``LabeledSymbol('g', 'A' + robot_id, keyframe_id)``, but the g2o files drop the
``'g'`` byte while ``pose_timestamps*.csv`` keeps it, so both readers pass keys
through :func:`normalize_key`.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from multi_rover_setup.cameras import quat_xyzw_to_matrix

from .run_data import CameraStream

_INDEX_BITS = 48


def decode_key(key: int) -> tuple[int, int]:
    """GTSAM ``LabeledSymbol`` key -> ``(robot_id, keyframe_id)``."""
    label = (key >> _INDEX_BITS) & 0xFF
    return label - ord("A"), key & ((1 << _INDEX_BITS) - 1)


def normalize_key(key: int) -> int:
    """Canonical key with the ``'g'`` char byte, whether or not the source wrote it."""
    return (ord("g") << 56) | (key & ((1 << 56) - 1))


def encode_key(robot_id: int, keyframe_id: int) -> int:
    return (ord("g") << 56) | ((ord("A") + robot_id) << _INDEX_BITS) | keyframe_id


@dataclass
class PoseGraph:
    vertices: dict[int, np.ndarray] = field(default_factory=dict)
    """key -> ``(4, 4)`` pose"""
    edges: list[tuple[int, int]] = field(default_factory=list)

    def inter_robot_edges(self) -> list[tuple[int, int]]:
        """Edges between keyframes of different robots, i.e. inter-robot loop closures."""
        return [(a, b) for a, b in self.edges if decode_key(a)[0] != decode_key(b)[0]]


def read_g2o(path: str | Path) -> PoseGraph:
    """Parse ``VERTEX_SE3:QUAT`` / ``EDGE_SE3:QUAT`` lines (as written by ``gtsam.writeG2o``)."""
    graph = PoseGraph()
    for line in Path(path).read_text().splitlines():
        parts = line.split()
        if not parts:
            continue
        if parts[0] == "VERTEX_SE3:QUAT":
            x, y, z, qx, qy, qz, qw = map(float, parts[2:9])
            pose = np.eye(4)
            pose[:3, :3] = quat_xyzw_to_matrix((qx, qy, qz, qw))
            pose[:3, 3] = (x, y, z)
            graph.vertices[normalize_key(int(parts[1]))] = pose
        elif parts[0] == "EDGE_SE3:QUAT":
            graph.edges.append((normalize_key(int(parts[1])), normalize_key(int(parts[2]))))
    return graph


def read_pose_timestamps(path: str | Path) -> dict[int, float]:
    """``pose_timestamps<id>.csv`` -> {vertex key: stamp [s]}."""
    with open(path, newline="") as f:
        return {normalize_key(int(row["vertice_id"])): int(row["sec"]) + int(row["nanosec"]) * 1e-9 for row in csv.DictReader(f)}


def latest_logs(slam_dir: str | Path) -> tuple[Path, dict[int, float]]:
    """Folder of the newest merged graph, plus the newest keyframe stamps of every robot."""
    slam_dir = Path(slam_dir)
    graphs = sorted(slam_dir.rglob("optimized_global_pose_graph.g2o"), key=lambda p: p.stat().st_mtime)
    if not graphs:
        raise FileNotFoundError(f"No optimized_global_pose_graph.g2o under {slam_dir} (did a loop closure merge the maps?)")
    stamps: dict[int, float] = {}
    newest: dict[str, Path] = {}
    for path in slam_dir.rglob("pose_timestamps*.csv"):
        if path.name not in newest or path.stat().st_mtime > newest[path.name].stat().st_mtime:
            newest[path.name] = path
    for path in newest.values():  # stamps accumulate over a run, so the newest file is the fullest
        stamps.update(read_pose_timestamps(path))
    return graphs[-1].parent, stamps


def umeyama(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Rigid ``(4, 4)`` transform minimising ``|T @ src - dst|`` for points ``(N, 3)`` (no scale)."""
    mu_s, mu_d = src.mean(axis=0), dst.mean(axis=0)
    u, _, vt = np.linalg.svd((dst - mu_d).T @ (src - mu_s))
    sign = np.diag([1.0, 1.0, np.sign(np.linalg.det(u @ vt))])
    T = np.eye(4)
    T[:3, :3] = u @ sign @ vt
    T[:3, 3] = mu_d - T[:3, :3] @ mu_s
    return T


def transform_points(T: np.ndarray, points: np.ndarray) -> np.ndarray:
    return points @ T[:3, :3].T + T[:3, 3]


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.sum((a - b) ** 2, axis=-1)))) if len(a) else float("nan")


@dataclass
class RobotTrack:
    """Keyframes of one robot that have a stamp, in keyframe order."""

    robot_id: int
    keys: list[int]
    gt: np.ndarray
    """``(K, 3)`` ground-truth positions"""
    odometry: np.ndarray
    """``(K, 3)`` initial (odometry-only) positions, aligned to GT per robot"""
    optimized: np.ndarray
    """``(K, 3)`` optimized positions, aligned to GT jointly for all robots; empty if not merged"""
    merged: bool = True
    """False if cslam never merged this robot into the graph; ``gt``/``odometry`` then come
    from the recorded run and the bag's ``/r<id>/odom`` (per frame instead of per keyframe)."""

    @property
    def ate_odometry(self) -> float:
        return rmse(self.odometry, self.gt)

    @property
    def ate_optimized(self) -> float:
        return rmse(self.optimized, self.gt)


@dataclass
class Evaluation:
    log_dir: Path | None
    """Folder of the merged graph; None if no robot was ever merged."""
    tracks: list[RobotTrack]
    loop_closures: list[tuple[int, int]]
    """Inter-robot loop-closure edges (vertex keys) of the optimized graph."""
    stamps: dict[int, float]
    optimized: PoseGraph
    alignment: np.ndarray
    """Transform applied to all optimized positions."""


def read_bag_odometry(bag_dir: str | Path, robot_id: int) -> tuple[np.ndarray, np.ndarray]:
    """``(stamps (N,), positions (N, 3))`` of ``/r<id>/odom`` in a bag; empty if the bag has no odometry."""
    from rosbags.rosbag2 import Reader

    from . import messages as msgs

    stamps, positions = [], []
    with Reader(bag_dir) as reader:
        conns = [c for c in reader.connections if c.topic == msgs.topic(robot_id, "odom")]
        for conn, _, raw in reader.messages(connections=conns):
            msg = msgs.TYPESTORE.deserialize_cdr(raw, conn.msgtype)
            p = msg.pose.pose.position
            stamps.append(msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9)
            positions.append((p.x, p.y, p.z))
    return np.asarray(stamps), np.asarray(positions, dtype=float).reshape(-1, 3)


def _unmerged_track(stream: CameraStream, bags_dir: Path | None) -> RobotTrack:
    """GT of the whole run, plus the bag's odometry (aligned to GT) if the bag has any."""
    gt = stream.poses[:, :3, 3]
    odom = np.zeros((0, 3))
    bag = bags_dir / f"r{stream.robot_id}" if bags_dir else None
    if bag is not None and bag.is_dir():
        try:
            stamps, odom = read_bag_odometry(bag, stream.robot_id)
        except ImportError:  # rosbags missing: plot GT only
            stamps = np.zeros(0)
        if len(odom):
            gt = np.array([stream.poses[stream.nearest(t), :3, 3] for t in stamps])
            if len(odom) >= 3:
                odom = transform_points(umeyama(odom, gt), odom)
    return RobotTrack(stream.robot_id, [], gt, odom if len(odom) == len(gt) else np.zeros((0, 3)), np.zeros((0, 3)), merged=False)


def evaluate(streams: list[CameraStream], slam_dir: str | Path, bags_dir: str | Path | None = None) -> Evaluation:
    """Match keyframes to GT by stamp, align, and compute per-robot ATE.

    Odometry is aligned to GT robot by robot (robots share no frame before merging);
    the optimized graph is aligned with **one** transform for all robots, so a wrong
    merge shows up as a displaced robot instead of being hidden by the alignment.
    Robots missing from the merged graph get an unmerged track (see :attr:`RobotTrack.merged`),
    whose odometry is read from ``bags_dir`` (default ``<run>/bags``).
    """
    slam_dir = Path(slam_dir)
    bags_dir = Path(bags_dir) if bags_dir is not None else slam_dir.parent / "bags"
    try:
        log_dir, stamps = latest_logs(slam_dir)
        optimized = read_g2o(log_dir / "optimized_global_pose_graph.g2o")
        initial_path = log_dir / "initial_global_pose_graph.g2o"
        initial = read_g2o(initial_path) if initial_path.exists() else optimized
    except FileNotFoundError:
        log_dir, stamps, optimized = None, {}, PoseGraph()
        initial = optimized

    raw, unmerged = [], {}
    for stream in streams:
        keys = sorted(
            (k for k in optimized.vertices if decode_key(k)[0] == stream.robot_id and k in stamps),
            key=lambda k: decode_key(k)[1],
        )
        if not keys:
            unmerged[stream.robot_id] = _unmerged_track(stream, bags_dir)
            continue
        gt = np.array([stream.poses[stream.nearest(stamps[k]), :3, 3] for k in keys]).reshape(-1, 3)
        odom = np.array([initial.vertices.get(k, optimized.vertices[k])[:3, 3] for k in keys]).reshape(-1, 3)
        opt = np.array([optimized.vertices[k][:3, 3] for k in keys]).reshape(-1, 3)
        if len(keys) >= 3:
            odom = transform_points(umeyama(odom, gt), odom)
        raw.append((stream.robot_id, keys, gt, odom, opt))

    all_opt = np.concatenate([r[4] for r in raw]) if raw else np.zeros((0, 3))
    all_gt = np.concatenate([r[2] for r in raw]) if raw else np.zeros((0, 3))
    alignment = umeyama(all_opt, all_gt) if len(all_opt) >= 3 else np.eye(4)
    tracks = [RobotTrack(rid, keys, gt, odom, transform_points(alignment, opt)) for rid, keys, gt, odom, opt in raw]
    tracks = sorted(tracks + list(unmerged.values()), key=lambda t: t.robot_id)
    return Evaluation(log_dir, tracks, optimized.inter_robot_edges(), stamps, optimized, alignment)


def write_ate_csv(result: Evaluation, path: str | Path) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(("robot_id", "merged", "keyframes", "ate_odometry_m", "ate_optimized_m", "inter_robot_loop_closures"))
        for track in result.tracks:
            n_lc = sum(track.robot_id in (decode_key(a)[0], decode_key(b)[0]) for a, b in result.loop_closures)
            writer.writerow((track.robot_id, int(track.merged), len(track.keys), f"{track.ate_odometry:.3f}", f"{track.ate_optimized:.3f}", n_lc))


#############
## Figures ##
#############


# Paul Tol's "muted" qualitative scheme: colour-blind safe, print friendly.
ROVER_COLORS = ("#332288", "#88CCEE", "#44AA99", "#117733", "#999933", "#DDCC77", "#CC6677", "#882255", "#AA4499", "#DDDDDD")
_STYLE = {
    "gt": dict(ls="-", lw=1.2, alpha=0.3, label="Ground truth"),
    "odometry": dict(ls=(0, (0.6, 1.4)), lw=1.0, alpha=1.0, label="Odometry (drifted, aligned per rover)"),# dash_capstyle="round"),
    "optimized": dict(ls=(0, (4.0, 1.6)), lw=1.0, alpha=1.0, label="Swarm-SLAM optimized"),# solid_capstyle="round"),
}
# Shared by plot and legend so both always match.
_START = dict(ls="", marker="o", ms=3.6, mfc="white", mew=1.0)
_END = dict(ls="", marker="o", ms=3.6, mew=0.0)
LOOP_CLOSURE_COLOR = "red" 
_LC_MARKER = dict(ls="", marker="o", ms=7.0, mfc="none", mec=LOOP_CLOSURE_COLOR, mew=0.8)
_LC_LINE = dict(ls="-", lw=0.9, color=LOOP_CLOSURE_COLOR)
_RC = {
    "font.family": "serif",
    "font.serif": ["STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 9,
    "axes.linewidth": 0.6,
    "axes.edgecolor": "0.25",
    "axes.labelcolor": "0.15",
    "xtick.color": "0.25",
    "ytick.color": "0.25",
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "legend.frameon": False,
}


def _setup_lines(result: Evaluation, scene: dict | None) -> list[tuple[str, str]]:
    """``(key, value)`` rows describing the run; ``scene`` is the run's ``scene.yaml``."""
    rows: list[tuple[str, str]] = []
    if scene:
        run, terrain, motion = scene.get("run", {}), scene.get("terrain", {}), scene.get("motion", {})
        cams, layout = scene.get("cameras", {}), scene.get("layout", {})
        types = sorted({r.get("type", "?") for r in scene.get("rovers", [])})
        rows += [
            ("Terrain", f"{terrain.get('kind', '?')}, {terrain.get('size_m', '?')} m"),
            ("Rovers", f"{len(scene.get('rovers', []))} × {', '.join(types)}"),
            ("Layout", f"{layout.get('kind', '?')}, {layout.get('spacing_m', '?')} m spacing"),
            ("Motion", f"{motion.get('kind', '?')}, {motion.get('speed_mps', '?')} m/s"),
        ]
        if "steps_run" in run and "step_dt_s" in run:
            rows.append(("Duration", f"{run['steps_run'] * run['step_dt_s']:.0f} s ({run.get('frames', '?')} frames)"))
        if cams:
            w, h = cams.get("resolution", ("?", "?"))
            rows.append(("Camera", f"RGB-D {w}×{h}, {cams.get('hfov_deg', '?'):g}° HFOV"))
    rows += [
        ("Merged rovers", f"{sum(t.merged for t in result.tracks)} / {len(result.tracks)}"),
        ("Keyframes", f"{sum(len(t.keys) for t in result.tracks)}"),
        ("Inter-robot LC", f"{len(result.loop_closures)}"),
    ]
    return rows


def plot_trajectories(result: Evaluation, path: str | Path, scene: dict | None = None) -> None:
    """GT, odometry and optimized trajectories in one top-down plot, with a side panel of legend and stats.

    ``scene`` (the run's ``scene.yaml``) adds the setup to the stats panel.
    Inter-robot loop closures are bright circles on both optimized keyframes, joined by a line.
    Robots that were never merged show GT and odometry only.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    with plt.rc_context(_RC):
        fig = plt.figure(figsize=(10.5, 6.5))
        gs = fig.add_gridspec(1, 2, width_ratios=(1.0, 0.42), wspace=0.04, left=0.07, right=0.985, top=0.95, bottom=0.09)
        ax, side = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])
        side.axis("off")

        positions: dict[int, np.ndarray] = {}
        for track in result.tracks:
            color = ROVER_COLORS[track.robot_id % len(ROVER_COLORS)]
            for attr, style in _STYLE.items():
                xy = getattr(track, attr)[:, :2]
                if not len(xy):
                    continue
                kw = {k: v for k, v in style.items() if k != "label"}
                ax.plot(xy[:, 0], xy[:, 1], color=color, zorder=2, **kw)
                if attr == "gt":
                    ax.plot(*xy[0], mec=color, zorder=4, **_START)
                    ax.plot(*xy[-1], mfc=color, zorder=5, **_END)
            positions.update(zip(track.keys, track.optimized))

        n_lc = 0
        for a, b in result.loop_closures:
            if a in positions and b in positions:
                pair = np.stack((positions[a], positions[b]))[:, :2]
                # ax.plot(*pair.T, zorder=6, **_LC_LINE)
                ax.plot(*pair.T, zorder=7, **_LC_MARKER)
                n_lc += 1

        ax.set_aspect("equal", adjustable="datalim")
        ax.set_xlabel(r"$x$ [m]")
        ax.set_ylabel(r"$y$ [m]")
        ax.grid(color="0.88", lw=0.5, zorder=0)
        ax.set_axisbelow(True)

        # Side panel, laid out top-down in rows of fixed height.
        bbox = side.get_position()
        row = 13.0 / (bbox.height * fig.get_figheight() * 72.0)  # 13 pt per row, in axes fraction
        x0, cols = 0.04, (0.04, 0.50, 0.76)
        y = 1.0
        txt = dict(fontsize=8.5, va="top", transform=side.transAxes)
        rule = dict(color="0.6", lw=0.5, transform=side.transAxes, clip_on=False)

        def heading(text: str) -> None:
            nonlocal y
            side.text(x0, y, text, fontsize=9.5, weight="bold", va="top", transform=side.transAxes)
            y -= 1.4 * row

        # Section 1: rover colours, with each rover's ATE.
        heading("Rovers  ·  ATE (RMSE [m]")
        for x, head in zip(cols, ("Rover", "Odometry", "SLAM")):
            side.text(x + (0.11 if x == x0 else 0.0), y, head, color="0.35", **txt)
        y -= 1.1 * row
        side.plot([x0, 0.98], [y, y], **rule)
        y -= 0.3 * row
        ate = [(t.robot_id, t.ate_odometry, t.ate_optimized) for t in result.tracks]
        for track, (rid, odo, opt) in zip(result.tracks, ate):
            color = ROVER_COLORS[rid % len(ROVER_COLORS)]
            side.plot([x0, x0 + 0.08], [y - 0.42 * row] * 2, color=color, lw=3.0, solid_capstyle="butt", transform=side.transAxes)
            odo_s = "–" if np.isnan(odo) else f"{odo:.3f}"
            side.text(cols[0] + 0.11, y, f"{rid}", **txt)
            side.text(cols[1], y, odo_s, **txt)
            if track.merged:
                side.text(cols[2], y, f"{opt:.3f}", **txt)
            else:
                side.text(cols[2], y, "not merged", color="0.45", style="italic", **txt)
            y -= row
        side.plot([x0, 0.98], [y, y], **rule)
        y -= 0.3 * row
        means = ("Mean", f"{np.nanmean([a[1] for a in ate]):.3f}", f"{np.nanmean([a[2] for a in ate]):.3f}")
        for x, val in zip(cols, means):
            side.text(x + (0.11 if x == x0 else 0.0), y, val, weight="bold", **txt)
        y -= 2.2 * row

        # Section 2: line styles and markers.
        heading("Trajectory")
        ink = "0.2"
        style_handles = [Line2D([], [], color=ink, **{k: v for k, v in s.items() if k != "label"}, label=s["label"]) for s in _STYLE.values()]
        style_handles += [
            Line2D([], [], mec=ink, label="Initial pose (GT)", **_START),
            Line2D([], [], mfc=ink, label="Final pose (GT)", **_END),
        ]
        if n_lc:
            style_handles.append(Line2D([], [], label="Inter-robot loop closure", #**_LC_LINE,
                                        **{k: v for k, v in _LC_MARKER.items() if k != "ls"}))
        leg = side.legend(handles=style_handles, loc="upper left", bbox_to_anchor=(x0 - 0.035, y + 0.3 * row), fontsize=8.5,
                          handlelength=2.6, borderaxespad=0.0, labelspacing=0.45)
        fig.canvas.draw()
        y = leg.get_window_extent().transformed(side.transAxes.inverted()).y0 - 2.0 * row

        # Section 3: setup.
        heading("Setup")
        for key, value in _setup_lines(result, scene):
            side.text(x0, y, key, color="0.35", **txt)
            side.text(x0 + 0.34, y, value, **txt)
            y -= row

        fig.savefig(path, dpi=300)
        plt.close(fig)


def plot_loop_closures(result: Evaluation, streams: list[CameraStream], path: str | Path, max_pairs: int = 6) -> bool:
    """Side-by-side RGB of the keyframe pairs of up to ``max_pairs`` inter-robot loop closures."""
    pairs = [(a, b) for a, b in result.loop_closures if a in result.stamps and b in result.stamps][:max_pairs]
    if not pairs:
        return False
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    by_robot = {s.robot_id: s for s in streams}
    fig, axes = plt.subplots(len(pairs), 2, figsize=(8, 3 * len(pairs)), squeeze=False)
    for row, pair in zip(axes, pairs):
        for ax, key in zip(row, pair):
            robot_id, keyframe_id = decode_key(key)
            stream = by_robot[robot_id]
            k = stream.nearest(result.stamps[key])
            ax.imshow(Image.open(stream.rgb[k]))
            ax.set_title(f"r{robot_id} kf {keyframe_id}  t={stream.stamps[k]:.1f}s", fontsize=9)
            ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)
    return True
