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
import re
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
    measurements: dict[tuple[int, int], np.ndarray] = field(default_factory=dict)
    """edge -> ``(4, 4)`` measured relative pose ``T_a^-1 T_b``"""

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
            edge = (normalize_key(int(parts[1])), normalize_key(int(parts[2])))
            graph.edges.append(edge)
            if len(parts) >= 10:
                x, y, z, qx, qy, qz, qw = map(float, parts[3:10])
                rel = np.eye(4)
                rel[:3, :3] = quat_xyzw_to_matrix((qx, qy, qz, qw))
                rel[:3, 3] = (x, y, z)
                graph.measurements[edge] = rel
    return graph


def read_pose_timestamps(path: str | Path) -> dict[int, float]:
    """``pose_timestamps<id>.csv`` -> {vertex key: stamp [s]}."""
    with open(path, newline="") as f:
        return {normalize_key(int(row["vertice_id"])): int(row["sec"]) + int(row["nanosec"]) * 1e-9 for row in csv.DictReader(f)}


def latest_logs(logs_dir: str | Path) -> tuple[Path, dict[int, float]]:
    """Folder of the newest merged graph, plus the newest keyframe stamps of every robot."""
    logs_dir = Path(logs_dir)
    graphs = sorted(logs_dir.rglob("optimized_global_pose_graph.g2o"), key=lambda p: p.stat().st_mtime)
    if not graphs:
        raise FileNotFoundError(f"No optimized_global_pose_graph.g2o under {logs_dir} (did a loop closure merge the maps?)")
    stamps: dict[int, float] = {}
    newest: dict[str, Path] = {}
    for path in logs_dir.rglob("pose_timestamps*.csv"):
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


def anchor_to_first(odom_poses: np.ndarray, gt_first: np.ndarray) -> np.ndarray:
    """Positions ``(N, 3)`` of odometry poses ``(N, 4, 4)`` moved rigidly so that the first pose equals ``gt_first``.

    Unlike a best fit over the whole trajectory, this lets the odometry drift freely from the true start.
    """
    odom_poses = np.asarray(odom_poses, dtype=float).reshape(-1, 4, 4)
    if not len(odom_poses):
        return np.zeros((0, 3))
    T = gt_first @ np.linalg.inv(odom_poses[0])
    return (T @ odom_poses)[:, :3, 3]


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.sum((a - b) ** 2, axis=-1)))) if len(a) else float("nan")


def rotation_angle_deg(R: np.ndarray) -> float:
    return float(np.degrees(np.arccos(np.clip((np.trace(R[:3, :3]) - 1.0) / 2.0, -1.0, 1.0))))


def rpe(gt_poses: np.ndarray, est_poses: np.ndarray, distance: float) -> float:
    """Relative pose error: RMSE of the translation error of relative poses over ``distance`` metres of GT path.

    For every keyframe ``i`` the partner ``j`` is the first keyframe at least ``distance`` further along
    the ground-truth path; the error is the translation of ``(G_i^-1 G_j)^-1 (E_i^-1 E_j)``. Independent
    of where the trajectory is anchored, so odometry and SLAM are compared on equal terms. NaN if the
    path is shorter than ``distance``.
    """
    gt_poses, est_poses = np.asarray(gt_poses), np.asarray(est_poses)
    if len(gt_poses) < 2 or len(gt_poses) != len(est_poses):
        return float("nan")
    cum = np.concatenate(([0.0], np.cumsum(np.linalg.norm(np.diff(gt_poses[:, :3, 3], axis=0), axis=-1))))
    errors = []
    for i in range(len(cum)):
        j = int(np.searchsorted(cum, cum[i] + distance))
        if j >= len(cum):
            break
        G = np.linalg.inv(gt_poses[i]) @ gt_poses[j]
        E = np.linalg.inv(est_poses[i]) @ est_poses[j]
        errors.append(np.linalg.norm((np.linalg.inv(G) @ E)[:3, 3]))
    return float(np.sqrt(np.mean(np.square(errors)))) if errors else float("nan")


@dataclass
class RobotTrack:
    """Keyframes of one robot that have a stamp, in keyframe order."""

    robot_id: int
    keys: list[int]
    gt: np.ndarray
    """``(K, 3)`` ground-truth positions"""
    odometry: np.ndarray
    """``(K, 3)`` initial (odometry-only) positions, anchored to the GT pose of the first keyframe"""
    optimized: np.ndarray
    """``(K, 3)`` optimized positions, aligned to GT jointly for all robots; empty if not merged"""
    merged: bool = True
    """False if cslam never merged this robot into the graph; ``gt``/``odometry`` then come
    from the recorded run and the bag's ``/r<id>/odom`` (per frame instead of per keyframe)."""
    gt_poses: np.ndarray = field(default_factory=lambda: np.zeros((0, 4, 4)))
    """``(K, 4, 4)`` ground-truth camera poses of the keyframes (merged robots only)"""
    odometry_poses: np.ndarray = field(default_factory=lambda: np.zeros((0, 4, 4)))
    """``(K, 4, 4)`` initial (odometry-only) keyframe poses in the robot's odometry frame"""
    optimized_poses: np.ndarray = field(default_factory=lambda: np.zeros((0, 4, 4)))
    """``(K, 4, 4)`` optimized keyframe poses in the merged (unaligned) frame"""

    @property
    def ate_odometry(self) -> float:
        return rmse(self.odometry, self.gt)

    @property
    def ate_optimized(self) -> float:
        """Optimized positions after the **joint** alignment of all robots: includes inter-robot frame errors."""
        return rmse(self.optimized, self.gt)

    @property
    def optimized_anchored(self) -> np.ndarray:
        """Optimized positions anchored at the GT pose of the first keyframe, exactly like :attr:`odometry`."""
        if not len(self.optimized_poses):
            return np.zeros((0, 3))
        return anchor_to_first(self.optimized_poses, self.gt_poses[0])

    @property
    def ate_optimized_anchored(self) -> float:
        """Same anchoring as :attr:`ate_odometry`, so the two are directly comparable per rover."""
        return rmse(self.optimized_anchored, self.gt)

    def rpe_odometry(self, distance: float) -> float:
        return rpe(self.gt_poses, self.odometry_poses, distance)

    def rpe_optimized(self, distance: float) -> float:
        return rpe(self.gt_poses, self.optimized_poses, distance)


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
    """``(stamps (N,), poses (N, 4, 4))`` of ``/r<id>/odom`` in a bag; empty if the bag has no odometry."""
    from rosbags.rosbag2 import Reader

    from . import messages as msgs

    stamps, positions = [], []
    with Reader(bag_dir) as reader:
        conns = [c for c in reader.connections if c.topic == msgs.topic(robot_id, "odom")]
        if not conns:  # rosbags reads *every* topic for an empty connection list
            return np.zeros(0), np.zeros((0, 4, 4))
        for conn, _, raw in reader.messages(connections=conns):
            msg = msgs.TYPESTORE.deserialize_cdr(raw, conn.msgtype)
            p, q = msg.pose.pose.position, msg.pose.pose.orientation
            pose = np.eye(4)
            pose[:3, :3] = quat_xyzw_to_matrix((q.x, q.y, q.z, q.w))
            pose[:3, 3] = (p.x, p.y, p.z)
            stamps.append(msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9)
            positions.append(pose)
    return np.asarray(stamps), np.asarray(positions, dtype=float).reshape(-1, 4, 4)


def _unmerged_track(stream: CameraStream, bags_dir: Path | None) -> RobotTrack:
    """GT of the whole run, plus the bag's odometry (anchored at the first GT pose) if the bag has any."""
    gt = stream.poses[:, :3, 3]
    odom = np.zeros((0, 3))
    bag = bags_dir / f"r{stream.robot_id}" if bags_dir else None
    if bag is not None and bag.is_dir():
        try:
            stamps, odom_poses = read_bag_odometry(bag, stream.robot_id)
        except ImportError:  # rosbags missing: plot GT only
            stamps, odom_poses = np.zeros(0), np.zeros((0, 4, 4))
        if len(odom_poses):
            gt = np.array([stream.poses[stream.nearest(t), :3, 3] for t in stamps])
            odom = anchor_to_first(odom_poses, stream.poses[stream.nearest(stamps[0])])
    return RobotTrack(stream.robot_id, [], gt, odom if len(odom) == len(gt) else np.zeros((0, 3)), np.zeros((0, 3)), merged=False)


def evaluate(streams: list[CameraStream], logs_dir: str | Path, bags_dir: str | Path | None = None) -> Evaluation:
    """Match keyframes to GT by stamp, align, and compute per-robot ATE.

    Odometry is anchored robot by robot at the GT pose of its first keyframe and drifts freely from there;
    the optimized graph is aligned with **one** transform for all robots, so a wrong
    merge shows up as a displaced robot instead of being hidden by the alignment.
    Robots missing from the merged graph get an unmerged track (see :attr:`RobotTrack.merged`),
    whose odometry is read from ``bags_dir``.

    ``logs_dir`` is cslam's ``evaluation.log_folder``, i.e. ``<session>/logs`` of a SLAM session
    (see :mod:`.session`); ``bags_dir`` defaults to the session's ``bags``.
    """
    logs_dir = Path(logs_dir)
    bags_dir = Path(bags_dir) if bags_dir is not None else logs_dir.parent / "bags"
    try:
        log_dir, stamps = latest_logs(logs_dir)
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
        gt_poses = np.array([stream.poses[stream.nearest(stamps[k])] for k in keys]).reshape(-1, 4, 4)
        gt = gt_poses[:, :3, 3]
        odom_poses = np.array([initial.vertices.get(k, optimized.vertices[k]) for k in keys]).reshape(-1, 4, 4)
        odom = anchor_to_first(odom_poses, gt_poses[0])
        opt_poses = np.array([optimized.vertices[k] for k in keys]).reshape(-1, 4, 4)
        raw.append((stream.robot_id, keys, gt, odom, opt_poses[:, :3, 3], gt_poses, odom_poses, opt_poses))

    all_opt = np.concatenate([r[4] for r in raw]) if raw else np.zeros((0, 3))
    all_gt = np.concatenate([r[2] for r in raw]) if raw else np.zeros((0, 3))
    alignment = umeyama(all_opt, all_gt) if len(all_opt) >= 3 else np.eye(4)
    tracks = [RobotTrack(rid, keys, gt, odom, transform_points(alignment, opt), True, gt_poses, odom_poses, opt_poses)
              for rid, keys, gt, odom, opt, gt_poses, odom_poses, opt_poses in raw]
    tracks = sorted(tracks + list(unmerged.values()), key=lambda t: t.robot_id)
    return Evaluation(log_dir, tracks, optimized.inter_robot_edges(), stamps, optimized, alignment)


def write_ate_csv(result: Evaluation, path: str | Path) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(("robot_id", "merged", "keyframes", "ate_odometry_m", "ate_optimized_m", "ate_optimized_anchored_m",
                         *(f"rpe{d:g}m_{kind}_m" for d in RPE_DISTANCES for kind in ("odometry", "optimized")),
                         "inter_robot_loop_closures"))
        for track in result.tracks:
            n_lc = sum(track.robot_id in (decode_key(a)[0], decode_key(b)[0]) for a, b in result.loop_closures)
            rpes = [f"{v:.3f}" for d in RPE_DISTANCES for v in (track.rpe_odometry(d), track.rpe_optimized(d))]
            writer.writerow((track.robot_id, int(track.merged), len(track.keys), f"{track.ate_odometry:.3f}",
                             f"{track.ate_optimized:.3f}", f"{track.ate_optimized_anchored:.3f}", *rpes, n_lc))


RPE_DISTANCES = (1.0, 5.0)
"""Path lengths [m] over which :func:`rpe` is reported."""


@dataclass
class LoopClosureError:
    """One inter-robot loop closure of the optimized graph compared with ground truth."""

    key_a: int
    key_b: int
    baseline_m: float
    """GT distance between the two cameras."""
    baseline_deg: float
    """GT rotation between the two cameras."""
    error_m: float
    """Translation error of the measured relative pose."""
    error_deg: float
    """Rotation error of the measured relative pose."""


def loop_closure_errors(result: Evaluation, streams: list[CameraStream]) -> list[LoopClosureError]:
    """Measured vs. ground-truth relative pose of every inter-robot loop closure with known stamps."""
    by_id = {s.robot_id: s for s in streams}
    out = []
    for a, b in result.loop_closures:
        meas = result.optimized.measurements.get((a, b))
        if meas is None or a not in result.stamps or b not in result.stamps:
            continue
        (ra, _), (rb, _) = decode_key(a), decode_key(b)
        Ta = by_id[ra].poses[by_id[ra].nearest(result.stamps[a])]
        Tb = by_id[rb].poses[by_id[rb].nearest(result.stamps[b])]
        gt_rel = np.linalg.inv(Ta) @ Tb
        err = np.linalg.inv(gt_rel) @ meas
        out.append(LoopClosureError(a, b, float(np.linalg.norm(gt_rel[:3, 3])), rotation_angle_deg(gt_rel),
                                    float(np.linalg.norm(err[:3, 3])), rotation_angle_deg(err)))
    return out


def write_lc_accuracy_csv(errors: list[LoopClosureError], path: str | Path) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(("robot_a", "keyframe_a", "robot_b", "keyframe_b", "gt_baseline_m", "gt_baseline_deg", "error_m", "error_deg"))
        for e in errors:
            writer.writerow((*decode_key(e.key_a), *decode_key(e.key_b), f"{e.baseline_m:.3f}", f"{e.baseline_deg:.2f}",
                             f"{e.error_m:.4f}", f"{e.error_deg:.3f}"))


_LC_LOG_LINE = re.compile(r"\[r0\.cslam_loop_closure_detection\]: (Failed|New) inter-robot loop closure measurement: "
                      r"\((\d+),(\d+)\) -> \((\d+),(\d+)\)")


def loop_closure_funnel(pipeline_log: str | Path) -> dict[tuple[int, int], tuple[int, int]]:
    """Geometric verification outcomes per rover pair ``(a, b)`` with ``a < b``: ``(attempts, accepted)``.

    Read from robot 0's ``loop_closure_detection`` lines in ``pipeline.log`` (every robot logs every
    broadcast result, so one robot's view counts each attempt once). "Accepted" means the front end
    verified the match; the back end may still drop it.
    """
    funnel: dict[tuple[int, int], list[int]] = {}
    with open(pipeline_log, errors="replace") as f:
        for line in f:
            m = _LC_LOG_LINE.search(line)
            if m:
                pair = tuple(sorted((int(m.group(2)), int(m.group(4)))))
                counts = funnel.setdefault(pair, [0, 0])
                counts[0] += 1
                counts[1] += m.group(1) == "New"
    return {pair: (c[0], c[1]) for pair, c in sorted(funnel.items())}


def session_summary(result: Evaluation, lc_errors: list[LoopClosureError],
                    funnel: dict[tuple[int, int], tuple[int, int]], health: dict[int, "VoHealth"]) -> dict:
    """Scalar metrics of one SLAM session (``summary.yaml``), for comparing sessions/configurations.

    ATE/RPE means are over **merged** rovers only, odometry and SLAM on the same keyframes.
    """
    merged = [t for t in result.tracks if t.merged]

    def mean(values):
        values = [v for v in values if np.isfinite(v)]
        return round(float(np.mean(values)), 4) if values else None

    def median(values):
        return round(float(np.median(values)), 4) if len(values) else None

    out = {
        "rovers": len(result.tracks),
        "merged": len(merged),
        "keyframes_merged": int(sum(len(t.keys) for t in merged)),
        "inter_robot_lc": len(result.loop_closures),
        "lc_baseline_median_m": median([e.baseline_m for e in lc_errors]),
        "lc_baseline_max_m": round(max(e.baseline_m for e in lc_errors), 4) if lc_errors else None,
        "lc_error_median_m": median([e.error_m for e in lc_errors]),
        "lc_error_max_m": round(max(e.error_m for e in lc_errors), 4) if lc_errors else None,
        "lc_error_max_deg": round(max(e.error_deg for e in lc_errors), 3) if lc_errors else None,
        "verifications": int(sum(n for n, _ in funnel.values())),
        "verified": int(sum(ok for _, ok in funnel.values())),
        "ate_odometry_m": mean([t.ate_odometry for t in merged]),
        "ate_slam_anchored_m": mean([t.ate_optimized_anchored for t in merged]),
        "ate_slam_joint_m": mean([t.ate_optimized for t in merged]),
    }
    for d in RPE_DISTANCES:
        out[f"rpe{d:g}m_odometry_m"] = mean([t.rpe_odometry(d) for t in merged])
        out[f"rpe{d:g}m_slam_m"] = mean([t.rpe_optimized(d) for t in merged])
    if health:
        out["vo_tracked_ratio_min"] = round(min(h.tracked / max(h.frames, 1) for h in health.values()), 3)
    return out


def write_lc_funnel_csv(funnel: dict[tuple[int, int], tuple[int, int]], path: str | Path) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(("robot_a", "robot_b", "verifications", "accepted", "acceptance_ratio"))
        for (a, b), (n, ok) in funnel.items():
            writer.writerow((a, b, n, ok, f"{ok / max(n, 1):.3f}"))


@dataclass
class VoHealth:
    """How RTAB-Map visual odometry of one rover did during replay."""

    frames: int
    """Frames in the bag."""
    processed: int
    """Frames rgbd_odometry received and processed; fewer than ``frames`` means frames were lost in transport."""
    tracked: int
    """Processed frames with a valid estimate (``quality > 0``); the others were 'lost'."""


_VO_LINE = re.compile(r"\[r(\d+)\.rgbd_odometry\]: Odom: quality=(\d+)")


def vo_health(pipeline_log: str | Path, frames: dict[int, int]) -> dict[int, VoHealth]:
    """Per-rover VO frame counts from ``rgbd_odometry``'s ``Odom: quality=<inliers>`` lines in ``pipeline.log``.

    ``frames`` maps robot id -> frames in its bag. Empty if the log has no VO lines (GT odometry).
    """
    processed: dict[int, int] = {}
    tracked: dict[int, int] = {}
    with open(pipeline_log, errors="replace") as f:
        for line in f:
            m = _VO_LINE.search(line)
            if m:
                rid, quality = int(m.group(1)), int(m.group(2))
                processed[rid] = processed.get(rid, 0) + 1
                tracked[rid] = tracked.get(rid, 0) + (quality > 0)
    if not processed:
        return {}
    return {rid: VoHealth(n, processed.get(rid, 0), tracked.get(rid, 0)) for rid, n in sorted(frames.items())}


def write_vo_health_csv(health: dict[int, VoHealth], path: str | Path) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(("robot_id", "frames", "vo_processed", "vo_tracked", "processed_ratio", "tracked_ratio"))
        for rid, h in health.items():
            writer.writerow((rid, h.frames, h.processed, h.tracked,
                             f"{h.processed / max(h.frames, 1):.3f}", f"{h.tracked / max(h.frames, 1):.3f}"))


#############
## Figures ##
#############


# Paul Tol's "muted" qualitative scheme: colour-blind safe, print friendly.
ROVER_COLORS = ("#332288", "#88CCEE", "#44AA99", "#117733", "#999933", "#DDCC77", "#CC6677", "#882255", "#AA4499", "#DDDDDD")
_STYLE = {
    "gt": dict(ls="-", lw=1.2, alpha=0.3, label="Ground truth"),
    "odometry": dict(ls=(0, (0.6, 1.4)), lw=1.0, alpha=1.0, label="Odometry (anchored at first pose)"),# dash_capstyle="round"),
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

        fig.savefig(path, dpi=600)
        plt.close(fig)


# One colour per data type for the per-rover figure (Paul Tol "muted").
_INDIV_STYLE = {
    "gt": dict(color="#44AA99", ls="-", lw=1.0, alpha=0.45, solid_capstyle="round", label="Ground truth"),
    "odometry": dict(color="#C9B54F", ls="-", lw=1.0, alpha=0.95, label="Odometry"),
    "optimized": dict(color="#882255", ls="-", lw=1.0, alpha=0.95, label="Swarm-SLAM optimized"),
}


def _fmt_duration(seconds: float) -> str:
    seconds = int(round(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h} h {m:02d} min" if h else f"{m} min {s:02d} s"


def _run_info(result: Evaluation, run_dir: Path | None, session_dir: Path | None = None) -> dict[str, str]:
    """Odometry source (from the SLAM session) and wall-clock compute times, where the folders record them."""
    import re
    from datetime import datetime

    from . import session as sessions

    info: dict[str, str] = {}
    if session_dir is not None:
        mode = sessions.odom_mode(session_dir)
        if mode is not None:
            info["odometry"] = sessions.ODOM_LABEL.get(mode, mode)
    if run_dir is not None:
        log = run_dir / "run.log"
        if log.is_file():
            stamp = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})")
            lines = log.read_text().splitlines()
            first = next((stamp.match(l) for l in lines if stamp.match(l)), None)
            done = next((stamp.match(l) for l in reversed(lines) if "Finished" in l and stamp.match(l)), None)
            if first and done:
                t0, t1 = (datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S") for m in (first, done))
                info["sim_time"] = _fmt_duration((t1 - t0).total_seconds())
    if result.log_dir is not None:
        # <slam>/<dd-mm-YYYY_HH-MM-SS>_experiment_robot_<id>/<dd-mm-YYYY_HH-MM-SS>: cslam start -> newest graph
        fmt = "%d-%m-%Y_%H-%M-%S"
        try:
            t0 = datetime.strptime(result.log_dir.parent.name.split("_experiment")[0], fmt)
            t1 = datetime.strptime(result.log_dir.name, fmt)
            info["slam_time"] = _fmt_duration((t1 - t0).total_seconds())
        except ValueError:
            pass
    return info


def plot_trajectories_indiv(result: Evaluation, path: str | Path, scene: dict | None = None, run_dir: str | Path | None = None,
                            session_dir: str | Path | None = None, max_cols: int = 5) -> None:
    """One top-down panel per rover (GT / odometry / optimized), up to ``max_cols`` per row, on shared axes.

    A header holds the title, the setup (``scene`` = the run's ``scene.yaml``; ``run_dir`` adds the
    simulation time, ``session_dir`` the odometry source and SLAM time) and the legend. Red circles mark the rover's keyframes that
    take part in an inter-robot loop closure.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    run_dir = Path(run_dir) if run_dir is not None else None
    session_dir = Path(session_dir) if session_dir is not None else None
    tracks = result.tracks
    n = max(len(tracks), 1)
    ncols = min(max_cols, n)
    nrows = -(-n // ncols)
    info = _run_info(result, run_dir, session_dir)

    # Shared, square data window over everything that is plotted.
    pts = [getattr(t, a)[:, :2] for t in tracks for a in _INDIV_STYLE if len(getattr(t, a))]
    allp = np.concatenate(pts) if pts else np.zeros((1, 2))
    lo, hi = allp.min(axis=0), allp.max(axis=0)
    center, half = (lo + hi) / 2, max(float((hi - lo).max()) / 2 * 1.06, 1.0)

    # Absolute layout in inches.
    fig_w, left, right, gap_x = 13.0, 0.75, 0.25, 0.22
    cell = min((fig_w - left - right - (ncols - 1) * gap_x) / ncols, 3.4)
    title_h, header_h, sub_h, tick_h, bottom = 0.85, 1.25, 0.40, 0.32, 0.45
    pitch = sub_h + cell + tick_h
    fig_h = title_h + header_h + nrows * pitch + bottom
    fx = lambda inch: inch / fig_w  # noqa: E731
    fy = lambda inch: 1.0 - inch / fig_h  # noqa: E731  (inches from the top)

    with plt.rc_context(_RC):
        fig = plt.figure(figsize=(fig_w, fig_h))

        # Title.
        n_merged = sum(t.merged for t in tracks)
        terrain = (scene or {}).get("terrain", {}).get("kind", "")
        terrain = terrain.replace("_", " ") if terrain else ""
        fig.text(fx(left), fy(0.28), "Multi-Rover Swarm-SLAM: Per-Rover Trajectory Evaluation",
                 fontsize=15, weight="bold", va="top")
        sub = [f"{len(tracks)} rovers ({n_merged} merged)", f"{len(result.loop_closures)} inter-robot loop closures"]
        if terrain:
            sub.insert(0, terrain)
        merged = [t for t in tracks if t.merged]
        def _mean_m(values) -> str:
            values = [v for v in values if np.isfinite(v)]
            return f"{np.mean(values):.2f} m" if values else "–"

        if tracks:
            # Anchored at each rover's first GT pose, like the odometry, so the two are comparable;
            # "joint" = one alignment for all merged rovers (shows inter-robot frame errors, as drawn).
            sub.append(f"mean ATE: odometry {_mean_m([t.ate_odometry for t in merged])}, "
                       f"SLAM {_mean_m([t.ate_optimized_anchored for t in merged])} "
                       f"(joint {_mean_m([t.ate_optimized for t in merged])}), merged rovers")
        fig.text(fx(left), fy(0.62), "  ·  ".join(sub), fontsize=10, color="0.35", va="top")
        fig.add_artist(Line2D([fx(left), 1 - fx(right)], [fy(title_h - 0.05)] * 2, color="0.75", lw=0.6))

        # Setup, in three columns of key/value rows.
        rows = _setup_lines(result, scene)
        if "odometry" in info:
            rows.insert(5, ("Odometry", info["odometry"]))
        if "sim_time" in info:
            rows.append(("Simulation time", info["sim_time"]))
        if "slam_time" in info:
            rows.append(("SLAM time", info["slam_time"]))
        top = title_h + 0.2
        fig.text(fx(left), fy(top), "Setup", fontsize=10, weight="bold", va="top")
        setup_w = fig_w - left - right - 3.6
        col_w, per_col, row_h = setup_w / 3, -(-len(rows) // 3), 0.19
        for i, (key, value) in enumerate(rows):
            c, r = divmod(i, per_col)
            x, y = left + c * col_w, top + 0.27 + r * row_h
            fig.text(fx(x), fy(y), key, fontsize=8.5, color="0.4", va="top")
            fig.text(fx(x + 0.95), fy(y), value, fontsize=8.5, va="top")

        # Legend: data types (left column), markers (right column).
        handles = [Line2D([], [], **{**s, "alpha": max(s["alpha"], 0.6), "lw": 2.0}) for s in _INDIV_STYLE.values()]
        if "odometry" in info:
            handles[1].set_label(f"Odometry ({info['odometry']})")
        gt_c = _INDIV_STYLE["gt"]["color"]
        handles += [
            Line2D([], [], mec=gt_c, label="GT initial pose", **{**_START, "ms": 5.0}),
            Line2D([], [], mfc=gt_c, label="GT final pose", **{**_END, "ms": 4.0}),
            Line2D([], [], label="Inter-robot loop closure", **_LC_MARKER),
        ]
        fig.text(1 - fx(right + 3.45), fy(top), "Legend", fontsize=10, weight="bold", va="top")
        fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(1 - fx(right + 3.5), fy(top + 0.2)), ncol=2,
                   fontsize=8.5, handlelength=2.0, columnspacing=1.4, labelspacing=0.55, borderaxespad=0.0)

        # Rover panels.
        axes = []
        grid_top = title_h + header_h
        for i, track in enumerate(tracks):
            r, c = divmod(i, ncols)
            x0, y_top = left + c * (cell + gap_x), grid_top + r * pitch + sub_h
            ax = fig.add_axes((fx(x0), fy(y_top + cell), fx(cell), cell / fig_h))
            axes.append(ax)
            for attr, style in _INDIV_STYLE.items():
                xy = getattr(track, attr)[:, :2]
                if len(xy):
                    ax.plot(xy[:, 0], xy[:, 1], zorder={"gt": 2, "odometry": 3, "optimized": 4}[attr],
                            **{k: v for k, v in style.items() if k != "label"})
            if len(track.gt):
                ax.plot(*track.gt[0, :2], mec=gt_c, zorder=6, **{**_START, "ms": 5.0})
                ax.plot(*track.gt[-1, :2], mfc=gt_c, mec=gt_c, zorder=6, **{**_END, "ms": 4.0})
            own = dict(zip(track.keys, track.optimized))
            lc = [own[k] for a, b in result.loop_closures for k in (a, b) if k in own]
            if lc:
                ax.plot(*np.asarray(lc)[:, :2].T, zorder=7, **_LC_MARKER)

            ax.set_xlim(center[0] - half, center[0] + half)
            ax.set_ylim(center[1] - half, center[1] + half)
            ax.set_aspect("equal", adjustable="box")
            ax.grid(color="0.9", lw=0.5)
            ax.set_axisbelow(True)
            ax.tick_params(labelsize=7.5, length=2.5)
            below = i + ncols < len(tracks)
            ax.tick_params(labelbottom=not below, labelleft=c == 0)

            odo = "–" if np.isnan(track.ate_odometry) else f"{track.ate_odometry:.2f} m"
            slam = (f"{track.ate_optimized_anchored:.2f} m (joint {track.ate_optimized:.2f})" if track.merged
                    else "not merged")
            ax.text(0.0, 1.0 + 0.21 / cell, f"rover_{track.robot_id}", transform=ax.transAxes, fontsize=10, weight="bold", va="bottom")
            # ATE line; each number in its trajectory's colour (pieces placed one after another).
            pieces = [("Odometry ATE ", "0.35"), (odo, _INDIV_STYLE["odometry"]["color"] if odo != "–" else "0.35"),
                      (",  SLAM ATE ", "0.35"), (slam, _INDIV_STYLE["optimized"]["color"] if track.merged else "0.35")]
            renderer, x = fig.canvas.get_renderer(), 0.0
            for text, color in pieces:
                t = ax.text(x, 1.0 + 0.05 / cell, text, transform=ax.transAxes, fontsize=8, color=color, va="bottom",
                            weight="bold" if color not in ("0.35",) else "normal")
                x += t.get_window_extent(renderer).width / ax.get_window_extent(renderer).width

        # Shared axis labels.
        grid_bottom = grid_top + nrows * pitch - tick_h
        grid_w = ncols * cell + (ncols - 1) * gap_x
        fig.text(fx(left + grid_w / 2), fy(grid_bottom + 0.30), r"$x$ [m]", ha="center", va="top", fontsize=10)
        fig.text(fx(left - 0.5), fy((grid_top + grid_bottom) / 2), r"$y$ [m]", ha="center", va="center",
                 rotation=90, fontsize=10)

        fig.savefig(path, dpi=600)
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
