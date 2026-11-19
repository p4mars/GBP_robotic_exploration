"""Isaac-free, ROS-free tests: python3 -m pytest multi_rover_slam/cslam_bridge/tests -q"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # multi_rover_slam/

from cslam_bridge.evaluate import (  # noqa: E402
    anchor_to_first,
    decode_key,
    encode_key,
    evaluate,
    loop_closure_errors,
    loop_closure_funnel,
    read_g2o,
    rpe,
    vo_health,
    rmse,
    transform_points,
    umeyama,
)
from cslam_bridge import session as sessions  # noqa: E402
from cslam_bridge.odometry import drifted_odometry  # noqa: E402
from cslam_bridge.run_data import STAMP_OFFSET_S, load_run  # noqa: E402
from multi_rover_setup.cameras import matrix_to_quat_xyzw  # noqa: E402

K = [[10.0, 0.0, 4.0], [0.0, 10.0, 3.0], [0.0, 0.0, 1.0]]


def _pose(x, y, yaw):
    """Camera looking along world +x rotated by yaw, in ROS optical convention."""
    c, s = np.cos(yaw), np.sin(yaw)
    forward, left = np.array([c, s, 0.0]), np.array([-s, c, 0.0])
    T = np.eye(4)
    T[:3, :3] = np.column_stack((-left, (0.0, 0.0, -1.0), forward))  # x right, y down, z forward
    T[:3, 3] = (x, y, 0.5)
    return T


def _write_camera(cam_dir: Path, poses, has_depth=True):
    (cam_dir / "rgb").mkdir(parents=True)
    if has_depth:
        (cam_dir / "depth").mkdir()
    (cam_dir / "intrinsics.json").write_text(json.dumps({"width": 8, "height": 6, "K": K, "has_depth": has_depth}))
    rows = ["frame,step,time_s,x,y,z,qx,qy,qz,qw"]
    for k, T in enumerate(poses):
        Image.fromarray(np.full((6, 8, 3), k, np.uint8)).save(cam_dir / "rgb" / f"{k:06d}.png")
        if has_depth:
            Image.fromarray(np.full((6, 8), 1000 + k, np.uint16)).save(cam_dir / "depth" / f"{k:06d}.png")
        q = matrix_to_quat_xyzw(T[:3, :3])
        rows.append(",".join(map(str, (k, 2 * k, 0.08 * k, *T[:3, 3], *q))))
    (cam_dir / "poses.csv").write_text("\n".join(rows) + "\n")


@pytest.fixture
def run_dir(tmp_path):
    """Two rovers driving straight lines in opposite directions, plus an overseer without depth."""
    n = 10
    _write_camera(tmp_path / "cameras" / "rover_0__front", [_pose(0.1 * k, 0.0, 0.0) for k in range(n)])
    _write_camera(tmp_path / "cameras" / "rover_1__front", [_pose(2.0 - 0.1 * k, 1.0, np.pi) for k in range(n)])
    _write_camera(tmp_path / "cameras" / "overseer", [_pose(0.0, 0.0, 0.0)], has_depth=False)
    return tmp_path


def test_load_run_skips_overseer_and_offsets_stamps(run_dir):
    streams = load_run(run_dir)
    assert [(s.robot_id, s.label) for s in streams] == [(0, "rover_0__front"), (1, "rover_1__front")]
    s = streams[1]
    assert len(s) == 10 and s.resolution == (8, 6)
    assert s.stamps[0] == pytest.approx(STAMP_OFFSET_S)
    np.testing.assert_allclose(s.poses[3], _pose(1.7, 1.0, np.pi), atol=1e-5)
    assert s.nearest(STAMP_OFFSET_S + 0.17) == 2


def test_zero_drift_is_ground_truth_relative_to_start(run_dir):
    poses = load_run(run_dir)[0].poses
    odom = drifted_odometry(poses)
    expected = np.linalg.inv(poses[0]) @ poses
    np.testing.assert_allclose(odom, expected, atol=1e-9)


def test_drift_grows_with_distance():
    poses = np.array([_pose(0.1 * k, 0.0, 0.0) for k in range(200)])
    odom = drifted_odometry(poses, yaw_bias_deg_per_m=1.0)
    truth = np.linalg.inv(poses[0]) @ poses
    err = np.linalg.norm(odom[:, :3, 3] - truth[:, :3, 3], axis=-1)
    assert err[0] == 0.0 and err[50] > 0.0 and err[-1] > 5 * err[50]


def test_random_yaw_drift_depends_on_seed():
    poses = np.array([_pose(0.1 * k, 0.0, 0.0) for k in range(200)])
    a = drifted_odometry(poses, yaw_sigma_deg_per_m=5.0, seed=0)
    b = drifted_odometry(poses, yaw_sigma_deg_per_m=5.0, seed=1)
    np.testing.assert_allclose(a, drifted_odometry(poses, yaw_sigma_deg_per_m=5.0, seed=0))
    assert not np.allclose(a[-1], b[-1])
    truth = np.linalg.inv(poses[0]) @ poses
    assert np.linalg.norm(a[-1, :3, 3] - truth[-1, :3, 3]) > 0.0


def test_key_round_trip():
    key = encode_key(3, 1234)
    assert key == 7441072484322903250  # gtsam::LabeledSymbol('g', 'D', 1234): char << 56 | label << 48 | index
    assert decode_key(key) == (3, 1234)


def test_umeyama_recovers_rigid_transform():
    rng = np.random.default_rng(0)
    pts = rng.normal(size=(20, 3))
    T = _pose(1.0, -2.0, 0.7)
    est = umeyama(pts, transform_points(T, pts))
    np.testing.assert_allclose(est, T, atol=1e-9)
    assert rmse(pts, pts) == 0.0


def test_anchor_to_first_recovers_gt_and_lets_drift_grow():
    gt = np.array([_pose(0.1 * k, 0.02 * k, 0.01 * k) for k in range(200)])
    other_frame = _pose(-3.0, 7.0, 2.0) @ gt  # same trajectory in an unrelated frame
    np.testing.assert_allclose(anchor_to_first(other_frame, gt[0]), gt[:, :3, 3], atol=1e-9)

    odom = drifted_odometry(gt, yaw_bias_deg_per_m=1.0)
    err = np.linalg.norm(anchor_to_first(odom, gt[0]) - gt[:, :3, 3], axis=-1)
    assert err[0] == pytest.approx(0.0, abs=1e-9) and err[-1] > 5 * err[50] > 0.0


def _write_slam_logs(logs_dir: Path, streams, offset: np.ndarray):
    """Fake cslam logs: every 3rd frame is a keyframe, poses = ground truth moved by ``offset``."""
    folder = logs_dir / "01-01-2026_00-00-00_experiment_robot_0" / "01-01-2026_00-01-00"
    folder.mkdir(parents=True)
    vertices, edges = [], []
    for s in streams:
        stamp_rows = ["vertice_id,sec,nanosec"]
        for kf, k in enumerate(range(0, len(s), 3)):
            key = encode_key(s.robot_id, kf)
            T = offset @ s.poses[k]
            g2o_key = key & ((1 << 56) - 1)  # cslam's g2o files drop the 'g' char byte; the stamps keep it
            vertices.append(f"VERTEX_SE3:QUAT {g2o_key} {' '.join(map(str, T[:3, 3]))} {' '.join(map(str, matrix_to_quat_xyzw(T[:3, :3])))}")
            if kf:
                edges.append(f"EDGE_SE3:QUAT {encode_key(s.robot_id, kf - 1)} {key} 0 0 0 0 0 0 1")
            sec = int(s.stamps[k])
            stamp_rows.append(f"{key},{sec},{int(round((s.stamps[k] - sec) * 1e9))}")
        (folder / f"pose_timestamps{s.robot_id}.csv").write_text("\n".join(stamp_rows) + "\n")
    edges.append(f"EDGE_SE3:QUAT {encode_key(0, 1)} {encode_key(1, 2)} 0 0 0 0 0 0 1")  # inter-robot
    (folder / "optimized_global_pose_graph.g2o").write_text("\n".join(vertices + edges) + "\n")
    return folder


def test_evaluate_perfect_merge_has_zero_ate(run_dir):
    streams = load_run(run_dir)
    session = sessions.new_session(run_dir, "gt_drift")
    folder = _write_slam_logs(sessions.logs_dir(session), streams, offset=_pose(5.0, 3.0, 1.0))
    graph = read_g2o(folder / "optimized_global_pose_graph.g2o")
    assert len(graph.vertices) == 8 and graph.inter_robot_edges() == [(encode_key(0, 1), encode_key(1, 2))]

    result = evaluate(streams, sessions.logs_dir(sessions.resolve_session(run_dir)))
    assert result.log_dir == folder
    assert [len(t.keys) for t in result.tracks] == [4, 4]
    for track in result.tracks:
        assert track.ate_optimized == pytest.approx(0.0, abs=1e-6)
        assert track.ate_odometry == pytest.approx(0.0, abs=1e-6)
        assert track.ate_optimized_anchored == pytest.approx(0.0, abs=1e-6)
        assert track.rpe_optimized(0.2) == pytest.approx(0.0, abs=1e-6)

    # The fake inter-robot edge measures the identity, so its error equals the true relative pose.
    [lc] = loop_closure_errors(result, streams)
    assert lc.error_m == pytest.approx(lc.baseline_m, abs=1e-6) and lc.baseline_m > 0.5


def test_bag_round_trip(run_dir, tmp_path):
    pytest.importorskip("rosbags")
    from rosbags.rosbag2 import Reader

    from cslam_bridge.bags import write_bag
    from cslam_bridge.messages import TYPESTORE

    stream = load_run(run_dir)[1]
    bag = write_bag(stream, tmp_path / "bags" / "r1", drifted_odometry(stream.poses))
    with Reader(bag) as reader:
        counts = {c.topic: c.msgcount for c in reader.connections}
        assert counts == {t: 10 for t in ("/r1/color/image", "/r1/depth/image", "/r1/color/camera_info", "/r1/odom")}
        for conn, t_ns, raw in reader.messages():
            if conn.topic == "/r1/depth/image":
                msg = TYPESTORE.deserialize_cdr(raw, conn.msgtype)
                depth = np.frombuffer(msg.data.tobytes(), "<u2").reshape(msg.height, msg.width)
                assert msg.encoding == "16UC1" and msg.header.frame_id == "r1_camera" and depth[0, 0] == 1000
                assert t_ns == int(STAMP_OFFSET_S * 1e9)
                break


def test_sessions_are_separate_and_latest_follows(run_dir):
    from datetime import datetime

    now = datetime(2026, 11, 17, 12, 24)
    first = sessions.new_session(run_dir, "vo", now)
    second = sessions.new_session(run_dir, "vo", now)  # same minute -> suffix
    gt = sessions.new_session(run_dir, "gt_drift", now)
    assert [p.name for p in (first, second, gt)] == ["26-11-17_12-24_vo", "26-11-17_12-24_vo_2", "26-11-17_12-24_gt"]
    assert sessions.resolve_session(run_dir) == gt.resolve()
    assert sessions.resolve_session(run_dir / "slam" / "latest") == gt.resolve()
    assert sessions.resolve_session(first) == first.resolve()
    assert sessions.run_dir_of(first) == run_dir.resolve()
    with pytest.raises(ValueError):
        sessions.new_session(run_dir, "lidar", now)


def test_odom_mode_from_manifest_then_bags(run_dir):
    session = sessions.new_session(run_dir, "vo")
    assert sessions.odom_mode(session) is None
    bag = sessions.bags_dir(session) / "r0"
    bag.mkdir(parents=True)
    (bag / "metadata.yaml").write_text("name: /r0/odom\n")
    assert sessions.odom_mode(session) == "gt_drift"  # no manifest: guessed from the bags
    sessions.write_manifest(session, odom="vo", run=run_dir.name)
    assert sessions.odom_mode(session) == "vo"
    assert sessions.read_manifest(session) == {"odom": "vo", "run": run_dir.name}


def test_vo_health_counts_processed_and_tracked(tmp_path):
    log = tmp_path / "pipeline.log"
    log.write_text(
        "[rgbd_odometry-4] [INFO] [1.0] [r0.rgbd_odometry]: Odom: quality=0, std dev=99.995000m|99.995000rad\n"
        "[rgbd_odometry-4] [INFO] [1.1] [r0.rgbd_odometry]: Odom: quality=453, std dev=0.001479m|0.020882rad\n"
        "[rgbd_odometry-8] [INFO] [1.2] [r1.rgbd_odometry]: Odom: quality=0, std dev=0.000000m|0.000000rad\n"
        "[map_manager-2] [WARN] [1.3] [r0.cslam_map_manager]: Odom tracking failed, skipping frame\n"
    )
    health = vo_health(log, {0: 5, 1: 5, 2: 5})
    assert [(h.frames, h.processed, h.tracked) for h in health.values()] == [(5, 2, 1), (5, 1, 0), (5, 0, 0)]
    log.write_text("no visual odometry here\n")
    assert vo_health(log, {0: 5}) == {}


def test_rpe_is_zero_for_a_rigidly_moved_copy_and_grows_with_heading_drift():
    gt = np.array([_pose(0.1 * k, 0.0, 0.0) for k in range(100)])
    moved = np.array([_pose(3.0, -2.0, 1.0) @ T for T in gt])
    assert rpe(gt, moved, 1.0) == pytest.approx(0.0, abs=1e-9)
    drifted = np.array([_pose(0.1 * k * np.cos(0.002 * k), 0.1 * k * np.sin(0.002 * k), 0.002 * k) for k in range(100)])
    assert rpe(gt, drifted, 5.0) > rpe(gt, drifted, 1.0) > 0.0
    assert np.isnan(rpe(gt, gt, 100.0))


def test_loop_closure_funnel_counts_robot0_view_once_per_attempt(tmp_path):
    log = tmp_path / "pipeline.log"
    line = "[loop_closure_detection_node.py-{n}] [INFO] [1.0] [r{r}.cslam_loop_closure_detection]: {kind} inter-robot loop closure measurement: ({a},3) -> ({b},7)\n"
    log.write_text("".join(line.format(n=i, r=r, kind=kind, a=a, b=b)
                           for i, (kind, a, b) in enumerate([("Failed", 5, 8), ("New", 8, 5), ("Failed", 0, 3)])
                           for r in range(3)))  # every robot logs every result
    assert loop_closure_funnel(log) == {(0, 3): (1, 0), (5, 8): (2, 1)}
