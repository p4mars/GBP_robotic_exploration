"""Isaac- and ROS-free tests: python3 -m pytest multi_rover_slam/overseer_viz/tests -q"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # multi_rover_slam/

from cslam_bridge.run_data import load_overseer  # noqa: E402
from multi_rover_setup.cameras import matrix_to_quat_xyzw  # noqa: E402
from multi_rover_setup.terrain import (  # noqa: E402
    TERRAIN_FILE,
    TerrainHeightField,
    load_terrain_geometry,
    save_terrain_geometry,
)
from overseer_viz.geometry import (  # noqa: E402
    convex_hull_2d,
    fov_wedge,
    grid_lines,
    ground_ellipse,
    height_grid,
    offset_segments,
    project,
    shorten_segments,
)

K = np.array([[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]])
# Nadir camera 10 m above the origin: optical z points down, x along world x, so y along world -y.
NADIR = np.eye(4)
NADIR[:3, :3] = np.diag([1.0, -1.0, -1.0])
NADIR[:3, 3] = (0.0, 0.0, 10.0)


def test_project_nadir_camera():
    uv = project(K, NADIR, np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 20.0]]))
    np.testing.assert_allclose(uv[0], (320.0, 240.0))
    np.testing.assert_allclose(uv[1], (370.0, 240.0))  # +x world → right
    np.testing.assert_allclose(uv[2], (320.0, 140.0))  # +y world → up in the image
    assert np.isnan(uv[3]).all()  # above the camera = behind it


def test_ground_ellipse_axes():
    cov = np.array([[4.0, 0.0], [0.0, 1.0]])
    pts = ground_ellipse(np.array([1.0, 2.0, 3.0]), cov, n_sigma=2.0, n=360)
    d = pts[:, :2] - (1.0, 2.0)
    assert np.allclose(pts[:, 2], 3.0)
    assert np.isclose(np.abs(d[:, 0]).max(), 4.0, atol=1e-3)  # 2σ along x, σx = 2
    assert np.isclose(np.abs(d[:, 1]).max(), 2.0, atol=1e-3)
    rot = np.array([[np.cos(0.5), -np.sin(0.5)], [np.sin(0.5), np.cos(0.5)]])
    d_rot = ground_ellipse(np.zeros(3), rot @ cov @ rot.T, 1.0, n=720)[:, :2]
    assert np.isclose(np.linalg.norm(d_rot, axis=1).max(), 2.0, atol=1e-3)
    assert np.isclose(np.linalg.norm(d_rot, axis=1).min(), 1.0, atol=1e-3)


def test_height_grid_interpolates_inside_hull_and_grid_lines():
    pts = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])  # triangle on the plane z = x + 2y
    nodes = height_grid(np.column_stack((pts, pts[:, 0] + 2 * pts[:, 1])), cell_m=0.5)
    assert nodes.shape == (3, 3, 3)
    np.testing.assert_allclose(nodes[1, 0], (0.5, 0.0, 0.5))
    np.testing.assert_allclose(nodes[0, 1], (0.0, 0.5, 1.0))
    assert np.isnan(nodes[2, 2]).all()  # (1, 1) lies outside the triangle
    along_x, along_y = grid_lines(nodes[..., :2])
    assert along_x.shape == along_y.shape == (3 * 4, 2)
    assert np.isnan(along_x[3::4]).all()  # one gap after each line


def test_fov_wedge_opens_along_optical_axis():
    cam = np.eye(4)
    cam[:3, :3] = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])  # optical z = world x
    w = fov_wedge(cam, np.radians(90.0), 2.0, n=3)
    np.testing.assert_allclose(w[0], w[-1])
    np.testing.assert_allclose(w[2, :2], (2.0, 0.0), atol=1e-12)
    np.testing.assert_allclose(np.linalg.norm(w[1:-1, :2], axis=1), 2.0)


def test_convex_hull_and_segments():
    hull = convex_hull_2d(np.array([[0, 0], [1, 0], [1, 1], [0, 1], [0.5, 0.5], [np.nan, 0]]))
    assert len(hull) == 5 and np.allclose(hull[0], hull[-1])
    a, b = np.array([[0.0, 0.0, 0.0]]), np.array([[4.0, 0.0, 0.0]])
    a2, b2 = shorten_segments(a, b, np.array([1.0]), np.array([0.5]))
    np.testing.assert_allclose(a2[0], (1.0, 0.0, 0.0))
    np.testing.assert_allclose(b2[0], (3.5, 0.0, 0.0))
    a3, b3 = offset_segments(a, b, 0.1)
    np.testing.assert_allclose(a3[0], (0.0, -0.1, 0.0))  # right of +x is -y
    np.testing.assert_allclose(b3[0], (4.0, -0.1, 0.0))


def test_terrain_geometry_round_trip(tmp_path):
    rng = np.random.default_rng(0)
    ground, rock = rng.uniform(-5, 5, (200, 3)), rng.uniform(0, 1, (20, 3))
    save_terrain_geometry(tmp_path / TERRAIN_FILE, [("/World/ground", ground), ("/World/rock", rock)])
    meshes = load_terrain_geometry(tmp_path / TERRAIN_FILE)
    assert [name for name, _ in meshes] == ["/World/ground", "/World/rock"]
    np.testing.assert_allclose(meshes[0][1], ground, atol=1e-5)
    all_pts = np.concatenate([p for _, p in meshes])
    field = TerrainHeightField(all_pts)
    assert field.num_points == 220


def _write_run(root: Path, n_frames: int = 4) -> Path:
    """Tiny run folder: 2 rovers, a nadir overseer, one observation per frame, flat terrain."""
    run = root / "run"
    rovers = ["rover_0", "rover_1"]
    (run / "cameras" / "overseer" / "rgb").mkdir(parents=True)
    info = {"width": 640, "height": 480, "K": K.tolist(), "has_depth": False}
    (run / "cameras" / "overseer" / "intrinsics.json").write_text(json.dumps(info))
    q = matrix_to_quat_xyzw(NADIR[:3, :3])
    lines = ["frame,step,time_s,x,y,z,qx,qy,qz,qw"]
    traj = ["frame,step,time_s,rover,x,y,z,qx,qy,qz,qw,cmd_v,cmd_w"]
    obs = ["frame,step,observer,target,u,v,range_m,in_fov,visible"]
    for f in range(n_frames):
        t = 0.08 * f
        lines.append(f"{f},{2 * f},{t},0,0,10,{q[0]},{q[1]},{q[2]},{q[3]}")
        (run / "cameras" / "overseer" / "rgb" / f"{f:06d}.png").write_bytes(b"")
        traj.append(f"{f},{2 * f},{t},rover_0,{0.1 * f},0,0,0,0,0,1,0,0")
        traj.append(f"{f},{2 * f},{t},rover_1,2,1,0,0,0,0.7071068,0.7071068,0,0")
        obs.append(f"{f},{2 * f},rover_0__front,rover_1,1,1,2.2,1,{f % 2}")
    (run / "cameras" / "overseer" / "poses.csv").write_text("\n".join(lines) + "\n")
    (run / "trajectories.csv").write_text("\n".join(traj) + "\n")
    (run / "observations.csv").write_text("\n".join(obs) + "\n")
    for i, rover in enumerate(rovers):  # rover front cameras (with depth) looking along world x
        cam = run / "cameras" / f"{rover}__front"
        (cam / "rgb").mkdir(parents=True)
        (cam / "depth").mkdir()
        (cam / "intrinsics.json").write_text(json.dumps({**info, "has_depth": True}))
        qc = matrix_to_quat_xyzw(np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]]))
        rows = ["frame,step,time_s,x,y,z,qx,qy,qz,qw"]
        for f in range(n_frames):
            rows.append(f"{f},{2 * f},{0.08 * f},{i},0,0.3,{qc[0]},{qc[1]},{qc[2]},{qc[3]}")
            (cam / "rgb" / f"{f:06d}.png").write_bytes(b"")
            (cam / "depth" / f"{f:06d}.png").write_bytes(b"")
        (cam / "poses.csv").write_text("\n".join(rows) + "\n")
    scene = (
        "output: {record_every: 2}\n"
        "run:\n  step_dt_s: 0.04\n  rovers:\n"
        + "".join(f"  - label: {r}\n    spawn: {{radius_m: 0.4, heading_offset_deg: 0.0}}\n" for r in rovers)
    )
    (run / "scene.yaml").write_text(scene)
    xs, ys = np.meshgrid(np.linspace(-6, 6, 25), np.linspace(-6, 6, 25))
    save_terrain_geometry(run / TERRAIN_FILE, [("/World/ground", np.stack((xs, ys, 0 * xs), -1).reshape(-1, 3))])
    return run


def test_load_overseer_without_depth(tmp_path):
    stream = load_overseer(_write_run(tmp_path))
    assert len(stream) == 4 and stream.depth == [] and stream.robot_id == -1
    np.testing.assert_allclose(stream.poses[0], NADIR, atol=1e-6)


def test_run_data_and_render(tmp_path):
    pytest.importorskip("matplotlib")
    from overseer_viz import OverseerFigure, RunData, make_layers

    data = RunData.load(_write_run(tmp_path))
    assert data.rovers == ["rover_0", "rover_1"] and np.isclose(data.fps, 12.5)
    rows = data.observations.at(2)
    assert list(data.observations.observer[rows]) == [0] and not data.observations.visible[rows][0]
    assert data.observations.visible[data.observations.at(3)][0]
    np.testing.assert_allclose(data.heading(0)[1], (0.0, 1.0), atol=1e-6)  # rover_1 yawed 90°
    np.testing.assert_allclose(np.degrees(data.cam_hfov_rad), np.degrees(2 * np.arctan(320 / 500)))

    figure = OverseerFigure(data, make_layers("wireframe", ("fov", "trails", "observations", "rovers")))
    image = figure.draw(3)
    assert image.shape == (480, 640, 3)
    assert (image < 250).any()  # something was drawn on the white background
    paths = figure.save(2, tmp_path / "still")
    assert all(p.stat().st_size > 0 for p in paths)
    with pytest.raises(ValueError, match="session"):
        make_layers("wireframe", ("slam",))


def test_missing_terrain_names_the_export_command(tmp_path):
    from overseer_viz import RunData

    run = _write_run(tmp_path)
    (run / TERRAIN_FILE).unlink()
    with pytest.raises(FileNotFoundError, match="--snapshot-only"):
        RunData.load(run).require_terrain()
