"""World-space shapes and their projection into the overseer image (pure numpy).

Everything is built in world coordinates and projected per frame, so drawings stay
attached to the ground while the overseer camera moves.
"""

from __future__ import annotations

import numpy as np

from multi_rover_setup.cameras import matrix_to_quat_xyzw, project_points


def project(K: np.ndarray, cam_pose: np.ndarray, points_w: np.ndarray, near: float = 0.05) -> np.ndarray:
    """Pixel coordinates ``(..., 2)`` of world points ``(..., 3)`` in a ROS-optical camera.

    Points behind the camera (or NaN input) come back as NaN, which matplotlib treats as a gap.
    """
    pts = np.asarray(points_w, dtype=float)
    shape = pts.shape[:-1]
    uv, z = project_points(K, cam_pose[:3, 3], matrix_to_quat_xyzw(cam_pose[:3, :3]), pts.reshape(-1, 3))
    uv[~(z > near)] = np.nan
    return uv.reshape(*shape, 2)


def height_grid(points: np.ndarray, cell_m: float) -> np.ndarray:
    """``(nx, ny, 3)`` regular grid nodes every ``cell_m`` with z linearly interpolated over the vertices.

    Interpolation (not binning) keeps the lines smooth where vertices are sparser than the grid;
    nodes outside the vertices' convex hull are NaN.
    """
    from scipy.interpolate import LinearNDInterpolator

    pts = np.asarray(points, dtype=float).reshape(-1, 3)
    lo, hi = pts[:, :2].min(axis=0), pts[:, :2].max(axis=0)
    xs = np.arange(lo[0], hi[0] + 1e-9, cell_m)
    ys = np.arange(lo[1], hi[1] + 1e-9, cell_m)
    gx, gy = np.meshgrid(xs, ys, indexing="ij")
    z = LinearNDInterpolator(pts[:, :2], pts[:, 2])(gx, gy)
    nodes = np.stack((gx, gy, z), axis=-1)
    nodes[np.isnan(z)] = np.nan
    return nodes


def grid_lines(nodes_uv: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Projected grid ``(nx, ny, 2)`` → two NaN-separated polylines: lines along x and lines along y."""
    nx, ny, _ = nodes_uv.shape
    along_x = np.concatenate((nodes_uv.transpose(1, 0, 2), np.full((ny, 1, 2), np.nan)), axis=1).reshape(-1, 2)
    along_y = np.concatenate((nodes_uv, np.full((nx, 1, 2), np.nan)), axis=1).reshape(-1, 2)
    return along_x, along_y


def ground_circle(center_w: np.ndarray, radius: float, n: int = 32) -> np.ndarray:
    """Closed ``(n + 1, 3)`` circle in the horizontal plane through ``center_w``."""
    a = np.linspace(0.0, 2.0 * np.pi, n + 1)
    c = np.asarray(center_w, dtype=float)
    return np.stack((c[0] + radius * np.cos(a), c[1] + radius * np.sin(a), np.full_like(a, c[2])), axis=-1)


def ground_ellipse(center_w: np.ndarray, cov_xy: np.ndarray, n_sigma: float = 2.0, n: int = 48) -> np.ndarray:
    """Closed ``(n + 1, 3)`` ``n_sigma`` ellipse of a 2×2 xy covariance, in the horizontal plane through ``center_w``.

    For a pose belief, ``cov_xy`` is the world-frame x/y block of its position covariance.
    """
    cov = np.asarray(cov_xy, dtype=float).reshape(2, 2)
    w, v = np.linalg.eigh(0.5 * (cov + cov.T))
    a = np.linspace(0.0, 2.0 * np.pi, n + 1)
    unit = np.stack((np.cos(a), np.sin(a)), axis=0)
    xy = (v @ (np.sqrt(np.clip(w, 0.0, None))[:, None] * unit)).T * n_sigma
    c = np.asarray(center_w, dtype=float)
    return np.column_stack((c[0] + xy[:, 0], c[1] + xy[:, 1], np.full(len(a), c[2])))


def fov_wedge(cam_pose: np.ndarray, hfov_rad: float, range_m: float, n: int = 16) -> np.ndarray:
    """``(n + 2, 3)`` closed wedge: camera position, arc at ``range_m``, camera position.

    The wedge lies in the horizontal plane through the camera and opens around the optical
    axis projected onto that plane, which is what a top view shows of a forward camera.
    """
    apex = np.asarray(cam_pose[:3, 3], dtype=float)
    forward = np.asarray(cam_pose[:3, 2], dtype=float)[:2]
    heading = np.arctan2(forward[1], forward[0])
    a = heading + np.linspace(-0.5 * hfov_rad, 0.5 * hfov_rad, n)
    arc = np.stack((apex[0] + range_m * np.cos(a), apex[1] + range_m * np.sin(a), np.full(n, apex[2])), axis=-1)
    return np.vstack((apex, arc, apex))


def convex_hull_2d(points: np.ndarray) -> np.ndarray:
    """Closed counter-clockwise hull ``(K + 1, 2)`` of 2-D points (monotone chain); NaN rows are ignored."""
    pts = np.asarray(points, dtype=float).reshape(-1, 2)
    pts = np.unique(pts[np.isfinite(pts).all(axis=1)], axis=0)
    if len(pts) < 3:
        return pts

    def turn(o, a, b) -> float:  # z of (a - o) × (b - o); np.cross on 2-D vectors is deprecated in numpy 2
        return float((a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]))

    def half(seq):
        out: list[np.ndarray] = []
        for p in seq:
            while len(out) >= 2 and turn(out[-2], out[-1], p) <= 0:
                out.pop()
            out.append(p)
        return out

    lower, upper = half(pts), half(pts[::-1])
    hull = np.array(lower[:-1] + upper[:-1])
    return np.vstack((hull, hull[:1]))


def offset_segments(a: np.ndarray, b: np.ndarray, offset: float) -> tuple[np.ndarray, np.ndarray]:
    """Shift segments ``a → b`` (``(N, 3)`` each) sideways by ``offset`` metres to their right in the xy plane.

    Draws ``i → j`` and ``j → i`` as two parallel lanes instead of one line on top of the other.
    """
    d = np.asarray(b, dtype=float)[:, :2] - np.asarray(a, dtype=float)[:, :2]
    norm = np.linalg.norm(d, axis=1, keepdims=True)
    right = np.column_stack((d[:, 1], -d[:, 0])) / np.where(norm > 0, norm, 1.0)
    shift = np.column_stack((right * offset, np.zeros(len(d))))
    return a + shift, b + shift


def shorten_segments(a: np.ndarray, b: np.ndarray, start_m: np.ndarray, end_m: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Pull both ends of segments ``a → b`` inwards (e.g. so lines start at a rover's outline, not its centre)."""
    d = np.asarray(b, dtype=float) - np.asarray(a, dtype=float)
    length = np.linalg.norm(d[:, :2], axis=1)
    safe = np.where(length > 0, length, 1.0)
    s = np.clip(np.asarray(start_m, dtype=float) / safe, 0.0, 0.45)[:, None]
    e = np.clip(np.asarray(end_m, dtype=float) / safe, 0.0, 0.45)[:, None]
    return a + s * d, b - e * d
