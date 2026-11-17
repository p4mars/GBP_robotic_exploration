"""Spawn layouts and pose/command helpers (pure numpy, Isaac-free).

The motion controllers live in :mod:`.motion`.

Poses are planar ``(x, y, yaw)`` in the env frame [m, m, rad]; commands are
``(v, w)`` in [m/s, rad/s].
"""

from __future__ import annotations

import math
import warnings

import numpy as np

from .config import LayoutCfg


def wrap_angle(angle):
    return (np.asarray(angle) + np.pi) % (2.0 * np.pi) - np.pi


def ring_radius(n: int, spacing: float) -> float:
    """Radius at which neighbouring rovers on a circle are ``spacing`` apart (chord)."""
    if n <= 1:
        return 0.0
    return spacing / (2.0 * math.sin(math.pi / n))


#############
## Layouts ##
#############


def spawn_layout(n: int, cfg: LayoutCfg, *, seed: int = 0, bound_m: float | None = None) -> np.ndarray:
    """Return ``(n, 3)`` spawn poses ``(x, y, yaw)`` centred on the origin.

    ``bound_m`` is the half-extent that every rover must stay within; layouts
    that do not fit are shrunk uniformly (with a warning).
    """
    rng = np.random.default_rng(seed)
    spacing = cfg.spacing_m
    explicit_yaw = None

    if cfg.kind == "ring":
        radius = ring_radius(n, spacing)
        # Clockwise numbering: driving counter-clockwise, rover i trails rover i-1.
        theta = -2.0 * np.pi * np.arange(n) / max(n, 1)
        xy = radius * np.stack((np.cos(theta), np.sin(theta)), axis=-1)
    elif cfg.kind == "line":
        xy = np.stack(((np.arange(n) - (n - 1) / 2.0) * spacing, np.zeros(n)), axis=-1)
    elif cfg.kind == "grid":
        cols = math.ceil(math.sqrt(n))
        rows = math.ceil(n / cols)
        idx = np.arange(n)
        xy = np.stack(
            ((idx % cols - (cols - 1) / 2.0) * spacing, (idx // cols - (rows - 1) / 2.0) * spacing),
            axis=-1,
        )
    elif cfg.kind == "random":
        xy = _poisson_disk(n, spacing, rng)
    elif cfg.kind == "explicit":
        positions = cfg.positions or []
        if len(positions) != n:
            raise ValueError(f"layout.positions has {len(positions)} entries for {n} rovers")
        xy = np.array([p[:2] for p in positions], dtype=float)
        if all(len(p) == 3 for p in positions):
            explicit_yaw = np.radians([p[2] for p in positions])
    else:
        raise ValueError(f"Unknown layout kind {cfg.kind!r}")

    xy = np.asarray(xy, dtype=float).reshape(n, 2)
    if bound_m is not None and n > 0:
        extent = float(np.max(np.abs(xy))) if xy.size else 0.0
        if extent > bound_m > 0.0:
            warnings.warn(
                f"Layout extent {extent:.1f} m exceeds the terrain bound {bound_m:.1f} m; "
                "shrinking it (reduce layout.spacing_m or increase terrain.size_m)",
                stacklevel=2,
            )
            xy *= bound_m / extent

    yaw = explicit_yaw if explicit_yaw is not None else headings(xy, cfg.heading, rng)
    return np.concatenate((xy, np.asarray(yaw, dtype=float).reshape(n, 1)), axis=-1)


def headings(xy: np.ndarray, mode: str, rng: np.random.Generator) -> np.ndarray:
    """Yaw per rover: chain (face the previous rover), counter-clockwise tangent,
    facing the centroid, or random."""
    rel = xy - xy.mean(axis=0, keepdims=True)
    theta = np.arctan2(rel[:, 1], rel[:, 0])
    at_center = np.linalg.norm(rel, axis=-1) < 1e-6
    if mode == "chain":
        yaw = theta + np.pi / 2.0
        ahead = xy[:-1] - xy[1:]
        yaw[1:] = np.arctan2(ahead[:, 1], ahead[:, 0])
        at_center[1:] = False
    elif mode == "tangent":
        yaw = theta + np.pi / 2.0
    elif mode == "center":
        yaw = theta + np.pi
    elif mode == "random":
        return rng.uniform(-np.pi, np.pi, size=len(xy))
    else:
        raise ValueError(f"Unknown heading mode {mode!r}")
    yaw = np.where(at_center, 0.0, yaw)
    return wrap_angle(yaw)


def _poisson_disk(n: int, spacing: float, rng: np.random.Generator, attempts: int = 2000) -> np.ndarray:
    half = max(spacing, spacing * math.sqrt(n) * 0.75)
    points: list[np.ndarray] = []
    while len(points) < n:
        for _ in range(attempts):
            candidate = rng.uniform(-half, half, size=2)
            if all(np.linalg.norm(candidate - p) >= spacing for p in points):
                points.append(candidate)
                break
        else:
            half *= 1.25  # region too dense, grow it
    xy = np.array(points)
    return xy - xy.mean(axis=0, keepdims=True)


#####################
## Command helpers ##
#####################


def normalize_commands(cmd: np.ndarray, scale_lin: np.ndarray, scale_ang: np.ndarray) -> np.ndarray:
    """Convert [m/s, rad/s] commands to the ``cmd_vel`` action range [-1, 1]."""
    out = np.zeros_like(cmd, dtype=float)
    out[:, 0] = cmd[:, 0] / np.maximum(scale_lin, 1e-6)
    out[:, 1] = cmd[:, 1] / np.maximum(scale_ang, 1e-6)
    return np.clip(out, -1.0, 1.0)


def yaw_from_quat_xyzw(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=float).reshape(-1, 4)
    x, y, z, w = quat.T
    return np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def quat_xyzw_from_yaw(yaw) -> np.ndarray:
    yaw = np.asarray(yaw, dtype=float).reshape(-1)
    half = yaw / 2.0
    return np.stack((np.zeros_like(half), np.zeros_like(half), np.sin(half), np.cos(half)), axis=-1)
