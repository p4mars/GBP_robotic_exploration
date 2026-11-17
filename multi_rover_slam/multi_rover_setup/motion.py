"""Rover motion controllers (pure numpy, Isaac-free).

Poses are planar ``(x, y, yaw)`` in the env frame [m, m, rad]; commands are
``(v, w)`` in [m/s, rad/s].

With ``MotionCfg.collision_avoidance`` the command of every motion kind (except
``idle``) is post-processed by a decentralised swarm rule, see
:meth:`MotionController._avoid`.
"""

from __future__ import annotations

import math

import numpy as np

from .config import MotionCfg
from .layout import wrap_angle


class MotionController:
    """Maps current poses ``(n, 3)`` to commands ``(n, 2)`` in [m/s, rad/s]."""

    def __init__(
        self,
        cfg: MotionCfg,
        *,
        center: np.ndarray,
        radius: float,
        bound_m: float,
        gap_m: float | np.ndarray = 4.0,
        max_lin: np.ndarray,
        max_ang: np.ndarray,
        dt: float,
        seed: int = 0,
        radii: float | np.ndarray = 0.5,
    ):
        self.cfg = cfg
        self.center = np.asarray(center, dtype=float)[:2]
        self.radius = float(radius)
        self.bound_m = float(bound_m)
        n = len(np.asarray(max_lin))
        # Follow distance between rover i and i-1 (index i-1 of this array).
        self.gaps = np.broadcast_to(np.asarray(gap_m, dtype=float), (max(n - 1, 0),)).copy()
        if cfg.kind == "follow" and n > 1:
            # Each follower trails on a smaller circle (r_i^2 = r_{i-1}^2 - gap^2),
            # so the leader's orbit must leave room for the whole chain.
            needed = 1.3 * math.sqrt(float(np.sum(self.gaps**2)))
            self.radius = min(max(self.radius, needed), 0.9 * self.bound_m)
        # Footprint radius per rover [m], used by collision avoidance.
        self.radii = np.broadcast_to(np.asarray(radii, dtype=float), (n,)).copy()
        self.max_lin = np.asarray(max_lin, dtype=float)
        self.max_ang = np.asarray(max_ang, dtype=float)
        self.dt = float(dt)
        self.rng = np.random.default_rng(seed)
        # Common convoy speed so rovers of different top speeds stay together;
        # followers keep headroom to close gaps.
        headroom = 0.6 if cfg.kind == "follow" else 0.8
        self.speed = float(min(cfg.speed_mps, headroom * float(np.min(self.max_lin)) if n else cfg.speed_mps))
        self._wander_state = np.zeros((n, 2))
        self._wander_state[:, 0] = self.speed

    def __call__(self, poses: np.ndarray) -> np.ndarray:
        poses = np.asarray(poses, dtype=float)
        kind = self.cfg.kind
        if kind == "idle":
            cmd = np.zeros((len(poses), 2))
        elif kind == "constant":
            cmd = np.tile(np.asarray(self.cfg.cmd, dtype=float), (len(poses), 1))
        elif kind == "orbit":
            cmd = self._orbit(poses)
        elif kind == "follow":
            cmd = self._follow(poses)
        elif kind == "wander":
            cmd = self._wander(poses)
        else:
            raise ValueError(f"Unknown motion kind {kind!r}")
        if self.cfg.collision_avoidance and kind != "idle":
            cmd = self._avoid(poses, cmd)
        cmd[:, 0] = np.clip(cmd[:, 0], -self.max_lin, self.max_lin)
        cmd[:, 1] = np.clip(cmd[:, 1], -self.max_ang, self.max_ang)
        return cmd

    def _orbit(self, poses: np.ndarray, k_radial: float = 0.8, k_heading: float = 1.5) -> np.ndarray:
        rel = poses[:, :2] - self.center
        dist = np.linalg.norm(rel, axis=-1)
        theta = np.arctan2(rel[:, 1], rel[:, 0])
        radius = max(self.radius, 1.0)
        # CCW tangent, tilted inwards when outside the ring and outwards inside it.
        desired = theta + np.pi / 2.0 + np.arctan(k_radial * (dist - radius))
        err = wrap_angle(desired - poses[:, 2])
        v = np.full(len(poses), self.speed) * np.clip(np.cos(err), 0.2, 1.0)
        w = v / radius + k_heading * err
        return np.stack((v, w), axis=-1)

    def _follow(self, poses: np.ndarray, k_gap: float = 0.5, k_heading: float = 1.5) -> np.ndarray:
        """Rover 0 orbits; every other rover pursues the one before it, keeping it in view."""
        cmd = self._orbit(poses)
        ahead = poses[:-1, :2] - poses[1:, :2]
        dist = np.linalg.norm(ahead, axis=-1)
        err = wrap_angle(np.arctan2(ahead[:, 1], ahead[:, 0]) - poses[1:, 2])
        v = np.clip(k_gap * (dist - self.gaps) + self.speed, 0.0, 1.5 * self.speed) * np.clip(np.cos(err), 0.0, 1.0)
        cmd[1:, 0] = v
        cmd[1:, 1] = k_heading * err
        return cmd

    def _wander(self, poses: np.ndarray, theta_ou: float = 0.5, sigma_v: float = 0.1, sigma_w: float = 0.6) -> np.ndarray:
        state = self._wander_state
        noise = self.rng.standard_normal(state.shape) * math.sqrt(self.dt)
        state[:, 0] += theta_ou * (self.speed - state[:, 0]) * self.dt + sigma_v * noise[:, 0]
        state[:, 1] += theta_ou * (0.0 - state[:, 1]) * self.dt + sigma_w * noise[:, 1]
        state[:, 0] = np.clip(state[:, 0], 0.0, self.speed * 1.5)
        cmd = state.copy()
        # Soft boundary: blend in a turn back towards the centre near the edge.
        rel = poses[:, :2] - self.center
        dist = np.linalg.norm(rel, axis=-1)
        soft = 0.7 * self.bound_m
        weight = np.clip((dist - soft) / max(self.bound_m - soft, 1e-6), 0.0, 1.0)
        home = wrap_angle(np.arctan2(-rel[:, 1], -rel[:, 0]) - poses[:, 2])
        cmd[:, 1] = (1.0 - weight) * cmd[:, 1] + weight * 2.0 * home
        return cmd

    def _avoid(
        self,
        poses: np.ndarray,
        cmd: np.ndarray,
        *,
        margin_m: float = 0.5,
        sense_factor: float = 2.5,
        k_tangent: float = 0.3,
        k_heading: float = 1.5,
        horizon_s: float = 2.5,
    ) -> np.ndarray:
        """Bend the nominal commands away from neighbouring rovers.

        1. Steering: each rover adds a repulsive velocity from every neighbour
           closer than ``sense_factor * d_safe`` (``d_safe`` = both footprint radii
           + ``margin_m``), scaled so that it equals the cruise speed at ``d_safe``
           and vanishes at the sensing radius. A tangential share ``k_tangent``
           makes rovers pass each other on the right instead of deadlocking.
           The sum with the nominal forward velocity is turned back into ``(v, w)``.
        2. Speed limit: if a neighbour ahead is on a collision course (closest
           approach below ``d_safe`` within ``horizon_s``, assuming constant
           velocities), forward speed is scaled by ``t_closest / horizon_s``; a
           neighbour ahead already inside ``d_safe`` stops forward motion (the
           rover may still turn away). Rovers never brake for one behind them.
        """
        n = len(poses)
        if n < 2:
            return cmd
        cmd = cmd.copy()
        xy, yaw = poses[:, :2], poses[:, 2]
        fwd = np.stack((np.cos(yaw), np.sin(yaw)), axis=-1)
        speed = max(self.speed, 0.05)

        # Pairwise geometry, indexed [i, j]: rover i reacting to neighbour j.
        rel = xy[:, None, :] - xy[None, :, :]  # p_i - p_j
        dist = np.linalg.norm(rel, axis=-1)
        np.fill_diagonal(dist, np.inf)
        d_safe = self.radii[:, None] + self.radii[None, :] + margin_m
        sense = sense_factor * d_safe
        away = rel / np.maximum(dist, 1e-6)[..., None]
        with np.errstate(divide="ignore"):
            gain = (1.0 / dist - 1.0 / sense) / (1.0 / d_safe - 1.0 / sense)
        gain = np.clip(np.where(dist < sense, gain, 0.0), 0.0, 3.0)
        tangent = np.stack((-away[..., 1], away[..., 0]), axis=-1)  # away rotated +90 deg -> pass on the right
        push = speed * np.sum(gain[..., None] * (away + k_tangent * tangent), axis=1)

        # 1. Steering (rovers commanded to reverse keep their command).
        steer = (np.linalg.norm(push, axis=-1) > 1e-9) & (cmd[:, 0] >= 0.0)
        desired = cmd[:, :1] * fwd + push
        err = wrap_angle(np.arctan2(desired[:, 1], desired[:, 0]) - yaw)
        v = np.linalg.norm(desired, axis=-1) * np.clip(np.cos(err), 0.0, 1.0)
        cmd[steer, 0] = v[steer]
        cmd[steer, 1] += k_heading * err[steer]

        # 2. Time-to-collision speed limit, using everyone's steered velocity.
        vel = np.clip(cmd[:, 0], -self.max_lin, self.max_lin)[:, None] * fwd
        dp = -rel  # p_j - p_i
        dv = vel[None, :, :] - vel[:, None, :]  # v_j - v_i
        dv2 = np.sum(dv**2, axis=-1)
        t_star = -np.sum(dp * dv, axis=-1) / np.maximum(dv2, 1e-12)
        closest = np.linalg.norm(dp + dv * t_star[..., None], axis=-1)
        ahead = np.sum(fwd[:, None, :] * dp, axis=-1) > 0.0
        risky = ahead & (dv2 > 1e-12) & (t_star > 0.0) & (t_star < horizon_s) & (closest < d_safe)
        scale = np.min(np.where(risky, t_star / horizon_s, 1.0), axis=1)
        scale[np.any(ahead & (dist < d_safe), axis=1)] = 0.0
        cmd[:, 0] = np.where(cmd[:, 0] > 0.0, cmd[:, 0] * scale, cmd[:, 0])
        return cmd
