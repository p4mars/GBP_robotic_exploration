"""Everything the overseer view draws, loaded once from a ``multi_rover_setup`` run folder.

All per-frame arrays are indexed by the recording's ``frame`` number, which the overseer
camera, ``trajectories.csv``, ``observations.csv`` and the rover cameras share.
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from cslam_bridge.run_data import STAMP_OFFSET_S, CameraStream, load_overseer, load_run
from multi_rover_setup.cameras import quat_xyzw_to_matrix
from multi_rover_setup.terrain import TERRAIN_FILE, load_terrain_geometry


@dataclass
class Observations:
    """Rows of ``observations.csv`` sorted by frame; :meth:`at` returns one frame's rows."""

    frame: np.ndarray
    observer: np.ndarray
    """rover index of the observing camera"""
    target: np.ndarray
    """rover index of the observed rover"""
    range_m: np.ndarray
    in_fov: np.ndarray
    visible: np.ndarray

    def at(self, frame: int) -> slice:
        lo, hi = np.searchsorted(self.frame, (frame, frame + 1))
        return slice(int(lo), int(hi))


@dataclass
class RunData:
    run_dir: Path
    scene: dict
    overseer: CameraStream
    overseer_frames: np.ndarray
    """``(F,)`` frame number of every overseer image, in order"""
    rovers: list[str]
    """rover labels in index order (``rover_0``, ...)"""
    radius_m: np.ndarray
    """``(R,)`` footprint radius per rover"""
    heading_offset_rad: np.ndarray
    """``(R,)`` heading of the rover's front relative to its root x axis"""
    time_s: np.ndarray
    """``(max_frame + 1,)`` simulation time per frame; NaN for frames that were not recorded"""
    rover_pos: np.ndarray
    """``(max_frame + 1, R, 3)`` ground-truth root positions; NaN where missing"""
    rover_rot: np.ndarray
    """``(max_frame + 1, R, 3, 3)`` ground-truth root orientations"""
    cameras: list[CameraStream]
    """rover front cameras, ``cameras[i]`` belongs to rover ``i``"""
    cam_pose: np.ndarray
    """``(max_frame + 1, R, 4, 4)`` ROS-optical pose of each rover's front camera; NaN where missing"""
    cam_hfov_rad: np.ndarray
    """``(R,)``"""
    observations: Observations
    terrain: list[tuple[str, np.ndarray]] | None
    """``(prim path, (N, 3) vertices)`` per terrain mesh from ``terrain.npz``; None if the run has none"""
    fps: float
    """recording rate [frames/s]"""

    @classmethod
    def load(cls, run_dir: str | Path) -> RunData:
        run_dir = Path(run_dir).resolve()
        scene = yaml.safe_load((run_dir / "scene.yaml").read_text())
        run_info = scene.get("run", {})
        rovers_info = run_info.get("rovers", [])
        rovers = [r["label"] for r in rovers_info]
        if not rovers:
            raise ValueError(f"{run_dir / 'scene.yaml'} lists no rovers under run.rovers")
        index = {label: i for i, label in enumerate(rovers)}

        overseer = load_overseer(run_dir)
        overseer_frames = np.array([int(p.stem) for p in overseer.rgb], dtype=int)

        time_s, rover_pos, rover_rot = _read_trajectories(run_dir / "trajectories.csv", index)
        n_frames = len(time_s)

        cameras = load_run(run_dir)
        cam_pose = np.full((n_frames, len(rovers), 4, 4), np.nan)
        hfov = np.full(len(rovers), np.nan)
        for stream in cameras:
            rover = index[stream.label.split("__", 1)[0]]
            frames = np.array([int(p.stem) for p in stream.rgb], dtype=int)
            keep = frames < n_frames
            cam_pose[frames[keep], rover] = stream.poses[keep]
            hfov[rover] = 2.0 * math.atan(0.5 * stream.resolution[0] / stream.K[0, 0])

        spawn = [r.get("spawn") or {} for r in rovers_info]
        radius = np.array([s.get("radius_m") or 0.4 for s in spawn], dtype=float)
        heading_offset = np.radians([s.get("heading_offset_deg") or 0.0 for s in spawn])

        terrain_path = run_dir / TERRAIN_FILE
        terrain = load_terrain_geometry(terrain_path) if terrain_path.is_file() else None

        step_dt = float(run_info.get("step_dt_s") or 0.04)
        record_every = int((scene.get("output") or {}).get("record_every") or 1)
        return cls(
            run_dir=run_dir,
            scene=scene,
            overseer=overseer,
            overseer_frames=overseer_frames,
            rovers=rovers,
            radius_m=radius,
            heading_offset_rad=np.asarray(heading_offset, dtype=float),
            time_s=time_s,
            rover_pos=rover_pos,
            rover_rot=rover_rot,
            cameras=cameras,
            cam_pose=cam_pose,
            cam_hfov_rad=hfov,
            observations=_read_observations(run_dir / "observations.csv", index),
            terrain=terrain,
            fps=1.0 / (step_dt * record_every),
        )

    # Per-frame accessors

    def overseer_index(self, frame: int) -> int:
        i = int(np.searchsorted(self.overseer_frames, frame))
        if i >= len(self.overseer_frames) or self.overseer_frames[i] != frame:
            raise KeyError(f"No overseer image for frame {frame}")
        return i

    def overseer_pose(self, frame: int) -> np.ndarray:
        return self.overseer.poses[self.overseer_index(frame)]

    def overseer_rgb(self, frame: int) -> Path:
        return self.overseer.rgb[self.overseer_index(frame)]

    def heading(self, frame: int) -> np.ndarray:
        """``(R, 2)`` unit vectors of each rover's driving direction in the world xy plane."""
        x_axis = self.rover_rot[frame, :, :2, 0]
        angle = np.arctan2(x_axis[:, 1], x_axis[:, 0]) + self.heading_offset_rad
        return np.stack((np.cos(angle), np.sin(angle)), axis=-1)

    def require_terrain(self) -> list[tuple[str, np.ndarray]]:
        if self.terrain is None:
            raise FileNotFoundError(
                f"{self.run_dir / TERRAIN_FILE} is missing (runs recorded before the terrain export). "
                "Recreate it in the SRB container:\n"
                f"  python3 multi_rover_slam/tools/multi_rover_setup.py --config {self.run_dir}/scene.yaml "
                "--snapshot-only --out multi_rover_slam/out/terrain_export\n"
                f"then copy <that snapshot folder>/{TERRAIN_FILE} into {self.run_dir}"
            )
        return self.terrain


def _read_trajectories(path: Path, index: dict[str, int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with open(path, newline="") as f:
        rows = [(int(r["frame"]), index[r["rover"]], float(r["time_s"]),
                 [float(r[k]) for k in ("x", "y", "z")], [float(r[k]) for k in ("qx", "qy", "qz", "qw")])
                for r in csv.DictReader(f) if r["rover"] in index]
    n_frames = max(r[0] for r in rows) + 1
    time_s = np.full(n_frames, np.nan)
    pos = np.full((n_frames, len(index), 3), np.nan)
    rot = np.full((n_frames, len(index), 3, 3), np.nan)
    for frame, rover, t, p, q in rows:
        time_s[frame] = t
        pos[frame, rover] = p
        rot[frame, rover] = quat_xyzw_to_matrix(q)
    return time_s, pos, rot


def _read_observations(path: Path, index: dict[str, int]) -> Observations:
    cols: dict[str, list] = {k: [] for k in ("frame", "observer", "target", "range_m", "in_fov", "visible")}
    if path.is_file():
        with open(path, newline="") as f:
            for r in csv.DictReader(f):
                observer = index.get(r["observer"].split("__", 1)[0])
                target = index.get(r["target"])
                if observer is None or target is None:
                    continue
                cols["frame"].append(int(r["frame"]))
                cols["observer"].append(observer)
                cols["target"].append(target)
                cols["range_m"].append(float(r["range_m"]))
                cols["in_fov"].append(r["in_fov"] == "1")
                cols["visible"].append(r["visible"] == "1")
    order = np.argsort(np.asarray(cols["frame"], dtype=int), kind="stable")
    return Observations(
        frame=np.asarray(cols["frame"], dtype=int)[order],
        observer=np.asarray(cols["observer"], dtype=int)[order],
        target=np.asarray(cols["target"], dtype=int)[order],
        range_m=np.asarray(cols["range_m"], dtype=float)[order],
        in_fov=np.asarray(cols["in_fov"], dtype=bool)[order],
        visible=np.asarray(cols["visible"], dtype=bool)[order],
    )


def time_of_stamp(stamp: float) -> float:
    """cslam/ROS stamp → simulation time (inverse of :data:`cslam_bridge.run_data.STAMP_OFFSET_S`)."""
    return stamp - STAMP_OFFSET_S
