"""Drawing layers of the overseer view.

A layer creates its matplotlib artists once in :meth:`Layer.setup` and only changes their
data in :meth:`Layer.update`, so rendering thousands of frames stays cheap. Everything is
built in world coordinates and projected with the overseer pose of the frame being drawn.

A future GBP layer (beliefs as :func:`.geometry.ground_ellipse`, messages as lines between
rovers) follows the same pattern.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.colors import to_rgba
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from .data import RunData, time_of_stamp
from .geometry import (
    convex_hull_2d,
    fov_wedge,
    grid_lines,
    ground_circle,
    height_grid,
    offset_segments,
    project,
    shorten_segments,
)

###########
## Style ##
###########


@dataclass(frozen=True)
class Style:
    background: str
    grid: str
    grid_alpha: float
    rock: str
    text: str
    loop_closure: str


STYLES = {
    "light": Style(background="#ffffff", grid="#7d8590", grid_alpha=0.45, rock="#3d434b", text="#1f2328",
                   loop_closure="#d62728"),
    "dark": Style(background="#0d1117", grid="#6e7781", grid_alpha=0.55, rock="#adb5bd", text="#e6edf3",
                  loop_closure="#ff6b6b"),
}


def rover_colors(n: int) -> list[tuple[float, float, float, float]]:
    """One colour per rover; tab10 without its red, which marks loop closures."""
    from matplotlib import colormaps

    tab10 = [c for i, c in enumerate(colormaps["tab10"].colors) if i != 3]
    palette = tab10 if n <= len(tab10) else list(colormaps["tab20"].colors)
    return [to_rgba(palette[i % len(palette)]) for i in range(n)]


@dataclass
class ViewContext:
    data: RunData
    style: Style
    colors: list[tuple[float, float, float, float]]

    def project(self, frame: int, points_w: np.ndarray) -> np.ndarray:
        return project(self.data.overseer.K, self.data.overseer_pose(frame), points_w)


def _with_alpha(color, alpha) -> tuple[float, float, float, float]:
    return (*color[:3], float(alpha))


def _nan_joined(polylines: list[np.ndarray], dim: int) -> np.ndarray:
    """Concatenate polylines with NaN rows in between (one artist, many disconnected lines)."""
    if not polylines:
        return np.zeros((0, dim))
    gap = np.full((1, dim), np.nan)
    return np.concatenate([part for line in polylines for part in (line, gap)], axis=0)


class Layer:
    zorder = 1.0

    def setup(self, ax, ctx: ViewContext) -> None:
        self.ax, self.ctx = ax, ctx

    def update(self, frame: int) -> None:
        pass

    def legend(self) -> list:
        return []


##################
## Backgrounds ##
##################


class PhotoLayer(Layer):
    """The rendered overseer image of the frame."""

    zorder = 0.0

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha

    def setup(self, ax, ctx):
        super().setup(ax, ctx)
        width, height = ctx.data.overseer.resolution
        self.image = ax.imshow(np.zeros((height, width, 3), np.uint8), extent=(0, width, height, 0),
                               alpha=self.alpha, zorder=self.zorder, interpolation="antialiased")

    def update(self, frame):
        from PIL import Image

        with Image.open(self.ctx.data.overseer_rgb(frame)) as img:
            self.image.set_data(np.asarray(img.convert("RGB")))


class WireframeLayer(Layer):
    """Terrain as a projected height grid; meshes smaller than ``rock_max_m`` (rocks) as outlines."""

    zorder = 1.0

    def __init__(self, grid_m: float = 0.5, rock_max_m: float = 4.0):
        self.grid_m, self.rock_max_m = grid_m, rock_max_m

    def setup(self, ax, ctx):
        super().setup(ax, ctx)
        meshes = ctx.data.require_terrain()
        extent = [float(np.ptp(points[:, :2], axis=0).max()) for _, points in meshes]
        ground = [p for (_, p), e in zip(meshes, extent) if e > self.rock_max_m] or [p for _, p in meshes]
        rocks = [p for (_, p), e in zip(meshes, extent) if e <= self.rock_max_m and len(ground) < len(meshes)]
        self.nodes = height_grid(np.concatenate(ground, axis=0), self.grid_m)
        outlines = []
        for points in rocks:  # seen from above, a rock's outline is the hull of its xy footprint at its top
            hull = convex_hull_2d(points[:, :2])
            if len(hull) >= 3:
                outlines.append(np.column_stack((hull, np.full(len(hull), points[:, 2].max()))))
        self.rocks = _nan_joined(outlines, 3)
        style = ctx.style
        kw = dict(color=style.grid, alpha=style.grid_alpha, linewidth=0.45, zorder=self.zorder)
        (self.lines_x,) = ax.plot([], [], **kw)
        (self.lines_y,) = ax.plot([], [], **kw)
        (self.rock_lines,) = ax.plot([], [], color=style.rock, linewidth=1.0, zorder=self.zorder + 0.1)

    def update(self, frame):
        along_x, along_y = grid_lines(self.ctx.project(frame, self.nodes))
        self.lines_x.set_data(along_x[:, 0], along_x[:, 1])
        self.lines_y.set_data(along_y[:, 0], along_y[:, 1])
        if len(self.rocks):
            uv = self.ctx.project(frame, self.rocks)
            self.rock_lines.set_data(uv[:, 0], uv[:, 1])


##############
## Overlays ##
##############


class FovLayer(Layer):
    """Horizontal field of view of every rover's front camera, out to ``range_m``."""

    zorder = 2.0

    def __init__(self, range_m: float = 4.0):
        self.range_m = range_m

    def setup(self, ax, ctx):
        super().setup(ax, ctx)
        self.patches = PolyCollection([], linewidths=0.6, zorder=self.zorder)
        ax.add_collection(self.patches)

    def update(self, frame):
        data, verts, faces, edges = self.ctx.data, [], [], []
        for i, color in enumerate(self.ctx.colors):
            pose = data.cam_pose[frame, i]
            if not np.isfinite(pose).all():
                continue
            uv = self.ctx.project(frame, fov_wedge(pose, data.cam_hfov_rad[i], self.range_m))
            if np.isfinite(uv).all():
                verts.append(uv)
                faces.append(_with_alpha(color, 0.07))
                edges.append(_with_alpha(color, 0.35))
        self.patches.set_verts(verts)
        self.patches.set_facecolor(faces)
        self.patches.set_edgecolor(edges)

    def legend(self):
        hfov = np.degrees(np.nanmax(self.ctx.data.cam_hfov_rad))
        return [Patch(facecolor=_with_alpha(to_rgba(self.ctx.style.text), 0.08),
                      edgecolor=_with_alpha(to_rgba(self.ctx.style.text), 0.4),
                      label=f"camera FOV ({hfov:.0f}°, {self.range_m:g} m)")]


class TrailLayer(Layer):
    """Ground-truth path of the last ``trail_s`` seconds, fading with age."""

    zorder = 3.0

    def __init__(self, trail_s: float = 20.0, max_segments: int = 200):
        self.trail_s, self.max_segments = trail_s, max_segments

    def setup(self, ax, ctx):
        super().setup(ax, ctx)
        self.n = max(int(round(self.trail_s * ctx.data.fps)), 1)
        self.step = max(1, self.n // self.max_segments)
        self.lines = LineCollection([], linewidths=1.4, capstyle="round", zorder=self.zorder)
        ax.add_collection(self.lines)

    def update(self, frame):
        idx = np.arange(max(frame - self.n, 0), frame + 1, self.step)
        if idx[-1] != frame:
            idx = np.append(idx, frame)
        if len(idx) < 2:
            self.lines.set_segments([])
            return
        uv = self.ctx.project(frame, self.ctx.data.rover_pos[idx])  # (k, R, 2)
        segs = np.stack((uv[:-1], uv[1:]), axis=2).transpose(1, 0, 2, 3)  # (R, k-1, 2, 2)
        alpha = 0.85 * (1.0 - (frame - idx[1:]) / self.n)
        colors = np.array([[_with_alpha(c, a) for a in alpha] for c in self.ctx.colors])  # (R, k-1, 4)
        segs, colors = segs.reshape(-1, 2, 2), colors.reshape(-1, 4)
        keep = np.isfinite(segs).all(axis=(1, 2))
        self.lines.set_segments(segs[keep])
        self.lines.set_color(colors[keep])

    def legend(self):
        return [Line2D([], [], color=self.ctx.style.text, linewidth=1.4, alpha=0.7,
                       label=f"ground truth, last {self.trail_s:g} s")]


class ObservationLayer(Layer):
    """Rover-to-rover sightings from ``observations.csv``, in the observer's colour.

    Solid: the target is in the camera's FOV and not hidden by terrain; dashed: in the FOV
    but occluded. ``i → j`` and ``j → i`` are drawn as two parallel lanes.
    """

    zorder = 5.0

    def __init__(self, lane_m: float = 0.07):
        self.lane_m = lane_m

    def setup(self, ax, ctx):
        super().setup(ax, ctx)
        self.visible = LineCollection([], linewidths=1.3, zorder=self.zorder)
        self.occluded = LineCollection([], linewidths=0.9, linestyles=(0, (3, 3)), zorder=self.zorder)
        ax.add_collection(self.visible)
        ax.add_collection(self.occluded)

    def update(self, frame):
        data, obs = self.ctx.data, self.ctx.data.observations
        rows = obs.at(frame)
        observer, target = obs.observer[rows], obs.target[rows]
        in_fov, visible = obs.in_fov[rows], obs.visible[rows]
        for artist, mask, alpha in ((self.visible, in_fov & visible, 0.9), (self.occluded, in_fov & ~visible, 0.45)):
            o, t = observer[mask], target[mask]
            a, b = data.rover_pos[frame, o], data.rover_pos[frame, t]
            a, b = shorten_segments(a, b, data.radius_m[o], data.radius_m[t])
            a, b = offset_segments(a, b, self.lane_m)
            segs = np.stack((self.ctx.project(frame, a), self.ctx.project(frame, b)), axis=1)
            keep = np.isfinite(segs).all(axis=(1, 2))
            artist.set_segments(segs[keep])
            artist.set_color([_with_alpha(self.ctx.colors[i], alpha) for i in o[keep]])

    def legend(self):
        c = self.ctx.style.text
        return [Line2D([], [], color=c, linewidth=1.3, label="sees rover (observer colour)"),
                Line2D([], [], color=c, linewidth=0.9, linestyle=(0, (3, 3)), alpha=0.6, label="in FOV, occluded")]


class SlamLayer(Layer):
    """Keyframe positions of a cslam session, each shown from its stamp on, plus inter-robot loop closures.

    Uses the final optimized graph, aligned to ground truth with one transform for all rovers
    (:func:`cslam_bridge.evaluate.evaluate`); it is revealed over time, not re-estimated per frame.
    Positions are camera positions, so they sit slightly ahead of the rover centre.
    """

    zorder = 4.0

    def __init__(self, session: str | Path):
        self.session = Path(session)

    def setup(self, ax, ctx):
        super().setup(ax, ctx)
        from cslam_bridge import session as sessions
        from cslam_bridge.evaluate import evaluate

        data = ctx.data
        result = evaluate(data.cameras, sessions.logs_dir(self.session), sessions.bags_dir(self.session))
        rover_of = {i: data.rovers.index(cam.label.split("__", 1)[0]) for i, cam in enumerate(data.cameras)}
        self.tracks = []  # (rover index, times (K,), positions (K, 3))
        position_of: dict[int, tuple[float, np.ndarray]] = {}
        for track in result.tracks:
            if not track.merged or not len(track.optimized):
                continue
            times = np.array([time_of_stamp(result.stamps[k]) for k in track.keys])
            order = np.argsort(times, kind="stable")
            self.tracks.append((rover_of[track.robot_id], times[order], track.optimized[order]))
            position_of.update({k: (t, p) for k, t, p in zip(track.keys, times, track.optimized)})
        closures = [(max(position_of[a][0], position_of[b][0]), position_of[a][1], position_of[b][1])
                    for a, b in result.loop_closures if a in position_of and b in position_of]
        closures.sort(key=lambda c: c[0])
        self.lc_times = np.array([c[0] for c in closures])
        self.lc_points = np.array([(c[1], c[2]) for c in closures]).reshape(-1, 2, 3)

        self.lines, self.heads = [], []
        for rover, _, _ in self.tracks:
            color = ctx.colors[rover]
            (line,) = ax.plot([], [], color=color, linewidth=1.1, linestyle=(0, (4, 2)), alpha=0.95, zorder=self.zorder)
            (head,) = ax.plot([], [], marker="o", markersize=5, markerfacecolor="none", markeredgecolor=color,
                              markeredgewidth=1.3, linestyle="none", zorder=self.zorder + 0.1)
            self.lines.append(line)
            self.heads.append(head)
        self.closures = LineCollection([], colors=[ctx.style.loop_closure], linewidths=1.6, zorder=self.zorder + 0.2)
        ax.add_collection(self.closures)

    def update(self, frame):
        t = self.ctx.data.time_s[frame]
        for (_, times, positions), line, head in zip(self.tracks, self.lines, self.heads):
            uv = self.ctx.project(frame, positions[: int(np.searchsorted(times, t, side="right"))])
            line.set_data(uv[:, 0], uv[:, 1])
            head.set_data(uv[-1:, 0], uv[-1:, 1])
        n = int(np.searchsorted(self.lc_times, t, side="right"))
        segs = self.ctx.project(frame, self.lc_points[:n])
        self.closures.set_segments(segs[np.isfinite(segs).all(axis=(1, 2))])

    def legend(self):
        c = self.ctx.style.text
        return [Line2D([], [], color=c, linewidth=1.1, linestyle=(0, (4, 2)), marker="o", markerfacecolor="none",
                       markevery=[-1], label=f"SLAM keyframes ({self.session.name})"),
                Line2D([], [], color=self.ctx.style.loop_closure, linewidth=1.6, label="inter-robot loop closure")]


class RoverLayer(Layer):
    """Footprint circle, heading tick and label of every rover at its ground-truth pose."""

    zorder = 6.0

    def setup(self, ax, ctx):
        super().setup(ax, ctx)
        self.outlines, self.ticks, self.labels = [], [], []
        for i, color in enumerate(ctx.colors):
            (outline,) = ax.plot([], [], color=color, linewidth=1.8, zorder=self.zorder)
            (tick,) = ax.plot([], [], color=color, linewidth=1.8, solid_capstyle="round", zorder=self.zorder)
            label = ax.text(0, 0, f"r{i}", color=color, fontsize=9, fontweight="bold", ha="center", va="bottom",
                            zorder=self.zorder, clip_on=True)
            self.outlines.append(outline)
            self.ticks.append(tick)
            self.labels.append(label)

    def update(self, frame):
        data = self.ctx.data
        positions, headings = data.rover_pos[frame], data.heading(frame)
        for i, (outline, tick, label) in enumerate(zip(self.outlines, self.ticks, self.labels)):
            p, r = positions[i], data.radius_m[i]
            if not np.isfinite(p).all():
                outline.set_data([], [])
                tick.set_data([], [])
                label.set_visible(False)
                continue
            circle = self.ctx.project(frame, ground_circle(p, r))
            nose = self.ctx.project(frame, np.array([p, p + np.array([*(1.6 * r * headings[i]), 0.0])]))
            outline.set_data(circle[:, 0], circle[:, 1])
            tick.set_data(nose[:, 0], nose[:, 1])
            if np.isfinite(circle).all():
                label.set_position((float(circle[:, 0].mean()), float(circle[:, 1].min()) - 3.0))
            label.set_visible(bool(np.isfinite(circle).all()))


class HudLayer(Layer):
    """Simulation time and frame number in the top-left corner."""

    zorder = 10.0

    def setup(self, ax, ctx):
        super().setup(ax, ctx)
        self.text = ax.text(12, 12, "", color=ctx.style.text, fontsize=10, family="monospace", ha="left", va="top",
                            zorder=self.zorder,
                            bbox=dict(boxstyle="round,pad=0.3", facecolor=ctx.style.background, alpha=0.75,
                                      edgecolor="none"))

    def update(self, frame):
        self.text.set_text(f"t = {self.ctx.data.time_s[frame]:6.1f} s   frame {frame}")
