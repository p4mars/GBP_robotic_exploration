"""Figure at the overseer resolution, layer assembly, and video/still output."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from .data import RunData
from .layers import (
    STYLES,
    FovLayer,
    HudLayer,
    Layer,
    ObservationLayer,
    PhotoLayer,
    RoverLayer,
    SlamLayer,
    Style,
    TrailLayer,
    ViewContext,
    WireframeLayer,
    rover_colors,
)

BACKGROUNDS = ("wireframe", "photo", "both")
OVERLAYS = ("fov", "trails", "slam", "observations", "rovers")


def make_layers(
    background: str = "wireframe",
    overlays: tuple[str, ...] | list[str] = ("fov", "trails", "observations", "rovers"),
    *,
    session: str | Path | None = None,
    grid_m: float = 0.5,
    trail_s: float = 20.0,
    fov_range_m: float = 4.0,
) -> list[Layer]:
    """Layers in drawing order: background, then overlays, then the time stamp."""
    if background not in BACKGROUNDS:
        raise ValueError(f"background must be one of {BACKGROUNDS}, got {background!r}")
    unknown = set(overlays) - set(OVERLAYS)
    if unknown:
        raise ValueError(f"Unknown overlays {sorted(unknown)}; choose from {OVERLAYS}")
    if "slam" in overlays and session is None:
        raise ValueError("The `slam` overlay needs a SLAM session")
    layers: list[Layer] = []
    if background in ("photo", "both"):
        layers.append(PhotoLayer(alpha=1.0 if background == "photo" else 0.6))
    if background in ("wireframe", "both"):
        layers.append(WireframeLayer(grid_m=grid_m))
    factories = {
        "fov": lambda: FovLayer(range_m=fov_range_m),
        "trails": lambda: TrailLayer(trail_s=trail_s),
        "slam": lambda: SlamLayer(session),
        "observations": ObservationLayer,
        "rovers": RoverLayer,
    }
    layers += [factories[name]() for name in OVERLAYS if name in overlays]
    layers.append(HudLayer())
    return layers


class OverseerFigure:
    """One matplotlib figure, pixel-aligned with the overseer image, redrawn per frame."""

    def __init__(self, data: RunData, layers: list[Layer], style: str | Style = "light", *, legend: bool = True,
                 dpi: int = 100):
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure

        self.data, self.layers = data, layers
        self.style = STYLES[style] if isinstance(style, str) else style
        width, height = data.overseer.resolution
        self.size = (width, height)
        self.fig = Figure(figsize=(width / dpi, height / dpi), dpi=dpi, facecolor=self.style.background)
        self.canvas = FigureCanvasAgg(self.fig)
        ax = self.fig.add_axes((0.0, 0.0, 1.0, 1.0))
        ax.set_xlim(0, width)
        ax.set_ylim(height, 0)
        ax.set_axis_off()
        ctx = ViewContext(data, self.style, rover_colors(len(data.rovers)))
        for layer in layers:
            layer.setup(ax, ctx)
        handles = [h for layer in layers for h in layer.legend()]
        if legend and handles:
            leg = ax.legend(handles=handles, loc="lower left", fontsize=8, framealpha=0.8,
                            facecolor=self.style.background, edgecolor=self.style.grid, labelcolor=self.style.text)
            leg.set_zorder(20)
        self.ax = ax

    def update(self, frame: int) -> None:
        for layer in self.layers:
            layer.update(frame)

    def draw(self, frame: int) -> np.ndarray:
        """``(H, W, 3)`` uint8 RGB image of ``frame``."""
        self.update(frame)
        self.canvas.draw()
        return np.asarray(self.canvas.buffer_rgba())[..., :3]

    def save(self, frame: int, stem: str | Path, formats: tuple[str, ...] = ("png", "pdf")) -> list[Path]:
        """Still of ``frame`` as ``<stem>.<fmt>``; PDF keeps lines as vectors."""
        self.update(frame)
        stem = Path(stem)
        stem.parent.mkdir(parents=True, exist_ok=True)
        paths = []
        for fmt in formats:
            path = stem.with_name(f"{stem.name}.{fmt}")
            self.fig.savefig(path, facecolor=self.fig.get_facecolor())
            paths.append(path)
        return paths


class VideoOut:
    """MP4 writer on OpenCV: H.264 (``avc1``) when available, else MPEG-4 (``mp4v``)."""

    def __init__(self, path: str | Path, fps: float, size: tuple[int, int]):
        try:
            import cv2
        except ImportError as err:
            raise RuntimeError("Writing video needs OpenCV (cv2): run in the Swarm-SLAM container "
                               "(multi_rover_slam/cslam_bridge/docker/run.bash), or render --stills only") from err
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        for codec in ("avc1", "mp4v"):
            writer = cv2.VideoWriter(str(self.path), cv2.VideoWriter_fourcc(*codec), float(fps), size)
            if writer.isOpened():
                self.codec, self._writer = codec, writer
                break
            writer.release()
        else:
            raise RuntimeError(f"OpenCV could not open an MP4 writer for {self.path}")

    def write(self, rgb: np.ndarray) -> None:
        self._writer.write(np.ascontiguousarray(rgb[..., ::-1]))  # OpenCV wants BGR

    def close(self) -> None:
        self._writer.release()

    def __enter__(self) -> VideoOut:
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def render_video(figure: OverseerFigure, frames, path: str | Path, fps: float, *, progress_every: int = 100) -> Path:
    frames = list(frames)
    start = time.monotonic()
    with VideoOut(path, fps, figure.size) as out:
        for i, frame in enumerate(frames, 1):
            out.write(figure.draw(frame))
            if progress_every and (i % progress_every == 0 or i == len(frames)):
                rate = i / max(time.monotonic() - start, 1e-9)
                print(f"  {i}/{len(frames)} frames, {rate:.1f} frames/s, ~{(len(frames) - i) / rate:.0f} s left",
                      flush=True)
        codec = out.codec
    print(f"wrote {path} ({len(frames)} frames at {fps:g} fps, {codec}, {time.monotonic() - start:.0f} s)")
    return Path(path)
