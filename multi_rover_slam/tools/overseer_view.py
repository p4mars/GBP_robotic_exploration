#!/usr/bin/env python3
"""Render the overseer top view: terrain wireframe with rovers, observations, FOVs and SLAM estimates.

Needs <run>/terrain.npz (recorded since the terrain export; for older runs see the error message).
Video needs OpenCV, so run it in the Swarm-SLAM container; stills (--stills without --frames) work anywhere.

Usage (from the repo root):
    python3 multi_rover_slam/tools/overseer_view.py <run>                                  # video of all frames
    python3 multi_rover_slam/tools/overseer_view.py <run> --session <run>/slam/latest      # + SLAM keyframes, loop closures
    python3 multi_rover_slam/tools/overseer_view.py <run> --frames 100:300:2 --style dark
    python3 multi_rover_slam/tools/overseer_view.py <run> --stills 0 200 --background both # PNG + PDF stills, alignment check

Outputs: <run>/viz/overseer[_<session>].mp4 and <run>/viz/overseer[_<session>]_frame_<n>.{png,pdf}
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # multi_rover_slam/

from cslam_bridge import session as sessions  # noqa: E402
from overseer_viz import BACKGROUNDS, OVERLAYS, OverseerFigure, RunData, make_layers, render_video  # noqa: E402
from overseer_viz.layers import STYLES  # noqa: E402


def parse_frames(text: str, available) -> list[int]:
    """``a:b[:step]`` over recorded frame numbers (end inclusive; either end may be empty)."""
    parts = text.split(":")
    if not 1 <= len(parts) <= 3:
        raise argparse.ArgumentTypeError(f"--frames expects a:b[:step], got {text!r}")
    if len(parts) == 1:
        parts += [parts[0]]
    lo = int(parts[0]) if parts[0] else min(available)
    hi = int(parts[1]) if parts[1] else max(available)
    step = int(parts[2]) if len(parts) == 3 and parts[2] else 1
    have = set(int(f) for f in available)
    return [f for f in range(lo, hi + 1, step) if f in have]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("run", type=Path, help="multi_rover_setup run folder")
    parser.add_argument("--session", type=Path, help="cslam session folder (or a run folder: its newest session)")
    parser.add_argument("--frames", help="a:b[:step] frame range for the video (default: all)")
    parser.add_argument("--stills", type=int, nargs="+", default=[], help="frames to save as PNG + PDF")
    parser.add_argument("--layers", default=None,
                        help=f"comma-separated overlays from {','.join(OVERLAYS)} "
                             "(default: all; slam only with --session)")
    parser.add_argument("--background", choices=BACKGROUNDS, default="wireframe",
                        help="both = wireframe over the faded rendered image (alignment check)")
    parser.add_argument("--grid-m", type=float, default=0.5, help="wireframe cell size [m]")
    parser.add_argument("--trail-s", type=float, default=20.0, help="ground-truth trail length [s]")
    parser.add_argument("--fov-range-m", type=float, default=4.0, help="drawn camera FOV range [m]")
    parser.add_argument("--speed", type=float, default=1.0, help="playback speed relative to simulation time")
    parser.add_argument("--style", choices=sorted(STYLES), default="light")
    parser.add_argument("--no-legend", action="store_true")
    parser.add_argument("--out", type=Path, help="video path (default: <run>/viz/overseer[_<session>].mp4)")
    args = parser.parse_args()

    session = sessions.resolve_session(args.session) if args.session else None
    if args.layers is None:
        overlays = [o for o in OVERLAYS if o != "slam" or session is not None]
    else:
        overlays = [o.strip() for o in args.layers.split(",") if o.strip()]

    print(f"loading {args.run} ...", flush=True)
    data = RunData.load(args.run)
    layers = make_layers(args.background, overlays, session=session, grid_m=args.grid_m, trail_s=args.trail_s,
                         fov_range_m=args.fov_range_m)
    figure = OverseerFigure(data, layers, args.style, legend=not args.no_legend)

    name = "overseer" + (f"_{session.name}" if session is not None else "")
    video = args.out or data.run_dir / "viz" / f"{name}.mp4"
    for frame in args.stills:
        paths = figure.save(frame, video.with_name(f"{video.stem}_frame_{frame:06d}"))
        print("wrote " + ", ".join(str(p) for p in paths))

    if args.frames is None and args.stills:
        return
    frames = parse_frames(args.frames or ":", data.overseer_frames)
    if not frames:
        raise SystemExit(f"No recorded overseer frames in {args.frames!r}")
    step = frames[1] - frames[0] if len(frames) > 1 else 1
    render_video(figure, frames, video, fps=data.fps * args.speed / step)


if __name__ == "__main__":
    main()
