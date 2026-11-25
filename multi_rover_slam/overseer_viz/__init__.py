"""Top view through the overseer camera: terrain wireframe plus data overlays drawn with matplotlib.

Isaac-free and ROS-free. Reads a ``multi_rover_setup`` run folder (overseer poses, trajectories,
observations, ``terrain.npz``) and optionally a cslam session; see ``tools/overseer_view.py``.
"""

from .data import RunData
from .render import BACKGROUNDS, OVERLAYS, OverseerFigure, make_layers, render_video

__all__ = ("BACKGROUNDS", "OVERLAYS", "OverseerFigure", "RunData", "make_layers", "render_video")
