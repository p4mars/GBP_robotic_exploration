"""One folder per SLAM pipeline trigger ("session") inside a run folder.

    <run>/slam/
        latest -> <yy-mm-dd_HH-MM>_<vo|gt>[_<label>]      newest session
        <yy-mm-dd_HH-MM>_<vo|gt>[_<label>]/
            session.yaml       odometry mode, label and bag parameters (written by run_to_bags.py)
            srb_rgbd.yaml      cslam configuration used by this session (copied by slam_pipeline.bash)
            pipeline.log       console output of slam_pipeline.bash
            bags/r<id>/        rosbags replayed in this session
            logs/              cslam logs (<date>_experiment_robot_<id>/<date>/...)
            ate.csv, summary.yaml, lc_accuracy.csv, lc_funnel.csv, vo_health.csv, *.png   (eval_slam.py)

Every session owns its bags, so a new conversion never overwrites the bags of an
earlier session (or of one that is still replaying), and the evaluation reads the
odometry mode from ``session.yaml`` instead of guessing it from whatever bags exist.

CLI (used by ``slam_pipeline.bash``; prints the new session folder):
    python3 -m cslam_bridge.session new <run_dir> --odom vo|gt_drift [--label budget5]
"""

from __future__ import annotations

import argparse
import re
from datetime import datetime
from pathlib import Path

import yaml

SLAM_DIR = "slam"
LATEST = "latest"
MANIFEST = "session.yaml"
NAME_FORMAT = "%y-%m-%d_%H-%M"  # same as the run folders of multi_rover_setup
ODOM_SUFFIX = {"vo": "vo", "gt_drift": "gt"}
ODOM_LABEL = {"vo": "RTAB-Map VO", "gt_drift": "GT + drift"}


def bags_dir(session: str | Path) -> Path:
    return Path(session) / "bags"


def logs_dir(session: str | Path) -> Path:
    return Path(session) / "logs"


def run_dir_of(session: str | Path) -> Path:
    """``<run>`` of ``<run>/slam/<session>``."""
    return Path(session).resolve().parent.parent


def is_session(path: str | Path) -> bool:
    path = Path(path)
    return path.is_dir() and path.parent.name == SLAM_DIR and path.name != LATEST


def new_session(run_dir: str | Path, odom: str, now: datetime | None = None, label: str | None = None) -> Path:
    """Create ``<run>/slam/<yy-mm-dd_HH-MM>_<vo|gt>[_<label>][_<n>]/`` and point ``slam/latest`` at it.

    ``label`` names a configuration variant (letters, digits, ``-``, ``.``), e.g. ``budget5``.
    """
    if odom not in ODOM_SUFFIX:
        raise ValueError(f"odom must be one of {sorted(ODOM_SUFFIX)}, got {odom!r}")
    if label is not None and not re.fullmatch(r"[A-Za-z0-9.-]+", label):
        raise ValueError(f"label may only contain letters, digits, '-' and '.', got {label!r}")
    slam = Path(run_dir) / SLAM_DIR
    slam.mkdir(parents=True, exist_ok=True)
    base = f"{(now or datetime.now()):{NAME_FORMAT}}_{ODOM_SUFFIX[odom]}" + (f"_{label}" if label else "")
    name, n = base, 1
    while (slam / name).exists():  # two triggers within one minute
        n += 1
        name = f"{base}_{n}"
    session = slam / name
    session.mkdir()
    link = slam / LATEST
    if link.is_symlink() or link.exists():
        link.unlink()
    link.symlink_to(name)
    if label:
        write_manifest(session, label=label)
    return session


def resolve_session(path: str | Path) -> Path:
    """A session folder as given, or the newest session of a run folder (``slam/latest``)."""
    path = Path(path).resolve()
    if is_session(path):
        return path
    slam = path / SLAM_DIR
    if not slam.is_dir():
        raise FileNotFoundError(f"{path} is neither a SLAM session nor a run folder with {SLAM_DIR}/")
    if (slam / LATEST).exists():
        return (slam / LATEST).resolve()
    sessions = sorted(p for p in slam.iterdir() if is_session(p) and not p.is_symlink())
    if not sessions:
        raise FileNotFoundError(f"No SLAM session under {slam}; run slam_pipeline.bash first")
    return sessions[-1]


def write_manifest(session: str | Path, **fields) -> Path:
    path = Path(session) / MANIFEST
    data = read_manifest(session)
    data.update(fields)
    path.write_text(yaml.safe_dump(data, sort_keys=False))
    return path


def read_manifest(session: str | Path) -> dict:
    path = Path(session) / MANIFEST
    return (yaml.safe_load(path.read_text()) or {}) if path.is_file() else {}


def odom_mode(session: str | Path) -> str | None:
    """``vo`` or ``gt_drift`` from ``session.yaml``; falls back to the bags (``/odom`` recorded or not)."""
    mode = read_manifest(session).get("odom")
    if mode:
        return mode
    metas = list(bags_dir(session).glob("r*/metadata.yaml"))
    if not metas:
        return None
    return "gt_drift" if any("/odom" in m.read_text() for m in metas) else "vo"


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a new SLAM session folder and print its path.")
    sub = parser.add_subparsers(dest="cmd", required=True)
    new = sub.add_parser("new")
    new.add_argument("run_dir", type=Path)
    new.add_argument("--odom", choices=sorted(ODOM_SUFFIX), required=True)
    new.add_argument("--label", default=None, help="configuration variant, appended to the folder name")
    args = parser.parse_args()
    if not (args.run_dir / "scene.yaml").is_file():
        parser.error(f"not a run folder (no scene.yaml): {args.run_dir}")
    print(new_session(args.run_dir, args.odom, label=args.label))


if __name__ == "__main__":
    main()
