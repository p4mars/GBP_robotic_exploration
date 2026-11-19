#!/usr/bin/env python3
"""Compare the summary.yaml of several SLAM sessions side by side (written by eval_slam.py).

Usage:
    python3 multi_rover_slam/tools/compare_sessions.py <run_dir>                 # all sessions of a run
    python3 multi_rover_slam/tools/compare_sessions.py <session> <session> ...
"""

import argparse
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # multi_rover_slam/

from cslam_bridge import session as sessions  # noqa: E402

COLUMNS = [
    ("merged", "merged"), ("inter_robot_lc", "LC"), ("verified", "verif ok"), ("verifications", "verif"),
    ("lc_baseline_median_m", "LC base med"), ("lc_baseline_max_m", "LC base max"), ("lc_error_max_m", "LC err max"),
    ("ate_odometry_m", "ATE odom"), ("ate_slam_anchored_m", "ATE SLAM"), ("ate_slam_joint_m", "ATE joint"),
    ("rpe1m_odometry_m", "RPE1 odom"), ("rpe1m_slam_m", "RPE1 SLAM"), ("rpe5m_odometry_m", "RPE5 odom"), ("rpe5m_slam_m", "RPE5 SLAM"),
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("paths", type=Path, nargs="+")
    args = parser.parse_args()
    found = []
    for path in args.paths:
        slam = path / sessions.SLAM_DIR
        found += sorted(p for p in slam.iterdir() if sessions.is_session(p) and not p.is_symlink()) if slam.is_dir() else [path]
    rows = [(p.name, yaml.safe_load((p / "summary.yaml").read_text())) for p in found if (p / "summary.yaml").is_file()]
    if not rows:
        sys.exit("no summary.yaml found; run eval_slam.py on the sessions first")
    width = max(len(name) for name, _ in rows)
    print(f"{'session':<{width}} " + " ".join(f"{label:>11}" for _, label in COLUMNS))
    for name, summary in rows:
        cells = []
        for key, _ in COLUMNS:
            v = summary.get(key)
            cells.append(f"{'–' if v is None else (f'{v:.3f}' if isinstance(v, float) else v):>11}")
        print(f"{name:<{width}} " + " ".join(cells))


if __name__ == "__main__":
    main()
