#!/usr/bin/env python3
"""Convert a multi_rover_setup run folder into one rosbag2 per rover for Swarm-SLAM.

Writes the bags into a SLAM session folder, <run>/slam/<yy-mm-dd_HH-MM>_<vo|gt>/bags/r<id>/,
with /r<id>/color/image, /r<id>/depth/image, /r<id>/color/camera_info and, with
--odom gt_drift, /r<id>/odom. The odometry mode and drift parameters go to the
session's session.yaml. Without --session a new session is created (see cslam_bridge/session.py).

Usage (from /root/ws in the Swarm-SLAM container, or anywhere with `rosbags` installed):
    python3 multi_rover_slam/tools/run_to_bags.py multi_rover_slam/out/multi_rover_setup/latest              # for odom:=vo
    python3 multi_rover_slam/tools/run_to_bags.py <run> --odom gt_drift --drift-trans 0                # GT odometry
    python3 multi_rover_slam/tools/run_to_bags.py <run> --odom gt_drift --drift-trans 0.02 --drift-yaw 0.5
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # multi_rover_slam/

from cslam_bridge import load_run  # noqa: E402
from cslam_bridge import session as sessions  # noqa: E402
from cslam_bridge.bags import write_bag  # noqa: E402
from cslam_bridge.odometry import drifted_odometry  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--odom", choices=("vo", "gt_drift"), default="vo",
                        help="vo: no odometry in the bags (RTAB-Map computes it); gt_drift: ground truth + drift")
    parser.add_argument("--drift-trans", type=float, default=0.02, help="translation noise std as a fraction of the distance driven per frame [-]")
    parser.add_argument("--drift-yaw", type=float, default=0.5, help="heading bias [deg per metre driven]")
    parser.add_argument("--drift-yaw-sigma", type=float, default=0.5, help="heading noise std, drawn per frame [deg per metre driven]")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--session", type=Path, default=None,
                        help="existing, still bag-less session folder to fill (default: create a new one)")
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    if not (run_dir / "scene.yaml").is_file():
        parser.error(f"not a run folder (no scene.yaml): {run_dir}")
    if args.session is None:
        session = sessions.new_session(run_dir, args.odom)
    else:
        session = args.session.resolve()
        if sessions.run_dir_of(session) != run_dir:
            parser.error(f"session {session} does not belong to run {run_dir}")
        if sessions.bags_dir(session).exists():
            parser.error(f"session {session} already has bags; start a new session instead")
        session.mkdir(parents=True, exist_ok=True)

    manifest = {"run": run_dir.name, "created": datetime.now().isoformat(timespec="seconds"), "odom": args.odom}
    if args.odom == "gt_drift":
        manifest["drift"] = {"trans": args.drift_trans, "yaw_deg_per_m": args.drift_yaw,
                             "yaw_sigma_deg_per_m": args.drift_yaw_sigma, "seed": args.seed}
    sessions.write_manifest(session, **manifest)

    for stream in load_run(run_dir):
        odometry = None
        if args.odom == "gt_drift":
            odometry = drifted_odometry(stream.poses, trans_sigma=args.drift_trans,
                                        yaw_bias_deg_per_m=args.drift_yaw,
                                        yaw_sigma_deg_per_m=args.drift_yaw_sigma, seed=args.seed + stream.robot_id)
        path = write_bag(stream, sessions.bags_dir(session) / f"r{stream.robot_id}", odometry)
        print(f"{stream.label} -> {path} ({len(stream)} frames)")
    print(f"session: {session}")


if __name__ == "__main__":
    main()
