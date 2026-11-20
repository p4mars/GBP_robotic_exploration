"""Replay a multi_rover_setup run through Swarm-SLAM (cslam), one namespace /r<id> per rover.

Per rover: cslam's three nodes, optionally RTAB-Map RGB-D visual odometry, and
``ros2 bag play`` of ``<session>/bags/r<id>``, where ``<session>`` is a SLAM session folder
``<run>/slam/<yy-mm-dd_HH-MM>_<vo|gt>`` written by tools/run_to_bags.py. cslam logs (g2o
graphs, keyframe stamps) go to ``<session>/logs``. The launch shuts down ``linger_s`` after
the last bag finished, which leaves time for a final pose graph optimization.

It fails fast instead of hanging: if a cslam or odometry node dies, or the replay overruns
``timeout_s``, it writes ``<session>/replay_failed.txt`` with the reason and shuts down
(slam_pipeline.bash then stops before the evaluation).

    ros2 launch multi_rover_slam/cslam_bridge/launch/srb_rgbd.launch.py session:=<run>/slam/latest [odom:=auto|vo|bag]

odom:=auto (default) from the session's session.yaml: vo for --odom vo bags, bag for --odom gt_drift
odom:=vo   RTAB-Map rgbd_odometry on the replayed images (bags written with --odom vo)
odom:=bag  odometry recorded in the bags (bags written with --odom gt_drift)
"""

import os
import sys
import time
from pathlib import Path

from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, ExecuteProcess, OpaqueFunction, RegisterEventHandler, SetEnvironmentVariable,
                            Shutdown, TimerAction)
from launch.event_handlers import OnProcessExit, OnShutdown
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
import yaml

CONFIG = Path(__file__).resolve().parent.parent / "config" / "srb_rgbd.yaml"
# Shared-memory-only Fast DDS transport; with the default one most image frames are lost (see the XML).
FASTDDS_PROFILE = Path(__file__).resolve().parent.parent / "config" / "fastdds_shm.xml"
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # multi_rover_slam/, for cslam_bridge.session
from cslam_bridge import session as sessions  # noqa: E402


FAILED_MARKER = "replay_failed.txt"


def bag_duration_s(bag: Path) -> float:
    meta = yaml.safe_load((bag / "metadata.yaml").read_text())["rosbag2_bagfile_information"]
    return meta["duration"]["nanoseconds"] * 1e-9


def robot_nodes(robot_id: int, num_robots: int, args: dict) -> list:
    ns = f"/r{robot_id}"
    params = [
        args["config"],
        {
            "robot_id": robot_id,
            "max_nb_robots": num_robots,
            "frontend.nn_checkpoint": args["nn_checkpoint"],
            "evaluation.log_folder": str(sessions.logs_dir(args["session"])),
        },
    ]
    nodes = [
        Node(package="cslam", executable="loop_closure_detection_node.py", name="cslam_loop_closure_detection",
             namespace=ns, parameters=params),
        Node(package="cslam", executable="map_manager", name="cslam_map_manager", namespace=ns, parameters=params),
        Node(package="cslam", executable="pose_graph_manager", name="cslam_pose_graph_manager",
             namespace=ns, parameters=params),
    ]
    if args["odom"] == "vo":
        # As cslam_experiments' rtabmap_rgbd_odometry.launch.py, but the odometry child frame is
        # the camera optical frame (no TF needed) and each rover gets its own odom frame.
        nodes.append(Node(
            package="rtabmap_odom", executable="rgbd_odometry", name="rgbd_odometry", namespace=ns, output="screen",
            parameters=[{
                "frame_id": f"r{robot_id}_camera",
                "odom_frame_id": f"r{robot_id}_odom",
                "publish_tf": False,
                "subscribe_depth": True,
                "approx_sync": False,  # bag images, depth and camera_info share exact stamps
                "Vis/MinInliers": "10",
            }],
            remappings=[("rgb/image", "color/image"), ("rgb/camera_info", "color/camera_info"),
                        ("depth/image", "depth/image"), ("odom", "odom")],
        ))
    return nodes


def launch_setup(context, *_, **__):
    args = {name: LaunchConfiguration(name).perform(context)
            for name in ("session", "odom", "rate", "start_delay_s", "linger_s", "timeout_s", "nn_checkpoint",
                         "config")}
    session = sessions.resolve_session(args["session"])  # follows `latest`; a run folder means its newest session
    args["session"] = str(session)
    if args["odom"] == "auto":
        mode = sessions.odom_mode(session)
        if mode is None:
            raise FileNotFoundError(f"Cannot tell the odometry mode of {session}: no session.yaml and no bags")
        args["odom"] = "bag" if mode == "gt_drift" else "vo"
    if args["odom"] not in ("vo", "bag"):
        raise ValueError(f"odom must be auto, vo or bag, got {args['odom']!r}")
    bags = sorted(sessions.bags_dir(session).glob("r[0-9]*"), key=lambda p: int(p.name[1:]))
    if not bags:
        raise FileNotFoundError(f"No bags under {sessions.bags_dir(session)}; run tools/run_to_bags.py first")
    marker = session / FAILED_MARKER
    marker.unlink(missing_ok=True)
    start_delay, linger, rate = float(args["start_delay_s"]), float(args["linger_s"]), float(args["rate"])
    if args["timeout_s"] == "auto":
        # Playback is paced by the wall clock, so it should end close to duration / rate.
        replay_s = max(bag_duration_s(b) for b in bags) / rate
        timeout = start_delay + 1.1 * replay_s + linger + 120.0
    else:
        timeout = float(args["timeout_s"])
    print(f"[srb_rgbd] session {session}, odom:={args['odom']}, {len(bags)} rovers, timeout {timeout:.0f} s")

    state = {"shutdown": False, "failed": False}

    def fail(reason: str) -> list:
        if state["shutdown"] or state["failed"]:
            return []
        state["failed"] = True
        print(f"[srb_rgbd] FAILED: {reason}", flush=True)
        marker.write_text(f"{time.strftime('%Y-%m-%dT%H:%M:%S')} {reason}\n")
        return [Shutdown(reason=reason)]

    def on_node_exit(event, _context):
        # Nodes never exit on their own; any exit before our shutdown is a crash (e.g. a gtsam exception).
        if state["shutdown"] or state["failed"]:
            return None
        return fail(f"{event.action.name} exited with code {event.returncode}")

    def on_shutdown(_event, _context):
        state["shutdown"] = True  # nodes exiting from here on are being stopped, not crashing

    actions, players = [RegisterEventHandler(OnShutdown(on_shutdown=on_shutdown))], []
    for bag in bags:
        nodes = robot_nodes(int(bag.name[1:]), len(bags), args)
        actions += nodes
        actions += [RegisterEventHandler(OnProcessExit(target_action=n, on_exit=on_node_exit)) for n in nodes]
        players.append(ExecuteProcess(cmd=["ros2", "bag", "play", str(bag), "--rate", args["rate"]], output="screen"))
    # All bags start together once every node had time to load its model.
    actions.append(TimerAction(period=start_delay, actions=players))

    finished = set()

    def on_bag_done(event, _context):
        finished.add(event.action)
        if len(finished) == len(players):
            return [TimerAction(period=linger, actions=[Shutdown(reason="all bags replayed")])]
        return None

    actions += [RegisterEventHandler(OnProcessExit(target_action=p, on_exit=on_bag_done)) for p in players]
    # Watchdog: a stalled bag player (seen after node crashes) would otherwise keep the launch alive forever.
    actions.append(TimerAction(period=timeout, actions=[OpaqueFunction(
        function=lambda _ctx: fail(f"replay timeout after {timeout:.0f} s, {len(finished)}/{len(players)} bags finished"))]))
    return actions


def generate_launch_description():
    profile = os.environ.get("FASTRTPS_DEFAULT_PROFILES_FILE") or str(FASTDDS_PROFILE)
    return LaunchDescription([
        SetEnvironmentVariable("FASTRTPS_DEFAULT_PROFILES_FILE", profile),
        SetEnvironmentVariable("FASTDDS_DEFAULT_PROFILES_FILE", profile),
        DeclareLaunchArgument("session", description="SLAM session folder <run>/slam/<name> (or a run folder: its newest session)"),
        DeclareLaunchArgument("odom", default_value="auto", description="auto (from session.yaml) | vo | bag"),
        DeclareLaunchArgument("rate", default_value="0.5", description="bag playback rate"),
        DeclareLaunchArgument("start_delay_s", default_value="15.0", description="wait before playing bags"),
        DeclareLaunchArgument("linger_s", default_value="30.0", description="wait after the bags before shutdown"),
        DeclareLaunchArgument("timeout_s", default_value="auto",
                              description="give up (replay_failed.txt) after this many seconds; auto: from the bag durations"),
        DeclareLaunchArgument("nn_checkpoint", default_value="/opt/cslam_models/resnet18_64.pth",
                              description="CosPlace weights (absolute path)"),
        DeclareLaunchArgument("config", default_value=str(CONFIG)),
        OpaqueFunction(function=launch_setup),
    ])
