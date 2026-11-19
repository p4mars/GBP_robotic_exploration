#!/usr/bin/env python3
"""Compare Swarm-SLAM's newest merged pose graph of a SLAM session with the ground truth.

Reads the cslam logs of a session folder <run>/slam/<yy-mm-dd_HH-MM>_<vo|gt>/ (a run
folder means its newest session, slam/latest) and writes into that session folder:
a top-down GT / odometry / optimized trajectory plot,
loop_closures.png (image pairs of inter-robot loop closures) and ate.csv. The plot is
trajectories_indiv.png (one panel per rover, default) or trajectories.png (all rovers in
one plot, --plot combined).

Usage (from /root/ws):
    python3 multi_rover_slam/tools/eval_slam.py multi_rover_slam/out/multi_rover_setup/latest [--plot combined]
    python3 multi_rover_slam/tools/eval_slam.py <run>/slam/26-11-19_14-05_vo
"""

import argparse
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # multi_rover_slam/

from cslam_bridge import load_run  # noqa: E402
from cslam_bridge import session as sessions  # noqa: E402
from cslam_bridge.evaluate import (  # noqa: E402
    evaluate,
    loop_closure_errors,
    loop_closure_funnel,
    plot_loop_closures,
    plot_trajectories,
    plot_trajectories_indiv,
    session_summary,
    vo_health,
    write_ate_csv,
    write_lc_accuracy_csv,
    write_lc_funnel_csv,
    write_vo_health_csv,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("path", type=Path, help="SLAM session folder, or a run folder (its newest session)")
    parser.add_argument("--plot", choices=("indiv", "combined"), default="indiv",
                        help="indiv: one panel per rover (trajectories_indiv.png); combined: all rovers in one plot (trajectories.png)")
    args = parser.parse_args()

    session = sessions.resolve_session(args.path)
    run_dir = sessions.run_dir_of(session)
    streams = load_run(run_dir)
    result = evaluate(streams, sessions.logs_dir(session), sessions.bags_dir(session))
    print(f"session: {session} (odometry: {sessions.odom_mode(session)})")
    print(f"graph: {result.log_dir}")
    print(f"inter-robot loop closures: {len(result.loop_closures)}")
    print(f"{'robot':>5} {'keyframes':>9} {'ATE odom':>9} {'ATE SLAM':>9} {'(joint)':>8} {'RPE1m odom':>10} {'RPE1m SLAM':>10}  [m]")
    for t in result.tracks:
        print(f"{'r%d' % t.robot_id:>5} {len(t.keys):>9} {t.ate_odometry:>9.3f} {t.ate_optimized_anchored:>9.3f} "
              f"{t.ate_optimized:>8.3f} {t.rpe_odometry(1.0):>10.3f} {t.rpe_optimized(1.0):>10.3f}")

    write_ate_csv(result, session / "ate.csv")
    scene_path = run_dir / "scene.yaml"
    scene = yaml.safe_load(scene_path.read_text()) if scene_path.is_file() else None
    if args.plot == "indiv":
        plot_name = "trajectories_indiv.png"
        plot_trajectories_indiv(result, session / plot_name, scene, run_dir, session)
    else:
        plot_name = "trajectories.png"
        plot_trajectories(result, session / plot_name, scene)
    outputs = ["ate.csv", plot_name]
    lc_errors = loop_closure_errors(result, streams)
    if lc_errors:
        write_lc_accuracy_csv(lc_errors, session / "lc_accuracy.csv")
        outputs.append("lc_accuracy.csv")
    log = session / "pipeline.log"
    funnel = loop_closure_funnel(log) if log.is_file() else {}
    if funnel:
        write_lc_funnel_csv(funnel, session / "lc_funnel.csv")
        outputs.append("lc_funnel.csv")
    health = vo_health(log, {st.robot_id: len(st) for st in streams}) if log.is_file() else {}
    if health:
        print(f"{'robot':>5} {'frames':>6} {'VO processed':>12} {'VO tracked':>10}")
        for rid, h in health.items():
            print(f"{'r%d' % rid:>5} {h.frames:>6} {h.processed:>12} {h.tracked:>10}")
        write_vo_health_csv(health, session / "vo_health.csv")
        outputs.append("vo_health.csv")
    if plot_loop_closures(result, streams, session / "loop_closures.png"):
        outputs.append("loop_closures.png")
    summary = session_summary(result, lc_errors, funnel, health)
    (session / "summary.yaml").write_text(yaml.safe_dump(summary, sort_keys=False))
    outputs.append("summary.yaml")
    print("summary: " + ", ".join(f"{k}={v}" for k, v in summary.items()))
    print(f"wrote {', '.join(outputs)} to {session}")


if __name__ == "__main__":
    main()
