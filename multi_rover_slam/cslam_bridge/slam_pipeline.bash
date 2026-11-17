#!/usr/bin/env bash
# Replay one multi_rover_setup run through Swarm-SLAM and evaluate it:
#   rosbags (GT odometry) -> cslam replay -> eval plots/metrics.
#
#   ./multi_rover_slam/cslam_bridge/slam_pipeline.bash              # newest run (`latest`)
#   ./multi_rover_slam/cslam_bridge/slam_pipeline.bash <run_dir>    # a specific run
#
# Works from the host (starts the cslam container via docker/run.bash) and from
# inside that container (runs directly, no nested docker).
set -euo pipefail

HERE="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"
REPO_DIR="$(cd "${HERE}/../.." && pwd)"
cd "${REPO_DIR}"

RUN="${1:-multi_rover_slam/out/multi_rover_setup/latest}"
# Pin `latest` to the concrete folder now, so a new recording can't move it mid-pipeline.
# Repo-relative, so the same path is valid on the host and under /root/ws in the container.
RUN="$(realpath --relative-to="${REPO_DIR}" "${RUN}")"
if [[ ! -f "${RUN}/scene.yaml" ]]; then
    echo >&2 "[slam_pipeline] not a run folder (no scene.yaml): ${RUN}"
    exit 1
fi

# Only start the container if we are not already in it (the cslam image sets CSLAM_WS).
if [[ -z "${CSLAM_WS:-}" ]]; then
    echo "[slam_pipeline] host -> starting cslam container for ${RUN}"
    exec "${HERE}/docker/run.bash" bash "multi_rover_slam/cslam_bridge/$(basename "$0")" "${RUN}"
fi

echo "[slam_pipeline] run: ${RUN}"
python3 multi_rover_slam/tools/run_to_bags.py "${RUN}" --odom gt_drift --drift-trans 0
ros2 launch multi_rover_slam/cslam_bridge/launch/srb_rgbd.launch.py run_dir:="${RUN}" odom:=bag
python3 multi_rover_slam/tools/eval_slam.py "${RUN}"
echo "[slam_pipeline] done: ${RUN}/slam"