#!/usr/bin/env bash
# Replay one multi_rover_setup run through Swarm-SLAM and evaluate it:
#   new session folder -> rosbags -> cslam replay -> eval plots/metrics.
#
#   ./multi_rover_slam/cslam_bridge/slam_pipeline.bash                       # newest run (`latest`), visual odometry
#   ./multi_rover_slam/cslam_bridge/slam_pipeline.bash --odom gt <run_dir>   # a specific run, ground-truth odometry
#   ./multi_rover_slam/cslam_bridge/slam_pipeline.bash --config multi_rover_slam/cslam_bridge/config/variants/budget5.yaml --label budget5 <run_dir>
#
# --odom vo  RTAB-Map visual odometry during replay (bags without /odom)        -> session suffix _vo
# --odom gt  ground-truth odometry in the bags (--odom gt_drift --drift-trans 0) -> session suffix _gt
# --config   cslam parameter file (default cslam_bridge/config/srb_rgbd.yaml); copied into the session
# --label    configuration variant name, appended to the session folder (e.g. _vo_budget5)
#
# Every call gets its own session folder <run>/slam/<yy-mm-dd_HH-MM>_<vo|gt>/ with the bags,
# cslam logs, pipeline.log and evaluation outputs; <run>/slam/latest points to it
# (layout: multi_rover_slam/cslam_bridge/session.py).
#
# Works from the host (starts the cslam container via docker/run.bash) and from
# inside that container (runs directly, no nested docker).
set -euo pipefail

HERE="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"
REPO_DIR="$(cd "${HERE}/../.." && pwd)"
cd "${REPO_DIR}"

ODOM=vo
CONFIG=multi_rover_slam/cslam_bridge/config/srb_rgbd.yaml
LABEL=""
POSITIONAL=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --odom) ODOM="$2"; shift 2 ;;
        --odom=*) ODOM="${1#*=}"; shift ;;
        --config) CONFIG="$2"; shift 2 ;;
        --label) LABEL="$2"; shift 2 ;;
        -h|--help) sed -n '2,20p' "$0"; exit 0 ;;
        *) POSITIONAL+=("$1"); shift ;;
    esac
done
case "${ODOM}" in
    vo) BAG_ARGS=(--odom vo) ;;
    gt) BAG_ARGS=(--odom gt_drift --drift-trans 0) ;;
    *) echo >&2 "[slam_pipeline] --odom must be vo or gt, got: ${ODOM}"; exit 1 ;;
esac

RUN="${POSITIONAL[0]:-multi_rover_slam/out/multi_rover_setup/latest}"
# Pin `latest` to the concrete folder now, so a new recording can't move it mid-pipeline.
# Repo-relative, so the same path is valid on the host and under /root/ws in the container.
RUN="$(realpath --relative-to="${REPO_DIR}" "${RUN}")"
CONFIG="$(realpath --relative-to="${REPO_DIR}" "${CONFIG}")"
if [[ ! -f "${CONFIG}" ]]; then
    echo >&2 "[slam_pipeline] config not found: ${CONFIG}"
    exit 1
fi
if [[ ! -f "${RUN}/scene.yaml" ]]; then
    echo >&2 "[slam_pipeline] not a run folder (no scene.yaml): ${RUN}"
    exit 1
fi

# Only start the container if we are not already in it (the cslam image sets CSLAM_WS).
if [[ -z "${CSLAM_WS:-}" ]]; then
    echo "[slam_pipeline] host -> starting cslam container for ${RUN} (odom: ${ODOM})"
    exec "${HERE}/docker/run.bash" bash "multi_rover_slam/cslam_bridge/$(basename "$0")" \
        --odom "${ODOM}" --config "${CONFIG}" ${LABEL:+--label "${LABEL}"} "${RUN}"
fi

SESSION="$(PYTHONPATH=multi_rover_slam python3 -m cslam_bridge.session new "${RUN}" --odom "${BAG_ARGS[1]}" ${LABEL:+--label "${LABEL}"})"
SESSION="$(realpath --relative-to="${REPO_DIR}" "${SESSION}")"
exec > >(tee -a "${SESSION}/pipeline.log") 2>&1

echo "[slam_pipeline] run: ${RUN}"
echo "[slam_pipeline] session: ${SESSION} (odom: ${ODOM}, config: ${CONFIG})"
cp "${CONFIG}" "${SESSION}/srb_rgbd.yaml"  # the exact parameters of this session
python3 multi_rover_slam/tools/run_to_bags.py "${RUN}" "${BAG_ARGS[@]}" --session "${SESSION}"
ros2 launch multi_rover_slam/cslam_bridge/launch/srb_rgbd.launch.py session:="${SESSION}" config:="${SESSION}/srb_rgbd.yaml"
python3 multi_rover_slam/tools/eval_slam.py "${SESSION}"
echo "[slam_pipeline] done: ${SESSION}"
