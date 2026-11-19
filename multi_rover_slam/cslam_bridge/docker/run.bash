#!/usr/bin/env bash
# Run the Swarm-SLAM (cslam) image with the repo mounted at /root/ws.
#   ./run.bash          -> interactive bash
#   ./run.bash CMD...   -> one-shot command (ROS + cslam overlay sourced)
# Rootless docker: the container's root maps to the host user, so files written to /root/ws stay yours.
REPO_DIR="$(cd "$(dirname "$(readlink -f "$0")")/../../.." && pwd)"
# Host time zone, so session folders and cslam log names match the run folders (local time).
OPTS=(--rm --gpus all --net host --ipc host -v "${REPO_DIR}:/root/ws" -v /etc/localtime:/etc/localtime:ro -w /root/ws)
# Shared-memory-only Fast DDS transport (cslam_bridge/config/fastdds_shm.xml): the default one loses most
# 640x480 image frames once several nodes read them. Applies to `docker exec` shells too.
PROFILE=/root/ws/multi_rover_slam/cslam_bridge/config/fastdds_shm.xml
OPTS+=(-e FASTRTPS_DEFAULT_PROFILES_FILE="${PROFILE}" -e FASTDDS_DEFAULT_PROFILES_FILE="${PROFILE}")
[[ -t 0 ]] && OPTS+=(-it)
if [[ $# -eq 0 ]]; then
    exec docker run "${OPTS[@]}" swarm_slam_cslam:jazzy bash
fi
exec docker run "${OPTS[@]}" swarm_slam_cslam:jazzy "$@"
