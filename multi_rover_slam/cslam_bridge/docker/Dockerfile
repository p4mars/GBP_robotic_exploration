# Swarm-SLAM (cslam) for offline RGB-D runs on rosbag2 data.
# Adapted from lajoiepy/cslam_experiments docker/Dockerfile.lidar, without TEASER++ (lidar only).
FROM ros:jazzy-perception
SHELL ["/bin/bash", "-c"]
ENV DEBIAN_FRONTEND=noninteractive

# Tools + RTAB-Map (cslam's RGB-D front end and the rgbd_odometry node from rtabmap_odom).
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl python3-pip python3-vcstool ros-jazzy-rtabmap-ros \
    && rm -rf /var/lib/apt/lists/*

# Swarm-SLAM system deps only. The sources live on the host in multi_rover_slam/swarm_slam/
# (editable, cslam = our fork) and are built inside the container with colcon (see README).
# The pinned sources are imported here just for rosdep, then deleted.
COPY swarm_slam.repos /tmp/
RUN mkdir /tmp/deps_src && vcs import /tmp/deps_src < /tmp/swarm_slam.repos \
    && apt-get update && rosdep update --rosdistro jazzy \
    && rosdep install --from-paths /tmp/deps_src --ignore-src -r -y \
    && rm -rf /var/lib/apt/lists/* /tmp/deps_src

# Python deps: Swarm-SLAM requirements.txt + rosbags (bag writing) + matplotlib (plots).
# setuptools<80 keeps colcon-core working for the in-container builds (torch pulls a newer one).
# psutil comes from Debian (no pip RECORD), so pip must replace it rather than uninstall it.
# numpy<2 keeps the apt numpy ABI that cv_bridge and other ROS Python bindings are built against.
RUN pip install --no-cache-dir --break-system-packages --ignore-installed psutil \
    && pip install --no-cache-dir --break-system-packages \
        "numpy<2" "setuptools<80" networkx numba open3d Pillow scikit_learn scipy sortedcontainers \
        torch torchvision distinctipy numpy-quaternion rosbags matplotlib

# CosPlace ResNet-18/64-d weights; the launch file passes this absolute path as frontend.nn_checkpoint.
RUN mkdir -p /opt/cslam_models \
    && curl -fsSL -o /opt/cslam_models/resnet18_64.pth \
        https://github.com/gmberton/CosPlace/releases/download/v1.0/resnet18_64_cosplace.pth

# cosplace_utils/network.py calls resnet18(pretrained=True): pre-fill the torch hub cache (offline runs).
RUN python3 -c "import torchvision; torchvision.models.resnet18(weights='IMAGENET1K_V1')"

# Source ROS, plus the host-built overlay once it exists (after the first colcon build).
ENV CSLAM_WS=/root/ws/multi_rover_slam/swarm_slam
RUN echo '[ -f $CSLAM_WS/install/setup.bash ] && source $CSLAM_WS/install/setup.bash' >> /root/.bashrc \
    && sed -i '/^source "\/opt\/ros\//a [ -f "$CSLAM_WS/install/setup.bash" ] \&\& source "$CSLAM_WS/install/setup.bash" --; true' /ros_entrypoint.sh
WORKDIR /root/ws
