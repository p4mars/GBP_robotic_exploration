---
tags: [swarm-slam, cslam, srb, howto]
created: 2026-11-10
---

# Running the cslam bridge on a recorded run

Replays a `multi_rover_setup` run through MISTLab Swarm-SLAM (`cslam`, ROS 2 Jazzy) and evaluates the merged map against ground truth.

Code: `multi_rover_slam/cslam_bridge/` (branch `gbp`), details in `multi_rover_slam/cslam_bridge/README.md`.

## Prerequisites

A run recorded for SLAM:
- `record_every: 2`, i.e. 12.5 Hz (~2 cm between frames at 0.25 m/s)
- `depth_format: png16`
- enough steps (≥ 1000) and rovers whose paths overlap

First proper run: `2026-11-10_15-50-20__moon_surface__5rovers__orbit__640x480` (5 rovers, orbit, 1000 steps = 40 s).

> [!info] Docker image
> `swarm_slam_cslam:jazzy` contains ROS 2, RTAB-Map, rosbags, matplotlib and the CosPlace weights; Swarm-SLAM itself is built from `multi_rover_slam/swarm_slam/`.

## 1. Start the Swarm-SLAM container

On the host, from the repo root (mounts the repo at `/root/ws`):

```bash
./multi_rover_slam/cslam_bridge/docker/run.bash
```

Run everything below **inside** the container.

First time, and after editing `multi_rover_slam/swarm_slam/` (sources are on the host, not in the image):

```bash
(cd multi_rover_slam/swarm_slam && colcon build --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=Release)
source multi_rover_slam/swarm_slam/install/setup.bash
```

## 2. Set the run folder

Use the exact name, not `latest`, because `latest` moves to whatever you record next.

```bash
RUN=multi_rover_slam/out/multi_rover_setup/2026-11-10_15-50-20__moon_surface__5rovers__orbit__640x480
```

## 3a. First pass: check the pipeline with exact ground-truth odometry

Takes visual odometry out of the picture, so any failure comes from topics, frames or timestamps.

```bash
python3 multi_rover_slam/tools/run_to_bags.py $RUN --odom gt_drift --drift-trans 0
```

```bash
ros2 launch multi_rover_slam/cslam_bridge/launch/srb_rgbd.launch.py run_dir:=$RUN odom:=bag
```

```bash
python3 multi_rover_slam/tools/eval_slam.py $RUN
```

## 3b. Second pass: the real test with RTAB-Map visual odometry

Rewrites the bags without `/odom` (replaces the ones from 3a); RTAB-Map's `rgbd_odometry` computes the odometry during replay.

```bash
python3 multi_rover_slam/tools/run_to_bags.py $RUN
```

```bash
ros2 launch multi_rover_slam/cslam_bridge/launch/srb_rgbd.launch.py run_dir:=$RUN
```

```bash
python3 multi_rover_slam/tools/eval_slam.py $RUN
```

## What to expect and watch

- **Runtime:** the launch waits 15 s for the models to load, plays the bags at half speed (40 s of driving takes ~80 s), waits another 30 s, then shuts down by itself. About 2 minutes in total.
- **Loop closures live:** in a second shell in the same container:
  ```bash
  docker exec -it <container> bash
  source /opt/ros/jazzy/setup.bash
  ros2 topic echo /cslam/inter_robot_loop_closure
  ```
  Messages with `success: true` are inter-robot loop closures that passed.
- **Visual odometry health (3b):** if `rgbd_odometry` keeps logging lost tracking or very few inliers, the terrain is too low-texture. Use the 3a results for now.

> [!warning] First run ever
> The bridge has never run end to end before this, so the first errors are probably in the code, not in the data.

## Results

In `$RUN/slam/`:

| File | Contents |
|---|---|
| `trajectories.png` | Top-down view: ground truth, odometry, optimized; grey lines = inter-robot loop closures |
| `loop_closures.png` | Camera images of matched keyframe pairs, to check by eye |
| `ate.csv` | Per-rover position error (RMSE in m) of odometry and of optimized keyframes |
| `<date>_experiment_robot_<id>/` | Raw cslam logs (g2o pose graphs, keyframe timestamps) |

- In 3a, the odometry error should be about 0.
- The optimized positions only line up if the rovers were merged through loop closures. They are aligned to ground truth with a single transform for all rovers, so a wrong merge shows up as a displaced rover.
