# cslam_bridge: Swarm-SLAM on multi_rover_setup runs

Offline proof of concept: replay a recorded `multi_rover_setup` run through
[MISTLab Swarm-SLAM](https://github.com/MISTLab/Swarm-SLAM) (`cslam`, ROS 2 Jazzy), then
check the merged map against the ground truth.

```
SRB container                       Swarm-SLAM container (docker/)
multi_rover_setup.py ──> <run>/ ──> run_to_bags.py ──> <run>/slam/<session>/bags/r<id>/, session.yaml
                                    srb_rgbd.launch.py ──> <run>/slam/<session>/logs/ (cslam logs)
                                    eval_slam.py ──> <run>/slam/<session>/{trajectories_indiv,loop_closures}.png, ate.csv
```

Every pipeline trigger is a **session** `<run>/slam/<yy-mm-dd_HH-MM>_<vo|gt>/` with its own bags,
logs and results, so sessions never overwrite each other; `<run>/slam/latest` points to the newest
one (see `session.py`).

## Quick start

From the host (starts the container) or inside it:

```bash
./multi_rover_slam/cslam_bridge/slam_pipeline.bash                      # latest run, visual odometry
./multi_rover_slam/cslam_bridge/slam_pipeline.bash --odom gt <run_dir>  # ground-truth odometry
./multi_rover_slam/cslam_bridge/slam_pipeline.bash --config multi_rover_slam/cslam_bridge/config/variants/budget5.yaml --label budget5 <run_dir>
python3 multi_rover_slam/tools/compare_sessions.py <run_dir>             # summary.yaml of all sessions side by side
```

Configuration variants live in `config/variants/`; each session keeps a copy of the configuration it ran
with (`srb_rgbd.yaml`), and `--label` is appended to the session folder name.

## Steps

1. Record a run at 12.5 Hz in the SRB container:

   ```bash
   python3 multi_rover_slam/tools/multi_rover_setup.py --config multi_rover_slam/multi_rover_setup/presets/slam_playground.yaml
   ```

2. Run the remaining steps in the Swarm-SLAM container (`multi_rover_slam/cslam_bridge/docker/run.bash`).
   The image holds only dependencies; the Swarm-SLAM source is `multi_rover_slam/swarm_slam/`
   (editable, `cslam` is our fork). Build it once, and again after editing it:

   ```bash
   (cd multi_rover_slam/swarm_slam && colcon build --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=Release)
   source multi_rover_slam/swarm_slam/install/setup.bash   # new shells source it automatically
   ```

   Set the run folder:

   ```bash
   RUN=multi_rover_slam/out/multi_rover_setup/latest
   ```

3. Choose an odometry source and convert the bags. This creates a new session and points
   `$RUN/slam/latest` at it; the odometry mode is stored in its `session.yaml`:

   | Odometry | Convert | Session suffix |
   |---|---|---|
   | Visual odometry (RTAB-Map) | `python3 multi_rover_slam/tools/run_to_bags.py $RUN` | `_vo` |
   | Fallback: ground truth + drift (`--drift-trans 0` makes it exact) | `python3 multi_rover_slam/tools/run_to_bags.py $RUN --odom gt_drift` | `_gt` |

4. Replay the session. The launch picks `odom:=vo` or `odom:=bag` from `session.yaml`:

   ```bash
   ros2 launch multi_rover_slam/cslam_bridge/launch/srb_rgbd.launch.py session:=$RUN/slam/latest
   ```

5. Evaluate (a run folder means its newest session; pass a session folder for an older one):

   ```bash
   python3 multi_rover_slam/tools/eval_slam.py $RUN/slam/latest
   ```

While the launch runs, `ros2 topic echo /cslam/inter_robot_loop_closure` shows the loop-closure
attempts between rovers; `success: true` marks the ones that passed. The launch ends by itself
`linger_s` after the last bag finishes.

## DDS transport

The replay uses a shared-memory-only Fast DDS profile, `config/fastdds_shm.xml` (set via
`FASTRTPS_DEFAULT_PROFILES_FILE` by `docker/run.bash` and the launch file). With the default
transports a 640×480 image does not fit the 0.5 MB shared-memory segment and falls back to
fragmented UDP. Once `rgbd_odometry` and cslam's `map_manager` both read the images, most frames
were lost: VO lost tracking for good and cslam skipped every frame. With the profile all frames arrive.
Check `vo_health.csv` after a VO session: `vo_processed` should equal `frames`.

## Contract

| What | Value |
|---|---|
| Namespace / cslam `robot_id` | `/r<id>`, rovers in order `rover_0, rover_1, ...` |
| Colour image | `/r<id>/color/image` (`rgb8`) |
| Depth image | `/r<id>/depth/image` (`16UC1`, millimetres, 0 = invalid) |
| Camera info | `/r<id>/color/camera_info` (pinhole, no distortion) |
| Odometry | `/r<id>/odom` (`nav_msgs/Odometry`, `r<id>_odom` → `r<id>_camera`) |
| Sensor frame | `r<id>_camera` = ROS optical frame (z forward, x right, y down); no TF needed |
| Stamps | simulation time + 1 s (`run_data.STAMP_OFFSET_S`) |

The topic names and frames are defined in `messages.py`, and `config/srb_rgbd.yaml` must use the
same ones. A later live mode can publish the same messages from `MultiRoverSetup.record()` with
rclpy. The Swarm-SLAM side stays unchanged; only `ros2 bag play` is dropped.

## Outputs (`<run>/slam/<yy-mm-dd_HH-MM>_<vo|gt>/`)

| File | Contents |
|---|---|
| `session.yaml` | Odometry mode (`vo` / `gt_drift`), label, drift parameters, creation time |
| `srb_rgbd.yaml` | cslam configuration this session ran with (pipeline sessions) |
| `summary.yaml` | Scalar metrics for comparing sessions (`tools/compare_sessions.py`) |
| `lc_accuracy.csv` | Per inter-robot loop closure: GT baseline between the cameras and error of the measured relative pose |
| `lc_funnel.csv` | Per rover pair: geometric verifications attempted and accepted (from `pipeline.log`) |
| `pipeline.log` | Console output of `slam_pipeline.bash` (only for sessions it started) |
| `bags/r<id>/` | The rosbags replayed in this session |
| `logs/<date>_experiment_robot_<id>/<date>/` | Raw cslam logs: `optimized_global_pose_graph.g2o`, `initial_global_pose_graph.g2o`, `pose_timestamps<id>.csv` and `log.csv` |
| `trajectories_indiv.png` | One top-down panel per rover: ground truth, odometry and optimized; red circles mark inter-robot loop closures (default plot) |
| `trajectories.png` | All rovers in one plot (`eval_slam.py --plot combined`) |
| `loop_closures.png` | Camera images of the matched keyframe pairs, to check by eye |
| `ate.csv` | Per-rover ATE (RMSE in metres) of odometry and of optimized keyframe positions |
| `vo_health.csv` | VO sessions: per rover, frames in the bag vs. frames `rgbd_odometry` processed and tracked (from `pipeline.log`) |

Metrics (`ate.csv`, `summary.yaml`; means over merged rovers):

- **ATE odometry** and **ATE SLAM (anchored)**: each rover anchored at the GT pose of its first keyframe, so
  odometry and SLAM are compared on equal terms. Use these to compare methods per rover.
- **ATE SLAM (joint)**: one transform for all merged rovers (as drawn), so a wrong merge or inter-robot frame
  error shows up as a displaced rover. Measures global map consistency.
- **RPE 1 m / 5 m**: translation error of relative poses over 1 m / 5 m of path, independent of anchoring.

## Tests

The tests need neither Isaac nor ROS; `rosbags` must be installed for the bag round trip.

```bash
python3 -m pytest multi_rover_slam/cslam_bridge/tests -q
```
