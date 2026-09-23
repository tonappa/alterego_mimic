# alterego_mimic

Person following for the AlterEgo v2 robot, with a RealSense camera on the head.

The robot sees the person in front of it and, once started, does four things:

- **Keeps the distance.** It memorizes the distance at the start. When you come closer it backs off, and when you step back it comes forward.
- **Looks at your face.** The neck turns so that your face stays at the center of the camera image. The robot therefore never loses sight of you when you move sideways or crouch.
- **Turns toward you.** When the neck gets close to its limit because you keep moving sideways, the base turns the same way, so you can walk around the robot and it keeps facing you.
- **Mimics your arms.** The direction of your upper arm and the angle of your elbow are copied onto the shoulder and elbow cubes. By default it works like a mirror: your right arm moves its left arm. The upper arm goes at most `max_arm_elevation_deg` above horizontal (20° in `follower.yaml`).

On top of the mimicking, it recognizes a small **dance vocabulary** (see [Dance vocabulary V1](#dance-vocabulary-v1)):
open arms held → it moves away, hands on the chest held → it reaches its arms out and comes closer, one hand above
the head → pirouette.

You start and stop it with the Enter key, with a gesture (both hands above the head for 1 s), from the GUI, or with a ROS topic.

The package is standalone: it does not depend on `ego_dance` (in the dance branch of `AlterEGO_v2`), and the two packages can coexist in the same workspace. Do not run the old `ego_dance/pose_tracker.py` together with `person_tracker`: both publish `person_pose`. A browser GUI shows the robot, the person and the camera image. It runs either a built-in simulation (no ROS needed) or the real robot.

```
 robot base PC                                        any PC on the robot's ROS master
┌──────────────────────────────┐                     ┌──────────────────────────────────────┐
│ RealSense ─► person_tracker  │ ── person_* ──────► │ person_follower ─► cmd_vel           │
│   (conda env, YOLOv8-pose)   │     topics          │                 ─► arm cubes (eq,    │
└──────────────────────────────┘                     │                    preset)           │
                                                     │                 ─► neck cubes        │
 pitch_correction (robot PC or any PC) ─► offset_phi │   ▲ command            │ status      │
                                                     │   │                    ▼             │
                                                     │ gui/web_gui.py  ◄─► browser          │
                                                     └──────────────────────────────────────┘
```

---

## Contents

### `scripts/`

**`person_tracker.py`** is the perception node. It runs on the robot base PC, where the camera is plugged in.

- It opens the RealSense directly with `pyrealsense2`, the same way `alterego_object_detection` does: no `realsense2_camera` driver, depth aligned to color, 640×480 at 30 fps.
- At 15 Hz it runs YOLOv8-pose on the color image and keeps one person: the largest one, or the nearest one with `select:=nearest`.
- With the aligned depth it computes the 3D position of the torso and of every keypoint. Each depth value is the median of a 7×7 window.
- It reopens the camera if it fails or disconnects.
- It uses only standard messages and no `cv_bridge`, because it runs in the Python 3.10 of the conda env, where the compiled Noetic modules do not load.

Published topics, all under `/$ROBOT_NAME/`:

| Topic | Type | Content |
|---|---|---|
| `person_pose` | `std_msgs/Float64MultiArray` | `[found, width, height, 17 × (x, y, conf)]`. x and y are normalized to [0, 1], keypoints in COCO order. |
| `person_keypoints_3d` | `std_msgs/Float64MultiArray` | `[found, 17 × (X, Y, Z, valid)]` in meters, in the camera optical frame (x right, y down, z forward). |
| `person_position` | `geometry_msgs/PointStamped` | Torso center in meters. Published only when the depth is valid. |
| `person_face` | `std_msgs/Float64MultiArray` | `[found, yaw_left, pitch_up, depth]`: angles of the face from the camera axis, in radians. If the face is out of the image, it is estimated above the shoulders and `depth` is 0. |
| `person_tracker/image/compressed` | `sensor_msgs/CompressedImage` | Annotated image. Encoded only when someone subscribes. |

**`person_tracker_wrapper.sh`** starts `person_tracker.py` with the Python of a conda env (default `ros_yolo`), without hardcoded paths. It looks for the conda installation in `$CONDA_BASE`, `~/miniconda3`, `~/anaconda3`, `~/miniforge3` and `/opt/conda`, then falls back to `conda info --base`.

**`gestures.py`** recognizes the dance vocabulary on the person's 3D keypoints, in the person's body frame (no ROS, tested offline). It has the pose rules, the hold timers with a grace for missed frames, one event per hold and the cooldown. `person_follower` calls it once per camera frame while following.

**`person_follower.py`** is the behavior node. It uses plain ROS Python, so it runs on any PC connected to the robot's master.

- **Two states: idle and following.** At the start it memorizes the distance of the person.
- **Every cycle, at `rate` Hz (100 in `follower.yaml`), four independent behaviors:**
  - **Distance.** From `person_position` it computes `v = k_lin × (d − d_start)` with a dead band, then limits, filters and ramps it before sending it to `cmd_vel`. It never moves forward below `min_distance`.
  - **Head.** From `person_face`, the neck turns at `k_head × error` until the face is at the image center. This works also before the start, so the robot sees the gesture. If the face is lost, the head holds; after 3 s it goes back to the center.
  - **Base rotation.** Only while following, never in idle. When |neck yaw| exceeds `turn_start` (0.35 rad, about 60 % of the neck range), the base turns the same way at `turn_gain × (|neck yaw| − turn_stop)`, up to `turn_max` (0.4 rad/s), and stops when the neck is back within `turn_stop` (0.1 rad). The camera is on the head, so while the base turns the face moves back toward the image center and the neck recenters by itself. The command is `cmd_vel.angular.z`, positive to the left as in `ego_dance`; set `turn_sign: -1` if the base turns away from the person.
  - **Arms.** From `person_keypoints_3d` it builds the person's body frame (shoulders and hips). It takes the upper-arm direction and the elbow angle and maps them onto q0 (shoulder flexion), q1 (shoulder cube) and q3 (elbow). The mapping uses the robot's own kinematic model (`pitch_correction.yaml`), with a lookup table that avoids joint jumps near the shoulder singularity. q2 and q4 stay at 0. The pose is always chosen among the allowed ones (see *Arm safety limits* below), whatever the camera sees.
- **Safety:**
  - Person lost for 0.5 s (`person_timeout`): the base stops, both translation and rotation, and the arms hold their pose. This covers short occlusions.
  - Person lost for 2 s (`lost_stop_s`) while following: the robot stops following. The arms go down slowly (20 °/s), the head goes back to straight, and the node returns to idle. It does **not** resume by itself when someone appears again: a new start is needed. This happens, for example, if you cross in front of the robot faster than the neck and the base can follow.
  - |pitch| above 0.25 rad: it stops, and arms and head go back to zero.
  - Ctrl-C: the base stops, arms and head go back to zero at 20 °/s, then the node exits.
  - It refuses to start if `body_movement` nodes (`arm_inv_kin`, `arm_inv_dyn`, `head_inv_kin`) are running, because they write the same cubes.
  - If an option is switched off while running, the arms or the head first go back to zero, then the node stops publishing to them. When both *Keep the distance* and *Turn toward the person* are off, the node sends one zero velocity, then leaves `cmd_vel` free for the pilot.
- **Arm safety limits** (right arm; the left arm is mirrored). Values in `config/follower.yaml`:

  | Joint | Limit | What it prevents |
  |---|---|---|
  | q0, shoulder flexion | `q0_limits_deg: [-20, 110]` | At most 20° backward, at most 20° above horizontal forward. Set the minimum to 0 to never go backward. |
  | q1, shoulder cube | `q1_limits_deg: [-110, 10]` | Out to the side at most 20° above horizontal (or 20° behind the shoulder line with the arm horizontal), across the body at most 10° |
  | q3, elbow | `elbow_limits_deg: [0, 110]` | The elbow never bends the wrong way (below 0°) nor beyond 110° |
  | q2, q4 | fixed at 0 | No twist of the upper arm or of the wrist |
  | Upper arm | `max_arm_elevation_deg: 20` | At most 20° above horizontal, whatever q0 and q1 allow. 0 = never above horizontal |
  | Hand | `hand_above_shoulder_max: 0.2` | At most 0.2 m above the shoulder: the elbow is reduced if needed |
  | Speed | `joint_speed_deg_s: 60`, `descent_speed_deg_s: 20` | Every joint at most 60 °/s while mimicking, 20 °/s when going back to zero |
  | Wrist not seen | last elbow flexion kept | The elbow does not snap straight when the camera loses the wrist |

  These limits come from the kinematic model of the robot, not from the end stops of the cubes. A wrong detection by YOLO (left and right swapped, another person) can still cause an unexpected movement, but always within these limits and at limited speed.
- **Parameters** are read once at startup, from `config/follower.yaml` and the private params. Changes from the GUI or the command topic apply in memory until the follower restarts; to keep them, copy the values into `follower.yaml`. See *Changing the parameters* below.
- **Remote control topics**, used by the GUI in real-robot mode:
  - `person_follower/status` (`std_msgs/String`, JSON at 10 Hz): active, distance, velocity, commands, options, parameters.
  - `person_follower/command` (`std_msgs/String`, JSON): `{"action": "start" | "stop" | "toggle"}`, `{"option": name, "value": bool}`, `{"param": name, "value": x}`. Parameters are checked against a safe range.
  - `person_follower/toggle` (`std_msgs/Empty`): toggles start and stop.

### `launch/`

| File | What it starts | Arguments |
|---|---|---|
| `person_tracker.launch` | `person_tracker` through the conda wrapper. Run it on the robot base PC. | `conda_env`, `rate`, `conf`, `kp_conf`, `select`, `device`, `serial`, `show_window`, `weights` |
| `follower.launch` | `person_follower`. roslaunch gives the node no keyboard: start and stop it from the GUI, with the gesture or with the toggle topic. | `mirror`, `keep_distance`, `track_head`, `turn_base`, `mimic_arms` |
| `pitch_correction.launch` | `alterego_body_inv_kin/pitch_correction`, which `body_movement` normally starts. It moves the LQR balance point when the arms move the center of mass. | `AlterEgoVersion` (2) |

### `config/follower.yaml`

All the parameters of `person_follower`, with units and comments. They cover the behaviors, the gesture, the distance gains and limits, the head, the base rotation, the arm joint limits and speeds, and safety.

#### Changing the parameters

There are three ways, which differ in how long the change lasts:

| Where | How long it lasts | Which parameters |
|---|---|---|
| `config/follower.yaml` | Permanent. Restart the follower. | All |
| `rosrun ... _name:=value` | This run only | All. Lists in quotes: `_q0_limits_deg:="[0, 90]"` |
| GUI, *Settings* | Until the follower restarts. Effect immediate. | A tuning subset, each with a safe range (values outside are refused) |

```bash
rosrun alterego_mimic person_follower.py _q0_limits_deg:="[0, 90]" _elbow_limits_deg:="[0, 90]" _turn_sign:=-1
```

`follower.launch` exposes only the behavior switches as arguments (`mirror`, `keep_distance`, `track_head`, `turn_base`, `mimic_arms`). For the other parameters use the file or `rosrun`.

Parameters that the GUI can change at run time, with the accepted range:

| Parameter | Range | Parameter | Range |
|---|---|---|---|
| `k_lin` | 0 – 3 | `k_head` | 0 – 6 |
| `dead_lin` | 0 – 0.5 m | `head_dead` | 0 – 0.3 rad |
| `max_lin` | 0 – 0.5 m/s | `turn_start` | 0.1 – 0.6 rad |
| `min_distance` | 0.3 – 3 m | `turn_gain` | 0 – 3 |
| `joint_speed_deg_s` | 5 – 180 °/s | `turn_max` | 0 – 0.6 rad/s |
| `gesture_hold` | 0.2 – 5 s | `max_arm_elevation_deg` | −60 – 30° |
| `expand_rate` | 0 – 0.5 m/s | `attract_rate` | 0 – 0.5 m/s |
| `max_follow_distance` | 1 – 4 m | `attract_min_distance` | 0.6 – 2 m |
| `pirouette_speed` | 0.2 – 1.5 rad/s | `pirouette_min_distance` | 0.8 – 3 m |

The joint limits, the stiffness, the signs (`yaw_sign`, `pitch_sign`, `turn_sign`) and the safety timeouts are not in the GUI on purpose: they are set once, with the robot still, and need a restart of the follower.

### `gui/`

**`web_gui.py`** is the GUI server. It serves a page in the browser, with a toggle at the top center to switch between two modes:

- **Simulation.** `person_follower.py` runs inside the server on a simulated head camera. You move the person with the sliders or the keyboard, or play a scripted demo. Needs `numpy` and `pyyaml` only.
- **Real robot.** The server is a ROS node. It shows the measured arms and neck, the wheel odometry, the person seen by the tracker (in 3D and in the camera image) and the follower state and log. It sends start and stop, options and parameters to `person_follower`. It never commands the robot directly.

**`real_bridge.py`** is the real-robot backend of the server. It subscribes to `alterego_state/upperbody` and `lowerbody`, the tracker topics, the camera image, `person_follower/status` and `/rosout`. It publishes on `person_follower/command`.

**`web/index.html`** is the page: a three.js scene, the camera view, the readouts and the controls. It loads three.js from a CDN.

**`web/robot_meshes.json.gz`** holds the low-poly meshes of the robot for the page, generated by `sim/build_web_meshes.py`.

### `sim/`

| File | Role |
|---|---|
| `sim_core.py` | Closed-loop simulation shared by the GUI and the video. A person (scripted or interactive) is seen by a virtual RealSense on the robot head: 640×480, fx = fy = 615, 15 Hz, 0.15 s delay, depth noise, points outside the image not detected. The messages are built with the functions of `person_tracker.py`. The real `person_follower.py` runs on them with ROS stubbed out and a simulated clock. The robot executes the commands ideally: no LQR, no balance, no compliance. |
| `urdf_fk.py` | URDF forward kinematics in numpy. |
| `alterego_v2.urdf` | Copy of the expanded AlterEgo v2 URDF. |
| `simulate_follower.py`, `render_video.py`, `robot_model.py` | Scripted demo rendered to MP4 with the robot meshes. |
| `build_web_meshes.py` | Regenerates `gui/web/robot_meshes.json.gz`. Needs `yourdfpy`, `trimesh`, `fast_simplification`, `pycollada` and the `alterego_description` meshes. |

### `test/`

Offline tests, with no ROS, camera or robot:

| Test | What it checks |
|---|---|
| `test_offline.py` | Tracker geometry, message layouts, face estimate |
| `test_follower_offline.py` | Arm model and limits, mirror mapping, distance, stop, face tracking in closed loop with delay, parameters read once, commands and ranges, `cmd_vel` release, base rotation logic (never in idle, threshold, hysteresis, both sides, person lost), elbow kept when the wrist is not seen, person lost: base stopped, then idle with arms down and head straight, no automatic restart |
| `test_gui_real_offline.py` | Real-robot backend of the GUI with a fake ROS: odometry, person in the world frame, face marker, commands, errors |
| `test_vocabulary_offline.py` | Dance vocabulary: pose rules (EXPAND, ATTRACT, PIROUETTE, FREEZE, DROP), hold, grace, one event per hold, cooldown; in the simulator EXPAND and ATTRACT move the target distance, ATTRACT reaches the arms out, the pirouette turns 360° both ways with neck and arms, is ignored when the person is too close, both hands up still stop; one minute of random ordinary arm movements to count false detections |
| `test_sim_offline.py` | Closed loop: in idle the base never moves; while following, the person walks to 60°, 120° and −45° around the robot and the robot turns to face them, with the neck back near the center and the distance kept; the person crosses in front of the robot too fast, the robot loses them, stops, lowers the arms and straightens the head |

### Other files

- **`create_env.sh`** creates the conda env from `requirements.txt`, only if it does not exist yet.
- **`models/`** holds the YOLO weights (`yolov8n-pose.pt`), downloaded on the first run.

---

## Dance vocabulary V1

While the robot is following, `person_follower` also recognizes these gestures. They are recognized in the
person's body frame, so they do not depend on where the camera looks. Every one of them can be switched on and off
in `follower.yaml` or from the GUI (*Settings*).

| Gesture | You | The robot | Default |
|---|---|---|---|
| Start / stop | Both hands above the head, 1 s | Starts or stops following | on |
| `EXPAND` | Both arms open to the side, about at shoulder height, **held** | Its arms mimic yours, so it opens them too. While you hold the pose, the target distance grows by `expand_rate` (0.15 m/s) and the robot moves away. | on |
| `ATTRACT` | Both hands on the chest, **held** | It reaches both arms out toward you. While you hold the pose, the target distance shrinks by `attract_rate` and the robot comes closer. | on |
| `PIROUETTE` | One hand above the head, the other arm below the shoulder, 1 s | Opens its arms, turns its neck toward the turn, turns 360° on the spot, then lowers the arms, straightens the neck and goes back to mimicking | on |
| `FREEZE` | Arms crossed in front of the chest, held | Holds the arm pose, the base slows to a stop, the head keeps looking at you | **off** |
| `DROP` | Both hands moved down fast, from shoulder height | Arms down fast, head down, base still for `drop_hold_s`, then back to mimicking | **off** |

**Distance control with `EXPAND` and `ATTRACT`.** The longer you hold the pose, the more the target distance changes. When you release it, the robot keeps the new distance. The target stays between `attract_min_distance` (1.0 m, because the robot's hands reach about 0.4 m ahead while attracting) and `max_follow_distance` (3.0 m, the depth gets noisy beyond). The pose must be held for `expand_hold` (0.8 s) or `attract_hold` (0.5 s) before anything changes: a quick opening of the arms while dancing is only mimicked. `ATTRACT` needs the forearms pointing toward the body midline, so elbows bent with the forearms forward, a common mimic pose, are not taken for it.

**Pirouette.**
1. You raise one hand above the head for `pirouette_hold` (1 s), with the other arm below the shoulder. With both hands up it is the stop gesture, not a pirouette.
2. If you are closer than `pirouette_min_distance` (1.2 m), the gesture is ignored: the open arms sweep about 0.6 m around the robot. After an `ATTRACT` down to 1.0 m, use `EXPAND` before a pirouette.
3. Preparation (`pirouette_prepare_s`): arms open to `pirouette_spread_deg`, neck turned by `pirouette_head_yaw` toward the direction of the turn.
4. Turn: 360° at up to `pirouette_speed` (1.0 rad/s) with the acceleration `pirouette_acc`. The turned angle comes from the IMU yaw (`alterego_state/lowerbody`), not from the camera. Distance keeping, mimicking and the lost-person rule are suspended.
5. End (`pirouette_settle_s`): arms down, neck straight. The robot then has `pirouette_reacquire_s` (3 s) to find you again before the lost-person rule applies, and goes back to mimicking.

Direction: `pirouette_direction: 1` makes your **right** hand up turn the robot to **its left**, toward your right side, consistent with the mirror. The command goes through `turn_sign` like the base rotation, so the same sign fix applies. The speed is limited by the LQR's `wheels/max_ang_vel` in the robot's `general.yaml`; with 1.5 rad/s there, a pirouette at 1.0 rad/s takes about 7 s with the ramps.

**In the GUI**, the card on the right shows the recognized gesture (*Gesture*, e.g. `ATTRACT` or `PIROUETTE spin 58%`) and the target distance. In simulation, the chips *Open arms (expand)*, *Hands on chest (attract)*, *Right hand up (pirouette)* and *Left hand up (pirouette)* put the simulated person in each pose.

**Tuning.** All thresholds and timings are in the *dance vocabulary V1* section of `follower.yaml`. If `EXPAND` starts when you only wanted to open the arms, raise `expand_hold`. If a pose is not recognized, check the person in the camera view of the GUI: the hands and elbows must be in the image.

---

## Installation

The package is its own git repository. Clone it directly into `catkin_ws/src/`, next to `AlterEGO_v2` and not inside it, so it works with any branch of `AlterEGO_v2` (it needs only `alterego_msgs`, `alterego_robot` and `alterego_body_inv_kin`, which are in `main`).

On the **robot base PC** (for the tracker):

```bash
cd ~/catkin_ws/src
git clone <repository url> alterego_mimic
cd alterego_mimic && ./create_env.sh     # nothing to do if ros_yolo exists (alterego_object_detection)
cd ~/catkin_ws && catkin build alterego_mimic && source devel/setup.bash
```

The first tracker run downloads `yolov8n-pose.pt` into `models/`. Without internet on the robot PC, copy the file there by hand. The weights are excluded from git by `models/.gitignore`.

On **your PC** (for the follower and the GUI), clone the same repository into the workspace the same way and run `catkin build alterego_mimic`. The follower and the GUI need no conda.

To update both machines: `git pull` in `catkin_ws/src/alterego_mimic`. A rebuild is needed only if `package.xml` or `CMakeLists.txt` change.

---

## Running in simulation

No robot and no ROS needed, for example on a Mac. From the package folder:

```bash
pip install numpy pyyaml
python3 gui/web_gui.py            # opens http://127.0.0.1:8765, mode "Simulation"
```

The page loads three.js from a CDN, so the browser needs internet. The offline tests run the same way:

```bash
for t in test/*.py; do python3 $t; done
```

**Moving the person.** Use the sliders on the left, the arm presets, or the keyboard:

| Key | Action |
|---|---|
| `W` / `S` | Closer / farther |
| `A` / `D` | Sideways |
| `Q` / `E` | Turn the body |
| `C` | Crouch |
| `H` | Hands up (start/stop gesture) |
| `Space` | Start or stop following |
| `P` | Pause |

**Other controls.**

- *Settings* switches the five behaviors on and off (look at the face, turn toward the person, keep the distance, mimic the arms, mirror) and tunes the parameters live, with the same names as in `follower.yaml`.
- The card on the right shows what the head camera sees, the distance, the base velocity and rotation, the neck and the arm joints.
- *Play demo* runs the scripted scenario of the video.

To render the video of the scripted demo:

```bash
pip install yourdfpy trimesh fast_simplification pycollada matplotlib     # and ffmpeg
export ALTEREGO_DESCRIPTION=~/catkin_ws/src/AlterEGO_v2/alterego_description
python3 sim/simulate_follower.py && python3 sim/render_video.py
```

---

## Running on the real robot

### 1. Robot side

Start the usual robot launches, **without `body_movement`**:

- `imu`
- `body_activation`
- `wheels`
- `pilot`, which sets `CMD_VEL_IN_topic`

### 2. Tracker, on the robot base PC

Only one process can open the RealSense. Stop `realsense2_camera`, `alterego_object_detection` and `realsense-viewer` first.

```bash
export ROBOT_NAME=robot_alterego6
roslaunch alterego_mimic person_tracker.launch
```

### 3. Pitch correction, on the robot PC or on your PC

Keep this running as long as the arms can be raised.

```bash
roslaunch alterego_mimic pitch_correction.launch
```

### 4. Follower, on your PC

Use the robot's ROS master:

```bash
export ROS_MASTER_URI=http://192.168.88.90:11311 ROS_IP=<your PC> ROBOT_NAME=robot_alterego6
rosrun alterego_mimic person_follower.py        # Enter = start / stop
# or: roslaunch alterego_mimic follower.launch  (start / stop from the GUI or the gesture)
```

For the first test, run only the head: add `_keep_distance:=false _turn_base:=false _mimic_arms:=false`. Check that the neck turns toward the face. If it turns away, set `_yaw_sign:=-1` or `_pitch_sign:=-1`. Then add the arms, with low stiffness and the base still, then the base rotation (check the direction, `_turn_sign:=-1` if it turns away from you), then the distance.

### 5. GUI, on your PC

Use the same environment as the follower:

```bash
python3 $(rospack find alterego_mimic)/gui/web_gui.py --mode real
# inside the docker container: add --no-browser and open http://127.0.0.1:8765 on the host
```

- The toggle at the top center switches between *Simulation* and *Real robot*.
- The *Connections* card shows which data is live: robot state, wheels, tracker, camera image, follower.
- *Start following* and *Stop following* send the command to the follower.
- *Settings* changes behaviors and parameters on the robot until the follower restarts.

The GUI is not an emergency stop.

### 6. Shutdown

1. Stop following, from the GUI, with Enter or with the gesture.
2. Stop the follower with Ctrl-C and wait for the arms to be down.
3. Stop `pitch_correction`: the LQR keeps the last offset it received.
4. Stop the tracker.

---

## Notes

- **Neck.** In `alterego_v2.urdf` both neck joints are pitch axes. `head_inv_kin`, which drives the neck in teleoperation, uses the left cube as yaw and the right cube as pitch. The follower and the simulation follow `head_inv_kin`.
- **Arms.** The joint conventions (q0 flexion, q1 abduction, q3 elbow, left arm mirrored) match both `pitch_correction.yaml` and the URDF. Check them on the robot with the base still.
- **Elbow.** The robot copies how much your elbow is bent, not the direction of the bend. With q2 and q4 at 0 the forearm always bends in the robot's own plane (forward with the arm down): with your arm out to the side and the forearm up, the robot bends it forward. If the camera sees the shoulder and the elbow but not the wrist, the elbow keeps its last flexion.
- **Backing off.** When it keeps the distance, the robot backs off without seeing behind it. Leave free space behind the robot.
- **Rotation sign.** The LQR uses `des_yaw_rate = -cmd_vel.angular.z`, so `yaw_angle` decreases when the base turns left. The follower sends `angular.z > 0` to turn left, like `ego_dance`, and the GUI odometry assumes the same convention. Both still need a check on the robot: `turn_sign` fixes the follower.
- **What the simulation tests.** It checks the logic, not the dynamics: the camera intrinsics are assumed, and there is no balance and no compliance.
