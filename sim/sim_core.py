"""Simulation core shared by the video (simulate_follower.py) and the web GUI (web_gui.py).

The REAL person_follower.py runs here, with ROS replaced by stubs and a simulated clock:
  person (scripted demo or interactive)  ->  virtual head camera  ->  tracker messages  ->  person_follower
  ->  commands executed ideally by the robot (base velocity, arm joints, neck). No LQR, no compliance.
Needs numpy and pyyaml only.
"""
import math
import os
import re
import sys
import types
from contextlib import contextmanager

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(PKG, "scripts"))
sys.path.insert(0, HERE)

# ================================================================================================ ROS stubs
ROBOT = "robot_alterego_sim"
EVENTS = []                 # (sim time, log line) from the follower
SIM_T = [0.0]


def _log(*a):
    try:
        msg = a[0] % a[1:] if len(a) > 1 else str(a[0])
    except TypeError:
        msg = " ".join(str(x) for x in a)
    EVENTS.append((SIM_T[0], msg))
    del EVENTS[:-200]


class Pub:
    registry = {}

    def __init__(self, name, *a, **k):
        self.name, self.last, self.t_last = name, None, -1e9
        Pub.registry[name] = self

    def publish(self, m):
        self.last = m
        self.t_last = SIM_T[0]


class Twist:
    def __init__(self):
        self.linear = types.SimpleNamespace(x=0.0)
        self.angular = types.SimpleNamespace(z=0.0)


PARAMS = {"/%s/CMD_VEL_IN_topic" % ROBOT: "cmd_vel",
          "/%s/left/stiffness_vec" % ROBOT: [0.2] * 5,
          "/%s/right/stiffness_vec" % ROBOT: [0.2] * 5}
_STUB_ROSPY = types.SimpleNamespace(
    Publisher=Pub, Subscriber=lambda *a, **k: None, get_param=lambda n, d=None: PARAMS.get(n, d),
    loginfo=_log, logwarn=_log, logerr=_log, logfatal=_log,
    logwarn_throttle=lambda p, *a: None, loginfo_throttle=lambda p, *a: None)
_STUBS = {"rospy": _STUB_ROSPY}
for _mod, _names in [("std_msgs.msg", ["Float64", "Float64MultiArray", "Empty", "String"]),
                     ("geometry_msgs.msg", ["PointStamped"]),
                     ("alterego_msgs.msg", ["UpperBodyState", "LowerBodyState"])]:
    _m = types.ModuleType(_mod)
    for _n in _names:
        setattr(_m, _n, lambda **kw: types.SimpleNamespace(**kw))
    _STUBS[_mod] = _m
_STUBS["geometry_msgs.msg"].Twist = Twist


@contextmanager
def ros_stubs():
    """Fake rospy and messages, only while the simulated follower is imported and built. The follower keeps
    references to what it got, so the real rospy can live in the same process (GUI in real-robot mode)."""
    saved = {k: sys.modules.get(k) for k in _STUBS}
    saved_env = os.environ.get("ROBOT_NAME")
    sys.modules.update(_STUBS)
    os.environ["ROBOT_NAME"] = ROBOT
    try:
        yield
    finally:
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v
        if saved_env is None:
            os.environ.pop("ROBOT_NAME", None)
        else:
            os.environ["ROBOT_NAME"] = saved_env


with ros_stubs():
    import person_follower as pf  # noqa: E402
import person_tracker as pt   # noqa: E402  (pure functions only)
from urdf_fk import URDFKinematics  # noqa: E402

pf.time = types.SimpleNamespace(time=lambda: SIM_T[0], sleep=lambda s: None)

# ================================================================================================ robot model
SRC_URDF = os.path.join(HERE, "alterego_v2.urdf")
HEAD_LINKS = {"neck", "neck_cube", "head", "camera", "camera_left", "camera_right", "realsense", "fake_realsense"}
NECK_PIVOT = np.array([0.0, 0.0, 0.7535])          # base_to_neck origin, in the base frame
RIGHT = ["base_to_right_shoulder_flange", "right_shoulder_cube_to_right_arm_flange",
         "right_arm_cube_to_right_elbow_flange", "right_elbow_cube_to_right_forearm_flange",
         "right_forearm_cube_to_right_wrist_flange"]
LEFT = [j.replace("right", "left") for j in RIGHT]


def rot_head(yaw, pitch):
    """Neck as head_inv_kin: yaw (left cube) about the vertical, then pitch (right cube), about the pivot.
    (In alterego_v2.urdf both neck joints are pitch axes, which does not match head_inv_kin.)"""
    cy, sy, cp, sp = math.cos(yaw), math.sin(yaw), math.cos(pitch), math.sin(pitch)
    R = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]]) @ np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    T = np.eye(4); T[:3, :3] = R
    P = np.eye(4); P[:3, 3] = NECK_PIVOT
    Pi = np.eye(4); Pi[:3, 3] = -NECK_PIVOT
    return P @ T @ Pi


class RobotKinematics:
    def __init__(self, urdf=SRC_URDF):
        self.fk = URDFKinematics(urdf)
        z = self.fk.link_poses()
        self.T_world_base0 = z["base"]
        self.T_cam0 = np.linalg.inv(self.T_world_base0) @ z["fake_realsense"]
        self.Tb0_inv = np.linalg.inv(self.T_world_base0)

    def base_pose(self, x, y=0.0, yaw=0.0):
        T = np.eye(4)
        T[:3, :3] = np.array([[math.cos(yaw), -math.sin(yaw), 0], [math.sin(yaw), math.cos(yaw), 0], [0, 0, 1]])
        T[:3, 3] = [x, y, 0.0]
        return T @ self.T_world_base0

    def link_poses(self, q_r, q_l, yaw, pitch, base_xy, base_yaw=0.0):
        q = {RIGHT[i]: float(q_r[i]) for i in range(5)}
        q.update({LEFT[i]: float(q_l[i]) for i in range(5)})
        P = self.fk.link_poses(q)
        Tb = self.base_pose(base_xy[0], base_xy[1], base_yaw)
        H = rot_head(yaw, pitch)
        out = {}
        for link, T in P.items():
            Tbl = self.Tb0_inv @ T
            if link in HEAD_LINKS:
                Tbl = H @ Tbl
            out[link] = Tb @ Tbl
        return out

    def camera_pose(self, base_x, yaw, pitch, base_y=0.0, base_yaw=0.0):
        return self.base_pose(base_x, base_y, base_yaw) @ rot_head(yaw, pitch) @ self.T_cam0


# ================================================================================================ person
W, H, FX, FY, CX, CY = 640, 480, 615.0, 615.0, 320.0, 240.0
L_UP, L_FORE = 0.29, 0.26
DOWN = (0, 0, -1)


def smooth(t, t0, t1):
    if t <= t0:
        return 0.0
    if t >= t1:
        return 1.0
    x = (t - t0) / (t1 - t0)
    return x * x * (3 - 2 * x)


def lerp(a, b, s):
    return (1 - s) * np.asarray(a, float) + s * np.asarray(b, float)


def nrm(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


def keyframe(t, keys):
    if t <= keys[0][0]:
        return nrm(keys[0][1])
    for (t0, d0), (t1, d1) in zip(keys, keys[1:]):
        if t0 <= t < t1:
            return nrm(lerp(nrm(d0), nrm(d1), smooth(t, t0, t1)) + 1e-9)
    return nrm(keys[-1][1])


# Arm poses of the demo person: (elevation, azimuth out, elbow, forearm twist) in degrees, see arm_dirs()
DOWN_P = (-90, 0, 0, 0)
UP_P = (80, 20, 10, 0)            # both hands above the head: start / stop
T_P = (0, 88, 5, 0)               # arms open to the side: EXPAND
CHEST_P = (-55, 5, 120, 80)       # hands on the chest: ATTRACT
RAISED_P = (80, 15, 5, 0)         # one hand above the head: PIROUETTE
R_KEYS = [(0, DOWN_P), (4.6, DOWN_P), (5.2, UP_P), (7.0, UP_P), (7.8, DOWN_P),
          (20.5, DOWN_P), (21.7, (0, 0, 0, 0)), (22.6, (0, 0, 0, 0)), (23.6, (0, 88, 0, 0)), (24.4, (0, 88, 0, 0)),
          (25.2, (15, 60, 0, 0)), (26.0, (15, 60, 0, 0)), (26.6, (0, 88, 0, 0)), (27.4, DOWN_P),
          (38.3, DOWN_P), (38.9, T_P), (41.8, T_P), (42.4, DOWN_P),
          (44.3, DOWN_P), (44.9, CHEST_P), (48.8, CHEST_P), (49.4, DOWN_P),
          (50.8, DOWN_P), (51.3, RAISED_P), (52.7, RAISED_P), (53.3, DOWN_P),
          (62.2, DOWN_P), (62.8, UP_P), (64.4, UP_P), (65.0, DOWN_P)]
L_KEYS = [(0, DOWN_P), (4.6, DOWN_P), (5.2, UP_P), (7.0, UP_P), (7.8, DOWN_P),
          (22.0, DOWN_P), (23.0, (-90, 0, 90, 0)), (25.8, (-90, 0, 90, 0)), (26.8, DOWN_P),
          (38.3, DOWN_P), (38.9, T_P), (41.8, T_P), (42.4, DOWN_P),
          (44.3, DOWN_P), (44.9, CHEST_P), (48.8, CHEST_P), (49.4, DOWN_P),
          (62.2, DOWN_P), (62.8, UP_P), (64.4, UP_P), (65.0, DOWN_P)]


def pose_keyframe(t, keys):
    """Arm pose parameters interpolated between keyframes, smooth."""
    if t <= keys[0][0]:
        return keys[0][1]
    for (t0, p0), (t1, p1) in zip(keys, keys[1:]):
        if t0 <= t < t1:
            return tuple(lerp(p0, p1, smooth(t, t0, t1)))
    return keys[-1][1]


def demo_script(t):
    """Scripted person of the demo: position, crouch, arm directions (upper, forearm) in the body frame."""
    x = 1.9
    x = lerp(x, 1.25, smooth(t, 8.0, 11.0))
    x = lerp(x, 2.55, smooth(t, 14.0, 19.0))
    y = 0.0
    y = lerp(y, 0.75, smooth(t, 27.0, 29.5))
    y = lerp(y, -0.55, smooth(t, 29.8, 32.6))
    y = lerp(y, 0.0, smooth(t, 32.8, 34.2))
    crouch = 0.38 * (smooth(t, 34.6, 35.6) - smooth(t, 36.6, 37.6))
    r = pose_keyframe(t, R_KEYS)
    l = pose_keyframe(t, L_KEYS)
    arms = (*arm_dirs(*r[:3], "right", r[3]), *arm_dirs(*l[:3], "left", l[3]))
    return np.array([float(x), float(y)]), float(crouch), arms


CAPTIONS = [
    (0.0, 4.6, "Idle: the head looks for your face (the camera is on the head)"),
    (4.6, 7.8, "Gesture: both hands above the head for 1 s  ->  START"),
    (7.8, 13.5, "You come closer  ->  it backs off to keep the distance"),
    (13.5, 20.3, "You step back  ->  it comes forward"),
    (20.3, 27.4, "Arms, mirror mode: your right arm -> its left arm"),
    (27.4, 34.4, "You move sideways  ->  the head keeps your face at the image center"),
    (34.4, 38.0, "You crouch  ->  the head looks down"),
    (38.0, 43.6, "EXPAND: arms open and held  ->  the target distance grows, it moves away"),
    (43.6, 50.4, "ATTRACT: hands on the chest, held  ->  it reaches its arms out and comes closer"),
    (50.4, 62.0, "Right hand up for 1 s  ->  PIROUETTE: arms open, head leads the turn, then back to mimicking"),
    (62.0, 67.0, "Both hands above the head  ->  STOP: arms down, base still"),
]
DEMO_LENGTH = 67.0


def demo_caption(t):
    for t0, t1, text in CAPTIONS:
        if t0 <= t < t1:
            return text
    return ""


def arm_dirs(elev_deg, azim_deg, elbow_deg, side, twist_deg=0.0):
    """Upper arm and forearm directions in the body frame (fwd, left, up) from GUI angles.
    elev: -90 down, 0 horizontal, 90 up. azim: 0 forward, 90 out to the side. elbow: 0 straight.
    twist: rotation of the elbow bend plane about the upper arm; > 0 turns the forearm toward the body midline
    (0 = the forearm bends forward, 90 = it bends inward, e.g. hands on the chest)."""
    e, a, b = math.radians(elev_deg), math.radians(azim_deg), math.radians(elbow_deg)
    out = -1.0 if side == "right" else 1.0
    up = np.array([math.cos(e) * math.cos(a), out * math.cos(e) * math.sin(a), math.sin(e)])
    ref = np.array([1.0, 0, 0]) if abs(up[0]) < 0.9 else np.array([0, 0, 1.0])
    perp = ref - (ref @ up) * up
    perp = perp / np.linalg.norm(perp)
    if twist_deg:
        # rotate the bend direction about the upper arm; the sign makes positive twist go toward the midline
        side_ax = np.cross(up, perp)
        if side_ax[1] * out > 0:
            side_ax = -side_ax
        tw = math.radians(twist_deg)
        perp = nrm(math.cos(tw) * perp + math.sin(tw) * side_ax)
    return nrm(up), nrm(math.cos(b) * up + math.sin(b) * perp)


def build_keypoints(pos, body_yaw, crouch, arms, vel, t):
    """17 COCO keypoints in the world frame. The person faces -x (the robot side), turned by body_yaw."""
    r_up, r_fo, l_up, l_fo = arms
    cy, sy = math.cos(body_yaw), math.sin(body_yaw)
    fwd = np.array([-cy, -sy, 0.0])
    left = np.array([sy, -cy, 0.0])
    up = np.array([0, 0, 1.0])
    R = np.stack([fwd, left, up], 1)
    base = np.array([pos[0], pos[1], 0.0])
    speed = float(np.linalg.norm(vel))
    k = np.zeros((17, 3))
    zc = lambda z: z - crouch if z > 0.5 else z - crouch * 0.45 * (z / 0.5)
    head_c = base + np.array([0, 0, zc(1.64)])
    k[0] = head_c + fwd * 0.10 - up * 0.02
    k[1] = head_c + fwd * 0.08 + left * 0.035 + up * 0.02
    k[2] = head_c + fwd * 0.08 - left * 0.035 + up * 0.02
    k[3] = head_c + left * 0.075
    k[4] = head_c - left * 0.075
    sh_z, hip_z = zc(1.45), zc(0.95)
    k[5] = base + left * 0.20 + up * sh_z; k[6] = base - left * 0.20 + up * sh_z
    k[11] = base + left * 0.12 + up * hip_z; k[12] = base - left * 0.12 + up * hip_z
    k[7] = k[5] + L_UP * (R @ l_up); k[9] = k[7] + L_FORE * (R @ l_fo)
    k[8] = k[6] + L_UP * (R @ r_up); k[10] = k[8] + L_FORE * (R @ r_fo)
    ph = t * 2 * math.pi * 1.6
    sw = min(speed / 0.25, 1.0) * 0.18
    move_dir = np.asarray(vel) / speed if speed > 1e-3 else np.zeros(2)
    for side, (hip, knee, ank) in ((1, (11, 13, 15)), (-1, (12, 14, 16))):
        s = side * math.sin(ph) * sw
        off = np.array([move_dir[0] * s, move_dir[1] * s, 0])
        k[knee] = k[hip] + np.array([0, 0, -(hip_z - zc(0.5))]) + off * 0.5 + fwd * crouch * 0.6
        k[ank] = np.array([k[hip][0], k[hip][1], 0.08]) + off
    return k


class InteractivePerson:
    """Person driven by the GUI: targets move smoothly (walking speed, arm speed)."""

    def __init__(self):
        self.target = dict(x=1.9, y=0.0, yaw=0.0, crouch=0.0,
                           r_elev=-90.0, r_azim=0.0, r_elbow=0.0, r_twist=0.0,
                           l_elev=-90.0, l_azim=0.0, l_elbow=0.0, l_twist=0.0)
        self.state = dict(self.target)
        self.vel = np.zeros(2)

    def step(self, dt):
        rates = dict(x=0.8, y=0.8, yaw=math.radians(60), crouch=0.5)
        prev = np.array([self.state["x"], self.state["y"]])
        for k, v in self.target.items():
            r = rates.get(k, 120.0) * dt
            self.state[k] += float(np.clip(v - self.state[k], -r, r))
        self.vel = (np.array([self.state["x"], self.state["y"]]) - prev) / dt

    def pose(self):
        s = self.state
        arms = (*arm_dirs(s["r_elev"], s["r_azim"], s["r_elbow"], "right", s["r_twist"]),
                *arm_dirs(s["l_elev"], s["l_azim"], s["l_elbow"], "left", s["l_twist"]))
        return np.array([s["x"], s["y"]]), s["yaw"], s["crouch"], arms


# ================================================================================================ camera
def observe(kw, T_cam, rng):
    """What person_tracker would publish for the world keypoints kw seen from the camera pose T_cam."""
    Tinv = np.linalg.inv(T_cam)
    P = kw @ Tinv[:3, :3].T + Tinv[:3, 3]
    sigma = 0.004 * P[:, 2] ** 2 + 0.005
    Pn = P + rng.normal(0, 1, P.shape) * sigma[:, None]
    z = np.where(np.abs(Pn[:, 2]) < 1e-6, 1e-6, Pn[:, 2])
    u = FX * Pn[:, 0] / z + CX
    v = FY * Pn[:, 1] / z + CY
    vis = (Pn[:, 2] > 0.3) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
    conf = np.where(vis, 0.9, 0.0)
    found = vis[[5, 6, 11, 12]].sum() >= 2 or vis[:5].sum() >= 3
    obs = dict(u=u, v=v, vis=vis, found=bool(found), face_uv=None, face_est=False)
    if not found:
        obs.update(pose=pt.pose_message_data(False, W, H), kp3d=[0.0] * 69, face=[0, 0, 0, 0], pos=None)
        return obs
    obs["pose"] = pt.pose_message_data(True, W, H, np.stack([u / W, v / H], 1), conf)
    kp3d = [1.0]
    for i in range(17):
        kp3d += [*Pn[i], 1.0] if vis[i] else [0, 0, 0, 0]
    obs["kp3d"] = kp3d
    torso = [Pn[i] for i in (5, 6, 11, 12) if vis[i]]
    obs["pos"] = np.mean(torso, 0) if torso else None
    face_uv, est = pt.face_pixel(np.stack([u, v], 1), conf, 0.3)
    obs["face_uv"], obs["face_est"] = face_uv, est
    obs["face"] = ([1.0, math.atan2(-(face_uv[0] - CX) / FX, 1), math.atan2(-(face_uv[1] - CY) / FY, 1), 0.0]
                   if face_uv is not None else [0.0, 0.0, 0.0, 0.0])
    return obs


# ================================================================================================ simulation
class Sim:
    """One closed-loop simulation. mode 'demo' (scripted person) or 'interactive' (InteractivePerson)."""

    def __init__(self, mode="demo", rate=50, cam_hz=15.0, delay=0.15, seed=3, robot=None):
        EVENTS.clear()
        SIM_T[0] = 0.0
        self.mode, self.dt, self.cam_dt, self.delay = mode, 1.0 / rate, 1.0 / cam_hz, delay
        self.rng = np.random.default_rng(seed)
        self.robot = robot or RobotKinematics()
        with ros_stubs():
            self.node = pf.PersonFollower()
        self.dt = self.node.dt              # the simulation steps at the follower's own rate (follower.yaml)
        self.person = InteractivePerson()
        self.t = 0.0
        self.base_x = self.base_y = self.base_yaw = 0.0     # base pose on the floor, yaw > 0 = turned left
        self.queue = []
        self.next_cam = 0.0
        self.last_view = None
        self.FA = lambda d: types.SimpleNamespace(data=list(d))

    def person_keypoints(self, t):
        if self.mode == "demo":
            pos, crouch, arms = demo_script(t)
            vel = (demo_script(t + 0.05)[0] - demo_script(t - 0.05)[0]) / 0.1
            return build_keypoints(pos, 0.0, crouch, arms, vel, t)
        pos, yaw, crouch, arms = self.person.pose()
        # the person faces the robot wherever it is (walking around it), plus the "body turn" of the GUI
        facing = math.atan2(pos[1] - self.base_y, pos[0] - self.base_x)
        return build_keypoints(pos, facing + yaw, crouch, arms, self.person.vel, t)

    def camera_pose(self):
        n = self.node
        return self.robot.camera_pose(self.base_x, n.yaw_cmd, n.pitch_cmd, self.base_y, self.base_yaw)

    def step(self):
        n, t = self.node, self.t
        SIM_T[0] = t
        if self.mode != "demo":
            self.person.step(self.dt)
        n.cb_upper(types.SimpleNamespace(left_meas_arm_shaft=list(n.q_l), right_meas_arm_shaft=list(n.q_r),
                                         left_meas_neck_shaft=n.yaw_cmd, right_meas_neck_shaft=n.pitch_cmd))
        # like the robot IMU: yaw_angle decreases when the base turns left (LQR: des_yaw_rate = -angular.z)
        n.cb_lower(types.SimpleNamespace(pitch_angle=0.0, yaw_angle=-self.base_yaw))
        if t >= self.next_cam:
            self.next_cam += self.cam_dt
            T_cam = self.camera_pose()
            obs = observe(self.person_keypoints(t), T_cam, self.rng)
            self.last_view = obs
            self.queue.append((t + self.delay, obs))
        while self.queue and self.queue[0][0] <= t:
            o = self.queue.pop(0)[1]
            n.cb_pose(self.FA(o["pose"]))
            n.cb_kp3d(self.FA(o["kp3d"]))
            n.cb_face(self.FA(o["face"]))
            if o["pos"] is not None:
                n.cb_pos(types.SimpleNamespace(point=types.SimpleNamespace(x=o["pos"][0], y=o["pos"][1], z=o["pos"][2])))
        n.cycle()
        # like the LQR on the robot: cmd_vel older than 0.5 s -> velocity zero. angular.z > 0 = turn left
        pub = Pub.registry.get("/%s/cmd_vel" % ROBOT)
        live = pub is not None and pub.last is not None and t - pub.t_last <= 0.5
        self.v = pub.last.linear.x if live else 0.0
        self.w = pub.last.angular.z if live else 0.0
        self.base_yaw += self.w * self.dt
        self.base_x += self.v * math.cos(self.base_yaw) * self.dt
        self.base_y += self.v * math.sin(self.base_yaw) * self.dt
        self.t += self.dt

    def state(self):
        n = self.node
        return dict(t=self.t, base_x=self.base_x, base_y=self.base_y, base_yaw=self.base_yaw,
                    q_r=n.q_r.copy(), q_l=n.q_l.copy(), yaw=n.yaw_cmd, pitch=n.pitch_cmd,
                    kw=self.person_keypoints(self.t), T_cam=self.camera_pose(), w=getattr(self, "w", 0.0),
                    active=n.active, v=getattr(self, "v", 0.0), d_ref=n.d_ref, d=n.distance())
