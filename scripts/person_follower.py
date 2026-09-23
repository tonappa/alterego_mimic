#!/usr/bin/env python3
# Person follower for AlterEgo: keeps the distance from the person, keeps the face in view with the head,
# turns the base when the neck reaches its limit, and mimics the arms.
# Input: the topics of person_tracker.py (same package). Output: cmd_vel, arm cubes, neck cubes.
# Plain ROS Python (no conda): it can run on the robot base PC or on the PC connected to the robot master.
#
# Start / stop (toggle), any of:
#   - Enter in the terminal (only with rosrun, roslaunch gives the node no keyboard)
#   - rostopic pub -1 /$ROBOT_NAME/person_follower/toggle std_msgs/Empty
#   - gesture: both wrists above the nose for gesture_hold s
# At start the node memorizes the distance of the person. The head tracks the face also before the start
# (head_track_idle), so the robot sees the gesture.
#
# Robot side: imu, body_activation, wheels, pilot running; NOT body_movement (its arm and head controllers
# write the same cubes). pitch_correction must run: launch/pitch_correction.launch of this package.
#
# Ctrl-C: base stopped, arms and head back to zero with a slow ramp, then exit.
#
# GUI / remote control (gui/web_gui.py uses them in "Real robot" mode):
#   /$ROBOT_NAME/person_follower/status   std_msgs/String, JSON at 10 Hz: active, distance, velocity, neck and
#                                         arm commands, options, tunable parameters
#   /$ROBOT_NAME/person_follower/command  std_msgs/String, JSON: {"action": "start" | "stop" | "toggle"},
#                                         {"option": <name>, "value": true|false}, {"param": <name>, "value": x}
import json
import math
import os
import signal
import sys
import threading
import time

import numpy as np

import gestures

OPTIONS = ("mirror", "keep_distance", "mimic_arms", "track_head", "turn_base",
           "gesture_expand_attract", "gesture_pirouette", "gesture_freeze", "gesture_drop")
# Parameters that the GUI / command topic may change at run time, with the accepted range
TUNABLE = {"k_lin": (0.0, 3.0), "dead_lin": (0.0, 0.5), "max_lin": (0.0, 0.5), "min_distance": (0.3, 3.0),
           "k_head": (0.0, 6.0), "head_dead": (0.0, 0.3), "joint_speed_deg_s": (5.0, 180.0),
           "gesture_hold": (0.2, 5.0), "max_arm_elevation_deg": (-60.0, 30.0),
           "turn_start": (0.1, 0.6), "turn_gain": (0.0, 3.0), "turn_max": (0.0, 0.6),
           "expand_rate": (0.0, 0.5), "attract_rate": (0.0, 0.5), "max_follow_distance": (1.0, 4.0),
           "attract_min_distance": (0.6, 2.0), "pirouette_speed": (0.2, 1.5), "pirouette_min_distance": (0.8, 3.0)}
N_KP = 17
NOSE, L_EYE, R_EYE, L_EAR, R_EAR = 0, 1, 2, 3, 4
L_SH, R_SH, L_EL, R_EL, L_WR, R_WR, L_HIP, R_HIP = 5, 6, 7, 8, 9, 10, 11, 12

# AlterEgo v2 right arm, from alterego_robot/config/v2/pitch_correction.yaml (used if the params are missing)
V2_MODEL = dict(
    T_b2t=[1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0.65, 0, 0, 0, 1],
    T_t2s_R=[0, 0, -1, 0, 1, 0, 0, -0.11, 0, -1, 0, 0, 0, 0, 0, 1],
    DH_Xtr_R=[0, 0, 0, 0, 0],
    DH_Xrot_R=[-1.57, 1.57, -1.57, 1.57, 0],
    DH_Ztr_R=[0.084, 0, 0.168, 0, 0.18],
    DH_Zrot_R=[0, 1.57, 1.57, 0, 3.14],
)


# ======================================================================================================
# Pure functions (no ROS): tested in test/test_follower_offline.py
# ======================================================================================================
def unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else None


def camera_up(yaw, pitch):
    """World up expressed in the camera optical frame when the camera turns with the head.
    Optical frame at zero head: x = -y_base, y = -z_base, z = x_base. Head rotation = Rz(yaw) Ry(pitch)
    (robot conventions: yaw > 0 = left, pitch < 0 = up)."""
    cy, sy, cp, sp = math.cos(yaw), math.sin(yaw), math.cos(pitch), math.sin(pitch)
    head = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]]) @ np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    r0 = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]], dtype=float)     # columns: optical x, y, z in base
    return (head @ r0).T @ np.array([0.0, 0.0, 1.0])


def person_frame(kp3d, up_hint=(0.0, -1.0, 0.0)):
    """Body frame of the person from 3D keypoints in the camera optical frame (x right, y down, z forward).
    Returns the rows (forward, left, up) as a 3x3 matrix, same convention as the robot base (x fwd, y left,
    z up), or None. kp3d: (17, 4) with X, Y, Z, valid. up_hint: world up in the camera frame, used when the
    hips are out of view (camera_up() with the head angles)."""
    if not (kp3d[L_SH, 3] and kp3d[R_SH, 3]):
        return None
    left = unit(kp3d[L_SH, :3] - kp3d[R_SH, :3])
    if left is None:
        return None
    if kp3d[L_HIP, 3] and kp3d[R_HIP, 3]:
        up = (kp3d[L_SH, :3] + kp3d[R_SH, :3]) / 2 - (kp3d[L_HIP, :3] + kp3d[R_HIP, :3]) / 2
    else:
        up = np.asarray(up_hint, dtype=float)    # hips out of view: world up seen by the camera
    up = unit(up - np.dot(up, left) * left)
    if up is None:
        return None
    fwd = np.cross(left, up)
    return np.vstack([fwd, left, up])


def arm_in_body(kp3d, frame, side):
    """Upper-arm unit direction in the person body frame and elbow flexion [rad] (0 = straight).
    side 'left' / 'right' = the person's arm. Direction None if shoulder or elbow is missing,
    flexion None if the wrist is missing."""
    sh, el, wr = (L_SH, L_EL, L_WR) if side == "left" else (R_SH, R_EL, R_WR)
    if not (kp3d[sh, 3] and kp3d[el, 3]):
        return None, None
    upper = kp3d[el, :3] - kp3d[sh, :3]
    d = unit(frame @ upper)
    if d is None:
        return None, None
    flex = None
    if kp3d[wr, 3]:
        fore = kp3d[wr, :3] - kp3d[el, :3]
        u, f = unit(upper), unit(fore)
        if u is not None and f is not None:
            flex = math.acos(float(np.clip(np.dot(u, f), -1.0, 1.0)))
    return d, flex


def dead_band(x, band):
    return x - band if x > band else (x + band if x < -band else 0.0)


def mirror_y(d):
    return np.array([d[0], -d[1], d[2]])


class ArmModel:
    """Forward kinematics of the RIGHT arm of the robot, same model as pitch_correction (KDL chain built
    from T_b2t, T_t2s_R and DH). The left arm is the mirror: q_left = -q_right for the mirrored pose.
    Solves shoulder (q0, q1) for an upper-arm direction with a lookup table, then the elbow (q3)."""

    def __init__(self, m, q0_lim, q1_lim, q3_lim, max_elev_rad, hand_margin, step_deg=2.0):
        self.m = m
        self.q3_lim = q3_lim
        self.max_elev = max_elev_rad
        self.last_h = {}
        self.last_q = {}
        self.hand_margin = hand_margin
        q0s = np.radians(np.arange(q0_lim[0], q0_lim[1] + 1e-9, step_deg))
        q1s = np.radians(np.arange(q1_lim[0], q1_lim[1] + 1e-9, step_deg))
        grid, dirs = [], []
        for a in q0s:
            for b in q1s:
                p = self.points([a, b, 0, 0, 0])
                d = unit(p[2] - p[0])
                if d is not None and d[2] <= math.sin(max_elev_rad) + 1e-9:   # upper arm not above the limit
                    grid.append((a, b))
                    dirs.append(d)
        self.grid = np.array(grid)
        self.dirs = np.array(dirs)
        self.shoulder = self.points([0, 0, 0, 0, 0])[0]

    @staticmethod
    def _fixed(T):
        # Same element order as pitch_correction.cpp: Rotation(T0,T4,T8, T1,T5,T9, T2,T6,T10)
        F = np.eye(4)
        F[:3, :3] = [[T[0], T[4], T[8]], [T[1], T[5], T[9]], [T[2], T[6], T[10]]]
        F[:3, 3] = [T[3], T[7], T[11]]
        return F

    @staticmethod
    def _dh(a, al, d, th):
        ct, st, ca, sa = math.cos(th), math.sin(th), math.cos(al), math.sin(al)
        F = np.eye(4)
        F[:3, :3] = [[ct, -st * ca, st * sa], [st, ct * ca, -ct * sa], [0, sa, ca]]
        F[:3, 3] = [a * ct, a * st, d]
        return F

    @staticmethod
    def _rz(q):
        F = np.eye(4)
        F[:2, :2] = [[math.cos(q), -math.sin(q)], [math.sin(q), math.cos(q)]]
        return F

    def points(self, q):
        """Origins of the 5 joint frames of the right arm in the base frame (x fwd, y left, z up)."""
        m = self.m
        T = self._fixed(m["T_b2t"]) @ self._fixed(m["T_t2s_R"])
        out = []
        for i in range(5):
            T = T @ self._rz(q[i]) @ self._dh(m["DH_Xtr_R"][i], m["DH_Xrot_R"][i], m["DH_Ztr_R"][i], m["DH_Zrot_R"][i])
            out.append(T[:3, 3].copy())
        return out

    def solve(self, direction, flex, key="r"):
        """Right-arm joints [q0..q4] for an upper-arm direction (robot base frame) and elbow flexion [rad].
        A direction above the elevation limit is brought down to the limit keeping its azimuth (straight up:
        the azimuth of the previous call with the same key). The hand is kept at most hand_margin above the
        shoulder by reducing the elbow."""
        direction = np.asarray(direction, dtype=float)
        s_max = math.sin(self.max_elev)
        if direction[2] > s_max:
            h = np.array([direction[0], direction[1], 0.0])
            if np.linalg.norm(h) < 0.2:
                h = self.last_h.get(key, np.array([1.0, 0.0, 0.0]))
            h = h / np.linalg.norm(h)
            direction = h * math.cos(self.max_elev) + np.array([0.0, 0.0, s_max])
        self.last_h[key] = np.array([direction[0], direction[1], 0.0]) if np.hypot(direction[0], direction[1]) >= 0.2 \
            else self.last_h.get(key, np.array([1.0, 0.0, 0.0]))
        # Near the gimbal (arm out to the side, q1 = +-90) many (q0, q1) give the same direction: among the
        # solutions within ~3 deg of the best one, take the closest to the previous one, so q0 never jumps.
        dots = self.dirs @ direction
        near = np.where(dots >= dots.max() - 0.0015)[0]
        if key in self.last_q and len(near) > 1:
            i = int(near[np.argmin(np.sum((self.grid[near] - self.last_q[key]) ** 2, axis=1))])
        else:
            i = int(near[np.argmax(dots[near])])
        q0, q1 = self.grid[i]
        self.last_q[key] = self.grid[i].copy()
        q3 = float(np.clip(flex if flex is not None else 0.0, math.radians(self.q3_lim[0]), math.radians(self.q3_lim[1])))
        while q3 > 0 and self.points([q0, q1, 0, q3, 0])[4][2] > self.shoulder[2] + self.hand_margin:
            q3 = max(0.0, q3 - math.radians(5))
        return np.array([q0, q1, 0.0, q3, 0.0])


# ======================================================================================================
# Node
# ======================================================================================================
class PersonFollower:
    def __init__(self):
        import rospy
        import yaml
        from std_msgs.msg import Float64, Float64MultiArray, Empty, String
        from geometry_msgs.msg import Twist, PointStamped
        from alterego_msgs.msg import UpperBodyState, LowerBodyState
        self.rospy, self.Float64, self.Float64MultiArray, self.Twist = rospy, Float64, Float64MultiArray, Twist

        self.robot = os.environ.get("ROBOT_NAME", "")
        if not self.robot:
            rospy.logfatal("person_follower: ROBOT_NAME is not set")
            raise SystemExit(1)
        ns = "/" + self.robot + "/"

        # Defaults from config/follower.yaml, overridden by private params (launch or _name:=value)
        cfg_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "config", "follower.yaml")
        with open(cfg_path) as f:
            self.cfg = yaml.safe_load(f)
        # Parameters: config/follower.yaml, overridden by the private params, read ONCE here (a get_param per use
        # would be an XMLRPC call to the master, hundreds per second). Changed at run time only by apply_command().
        self.params = {k: rospy.get_param("~" + k, v) for k, v in self.cfg.items()}
        P = lambda k: self.params[k]
        self.P = P
        self.String = String
        self.pending = None                 # "start" / "stop" requested by a command, handled in cycle()
        self.arms_releasing = False         # mimic_arms switched off: arms go back to zero, then stop publishing
        self.vel_releasing = False          # keep_distance switched off: one zero cmd_vel, then the pilot is free
        self.head_releasing = False
        self.t_status = 0.0

        self.keep_distance, self.track_head, self.mimic_arms = P("keep_distance"), P("track_head"), P("mimic_arms")
        self.turn_base = P("turn_base")
        self.mirror = P("mirror")
        for k in OPTIONS[5:]:                           # dance vocabulary switches
            setattr(self, k, bool(P(k)))
        self.rate_hz = float(P("rate"))
        self.dt = 1.0 / self.rate_hz

        # ---- arm model and joint limits
        model = {k: rospy.get_param(ns + k, V2_MODEL[k]) for k in V2_MODEL}
        self.n_cubes = int(rospy.get_param(ns + "arm_cubes_n", 5))
        self.arm = ArmModel(model, P("q0_limits_deg"), P("q1_limits_deg"), P("elbow_limits_deg"),
                            math.radians(P("max_arm_elevation_deg")), P("hand_above_shoulder_max"))
        stiff = P("stiffness")
        self.stiff_l = list(stiff) if stiff else rospy.get_param(ns + "left/stiffness_vec", [])
        self.stiff_r = list(stiff) if stiff else rospy.get_param(ns + "right/stiffness_vec", [])

        # ---- state
        self.lock = threading.Lock()
        self.kp2d = np.zeros((N_KP, 3)); self.found = False; self.t_pose = 0.0
        self.kp3d = np.zeros((N_KP, 4)); self.t_kp3d = 0.0
        self.pos = None; self.t_pos = 0.0
        self.face = None; self.t_face = 0.0; self.t_face_seen = 0.0
        self.pitch = 0.0
        self.meas_l = self.meas_r = None; self.meas_neck = (0.0, 0.0)
        self.active = False
        self.toggle_requested = False
        self.stop_requested = False
        self.last_toggle = 0.0
        self.hands_up_since = None
        self.hands_were_down = True
        # reference taken at start
        self.d_ref = None
        # commands
        self.q_r = np.zeros(self.n_cubes); self.q_l = np.zeros(self.n_cubes)
        self.q_r_tgt = np.zeros(self.n_cubes); self.q_l_tgt = np.zeros(self.n_cubes)
        self.yaw_cmd = self.pitch_cmd = 0.0
        self.yaw_tgt = self.pitch_tgt = 0.0
        self.v = self.v_f = 0.0
        self.w = self.w_f = 0.0             # base yaw rate [rad/s], > 0 = turn left (ROS convention)
        self.turning = False                # hysteresis of the base rotation
        self.last_flex = {"r": None, "l": None}   # last elbow flexion seen, per robot arm
        self.t_last_visible = 0.0           # last time the person was visible while following
        # dance vocabulary
        self.gest = gestures.GestureDetector(self.P)
        self.t_gest = 0.0                   # camera frame last given to the detector
        self.primitive = None               # running movement primitive (pirouette, drop), a dict
        self.grace_until = 0.0              # the lost-person rule is suspended until then (after a pirouette)
        self.yaw_imu = None; self.t_yaw = 0.0

        # ---- I/O
        rospy.Subscriber(ns + "person_pose", Float64MultiArray, self.cb_pose, queue_size=1)
        rospy.Subscriber(ns + "person_keypoints_3d", Float64MultiArray, self.cb_kp3d, queue_size=1)
        rospy.Subscriber(ns + "person_position", PointStamped, self.cb_pos, queue_size=1)
        rospy.Subscriber(ns + "person_face", Float64MultiArray, self.cb_face, queue_size=1)
        rospy.Subscriber(ns + "alterego_state/upperbody", UpperBodyState, self.cb_upper, queue_size=1)
        rospy.Subscriber(ns + "alterego_state/lowerbody", LowerBodyState, self.cb_lower, queue_size=1)
        rospy.Subscriber(ns + "person_follower/toggle", Empty, lambda _m: self.request_toggle("topic"), queue_size=1)
        rospy.Subscriber(ns + "person_follower/command", String, self.cb_command, queue_size=10)
        self.pub_status = rospy.Publisher(ns + "person_follower/status", String, queue_size=1)

        cmd_vel = rospy.get_param(ns + "CMD_VEL_IN_topic", "")
        self.ns = ns
        self.cmd_vel_topic = ns + cmd_vel if cmd_vel else ""
        if (self.keep_distance or self.turn_base) and not cmd_vel:
            rospy.logfatal("person_follower: param CMD_VEL_IN_topic not set (pilot.launch not running?)")
            raise SystemExit(1)
        self.pub_vel = rospy.Publisher(ns + cmd_vel, Twist, queue_size=1) if (self.keep_distance or self.turn_base) else None
        self.pub_eq = {s: rospy.Publisher(ns + s + "/ref_cubes_eq", Float64MultiArray, queue_size=1) for s in ("left", "right")}
        self.pub_pr = {s: rospy.Publisher(ns + s + "/ref_cubes_preset", Float64MultiArray, queue_size=1) for s in ("left", "right")}
        yaw_topic = rospy.get_param(ns + "left/ref_neck_topic", "/head/left/ref_neck")
        pitch_topic = rospy.get_param(ns + "right/ref_neck_topic", "/head/right/ref_neck")
        self.pub_yaw = rospy.Publisher("/" + self.robot + yaw_topic, Float64, queue_size=1)
        self.pub_pitch = rospy.Publisher("/" + self.robot + pitch_topic, Float64, queue_size=1)

    # ------------------------------------------------------------------------------------ callbacks
    def cb_pose(self, m):
        if len(m.data) < 3 + 3 * N_KP:
            return
        with self.lock:
            self.found = m.data[0] > 0.5
            self.kp2d = np.array(m.data[3:3 + 3 * N_KP]).reshape(N_KP, 3)
            self.t_pose = time.time()

    def cb_kp3d(self, m):
        if len(m.data) < 1 + 4 * N_KP:
            return
        with self.lock:
            self.kp3d = np.array(m.data[1:1 + 4 * N_KP]).reshape(N_KP, 4) if m.data[0] > 0.5 else np.zeros((N_KP, 4))
            self.t_kp3d = time.time()

    def cb_pos(self, m):
        with self.lock:
            self.pos = np.array([m.point.x, m.point.y, m.point.z])
            self.t_pos = time.time()

    def cb_face(self, m):
        if len(m.data) < 4:
            return
        with self.lock:
            self.t_face = time.time()
            if m.data[0] > 0.5:
                self.face = (float(m.data[1]), float(m.data[2]))   # yaw_left, pitch_up [rad] from the optical axis
                self.t_face_seen = self.t_face
            else:
                self.face = None

    def cb_upper(self, m):
        self.meas_l = np.array(list(m.left_meas_arm_shaft)[:self.n_cubes], dtype=float)
        self.meas_r = np.array(list(m.right_meas_arm_shaft)[:self.n_cubes], dtype=float)
        self.meas_neck = (float(m.left_meas_neck_shaft), float(m.right_meas_neck_shaft))

    def cb_lower(self, m):
        self.pitch = float(m.pitch_angle)
        if hasattr(m, "yaw_angle"):             # used to measure the pirouette
            self.yaw_imu = float(m.yaw_angle)
            self.t_yaw = time.time()

    def request_toggle(self, source):
        self.rospy.loginfo("person_follower: toggle (%s)", source)
        self.toggle_requested = True

    # ------------------------------------------------------------------------------------ helpers
    def fresh(self, t):
        return time.time() - t < self.P("person_timeout")

    def person_visible(self):
        with self.lock:
            return self.found and self.fresh(self.t_pose) and self.kp2d[NOSE, 2] >= self.P("kp_conf")

    def gesture(self):
        """Both wrists above the nose for gesture_hold s, after they were down; lockout after a toggle."""
        if not self.P("gesture_enabled"):
            return False
        with self.lock:
            k = self.kp2d.copy()
        c = self.P("kp_conf")
        up = self.person_visible() and k[L_WR, 2] >= c and k[R_WR, 2] >= c and k[L_WR, 1] < k[NOSE, 1] and k[R_WR, 1] < k[NOSE, 1]
        if not up:
            self.hands_up_since = None
            self.hands_were_down = True
            return False
        if self.hands_up_since is None:
            self.hands_up_since = time.time()
        if (time.time() - self.hands_up_since >= self.P("gesture_hold") and self.hands_were_down):
            self.hands_were_down = False
            return True
        return False

    def distance(self):
        with self.lock:
            if self.pos is None or not self.fresh(self.t_pos):
                return None
            return float(np.linalg.norm(self.pos))

    # ------------------------------------------------------------------------------------ start / stop
    def start(self):
        rospy = self.rospy
        if not self.person_visible():
            rospy.logwarn("person_follower: no person in view, not starting")
            return
        d = self.distance()
        if self.keep_distance and d is None:
            rospy.logwarn("person_follower: no valid depth on the person, not starting")
            return
        self.d_ref = d
        self.v = self.v_f = 0.0
        self.w = self.w_f = 0.0
        self.turning = False
        self.last_flex = {"r": None, "l": None}
        self.t_last_visible = time.time()
        self.gest.reset()
        self.primitive = None
        self.active = True
        rospy.loginfo("person_follower: STARTED  distance %s  (mirror=%s, distance=%s, head tracking=%s, base turn=%s, "
                      "arms=%s)", "%.2f m" % d if d is not None else "-", self.mirror, self.keep_distance,
                      self.track_head, self.turn_base, self.mimic_arms)

    def stop(self, why):
        self.active = False
        if self.primitive is not None:
            self.primitive = None
            self.vel_releasing = not (self.keep_distance or self.turn_base)
        self.gest.reset()
        self.v = self.v_f = 0.0
        self.w = self.w_f = 0.0
        self.turning = False
        self.q_r_tgt[:] = 0.0; self.q_l_tgt[:] = 0.0
        if not self.P("head_track_idle"):
            self.yaw_tgt = self.pitch_tgt = 0.0
        self.rospy.loginfo("person_follower: STOPPED (%s), arms back to zero", why)

    # ------------------------------------------------------------------------------------ targets
    def update_targets(self):
        P = self.P
        with self.lock:
            kp3d, kp2d, t3 = self.kp3d.copy(), self.kp2d.copy(), self.t_kp3d
        visible = self.person_visible()
        g = self.gest
        g_valid = not g.stale(time.time(), P("person_timeout"))

        # ---- FREEZE: hold the arm pose, base slows to a stop (head keeps looking at the person)
        if self.gesture_freeze and g_valid and g.freeze:
            self.ramp_base(0.0, 0.0)
            return

        # ---- EXPAND / ATTRACT: while the pose is held, the target distance moves
        if self.gesture_expand_attract and g_valid and self.d_ref is not None:
            if g.expand:
                self.d_ref = min(self.d_ref + P("expand_rate") * self.dt, max(P("max_follow_distance"), self.d_ref))
            elif g.attract:
                self.d_ref = max(self.d_ref - P("attract_rate") * self.dt, min(P("attract_min_distance"), self.d_ref))

        # ---- distance
        if self.keep_distance:
            d = self.distance()
            if visible and d is not None:
                e = d - self.d_ref                             # > 0: the person went away -> forward
                v = float(np.clip(P("k_lin") * dead_band(e, P("dead_lin")), -P("max_lin"), P("max_lin")))
                if d < P("min_distance") and v > 0:
                    v = 0.0                                    # never forward when already too close
            else:
                v = 0.0
                self.rospy.logwarn_throttle(1.0, "person_follower: person lost, base stopped")
            a = self.dt / (P("vel_filter_tau") + self.dt)
            self.v_f = (1 - a) * self.v_f + a * v
            step = P("acc_lin") * self.dt
            self.v = float(np.clip(self.v_f, self.v - step, self.v + step))

        # ---- base rotation: when the neck approaches its limit, the base turns the same way, so the robot keeps
        #      facing the person. The camera is on the head: while the base turns the face moves back toward the
        #      image center and the neck recenters by itself. Engages above turn_start, disengages below turn_stop.
        if self.turn_base:
            w = 0.0
            if visible:
                a = abs(self.yaw_cmd)
                if a > P("turn_start"):
                    self.turning = True
                elif a < P("turn_stop"):
                    self.turning = False
                if self.turning:
                    w = math.copysign(min(P("turn_gain") * (a - P("turn_stop")), P("turn_max")), self.yaw_cmd)
            else:
                self.turning = False
            a = self.dt / (P("turn_filter_tau") + self.dt)
            self.w_f = (1 - a) * self.w_f + a * w
            step = P("turn_acc") * self.dt
            self.w = float(np.clip(self.w_f, self.w - step, self.w + step))

        # ---- arms (the person's arm directions in the body frame, mapped on the robot)
        if self.mimic_arms and visible and self.fresh(t3):
            up_hint = camera_up(self.yaw_cmd, self.pitch_cmd) if P("camera_on_head") else (0.0, -1.0, 0.0)
            frame = person_frame(kp3d, up_hint)
            if frame is not None:
                arms = {s: arm_in_body(kp3d, frame, s) for s in ("left", "right")}
                if self.mirror:
                    # robot right <- person left, mirrored ; robot left <- person right, mirrored
                    src_r, src_l = arms["left"], arms["right"]
                    t_r = mirror_y(src_r[0]) if src_r[0] is not None else None
                    t_l = mirror_y(src_l[0]) if src_l[0] is not None else None
                else:
                    src_r, src_l = arms["right"], arms["left"]
                    t_r, t_l = src_r[0], src_l[0]
                # Elbow flexion None = wrist not seen: keep the last flexion of that arm instead of straightening it
                flex_r = src_r[1] if src_r[1] is not None else self.last_flex["r"]
                flex_l = src_l[1] if src_l[1] is not None else self.last_flex["l"]
                if t_r is not None:
                    self.q_r_tgt = self.arm.solve(t_r, flex_r, "r")[:self.n_cubes]
                    self.last_flex["r"] = flex_r
                if t_l is not None:
                    # left arm = mirror of the right arm solution for the mirrored direction
                    self.q_l_tgt = -self.arm.solve(mirror_y(t_l), flex_l, "l")[:self.n_cubes]
                    self.last_flex["l"] = flex_l

        # ---- ATTRACT: instead of copying the hands on the chest, the robot reaches both arms toward the person
        if self.mimic_arms and self.gesture_expand_attract and g_valid and g.attract:
            reach = np.zeros(self.n_cubes)
            reach[0], reach[3] = math.radians(P("attract_q0_deg")), math.radians(P("attract_elbow_deg"))
            self.q_r_tgt, self.q_l_tgt = reach.copy(), -reach

    # ------------------------------------------------------------------------------------ dance vocabulary
    def ramp_base(self, v_target, w_target):
        """Base velocities toward targets with the usual acceleration limits (freeze, primitives)."""
        P = self.P
        self.v_f = v_target
        self.v = float(np.clip(v_target, self.v - P("acc_lin") * self.dt, self.v + P("acc_lin") * self.dt))
        self.w_f = w_target
        self.w = float(np.clip(w_target, self.w - P("turn_acc") * self.dt, self.w + P("turn_acc") * self.dt))

    def update_gestures(self):
        """Give each new camera frame to the detector; start a primitive on an event."""
        with self.lock:
            kp3d, t3 = self.kp3d.copy(), self.t_kp3d
        if t3 == self.t_gest:
            return
        self.t_gest = t3
        f = None
        if self.person_visible() and self.fresh(t3):
            up_hint = camera_up(self.yaw_cmd, self.pitch_cmd) if self.P("camera_on_head") else (0.0, -1.0, 0.0)
            f = gestures.body_features(kp3d, person_frame(kp3d, up_hint))
        enabled = dict(expand_attract=self.gesture_expand_attract, pirouette=self.gesture_pirouette,
                       freeze=self.gesture_freeze, drop=self.gesture_drop)
        for name, arg in self.gest.update(time.time(), f, enabled):
            self.start_primitive(name, arg)
            if self.primitive is not None:
                break

    def start_primitive(self, name, arg):
        P, rospy, now = self.P, self.rospy, time.time()
        if name == "PIROUETTE":
            d = self.distance()
            if d is None or d < P("pirouette_min_distance"):
                rospy.logwarn("person_follower: pirouette ignored, person at %s (min %.1f m)",
                              "%.2f m" % d if d is not None else "unknown distance", P("pirouette_min_distance"))
                return
            if self.pub_vel is None:
                if not self.cmd_vel_topic:
                    rospy.logwarn("person_follower: pirouette ignored, CMD_VEL_IN_topic not set")
                    return
                self.pub_vel = self.rospy.Publisher(self.cmd_vel_topic, self.Twist, queue_size=1)
            # person's right arm up -> +1 = turn left (ROS convention), toward the person's right side
            direction = P("pirouette_direction") * (1.0 if arg == "right" else -1.0)
            self.primitive = dict(name=name, phase="prepare", t0=now, dir=direction, turned=0.0,
                                  last_yaw=None, side=arg)
            rospy.loginfo("person_follower: PIROUETTE (%s arm up) -> turn %s", arg, "left" if direction > 0 else "right")
        elif name == "DROP":
            self.primitive = dict(name=name, phase="down", t0=now)
            rospy.loginfo("person_follower: DROP")

    def update_primitive(self):
        """Run the current primitive: it owns base, arms and neck until it ends."""
        P, pr, now = self.P, self.primitive, time.time()
        z = np.zeros(self.n_cubes)
        if pr["name"] == "PIROUETTE":
            spread = z.copy(); spread[1] = -math.radians(P("pirouette_spread_deg"))
            if pr["phase"] == "prepare":
                self.q_r_tgt, self.q_l_tgt = spread.copy(), -spread
                self.yaw_tgt = pr["dir"] * P("pirouette_head_yaw")          # look toward the turn
                self.pitch_tgt = 0.0
                self.ramp_base(0.0, 0.0)
                if now - pr["t0"] >= P("pirouette_prepare_s"):
                    pr.update(phase="spin", t_spin=now)
            elif pr["phase"] == "spin":
                self.q_r_tgt, self.q_l_tgt = spread.copy(), -spread
                self.yaw_tgt = pr["dir"] * P("pirouette_head_yaw")
                # turned angle: from the IMU yaw if it is live, else from the commanded rate
                if self.yaw_imu is not None and now - self.t_yaw < 0.2:
                    if pr["last_yaw"] is not None:
                        dy = self.yaw_imu - pr["last_yaw"]
                        pr["turned"] += abs(math.atan2(math.sin(dy), math.cos(dy)))
                    pr["last_yaw"] = self.yaw_imu
                else:
                    pr["turned"] += abs(self.w) * self.dt
                    pr["last_yaw"] = None
                remaining = max(0.0, 2 * math.pi - pr["turned"])
                w_des = min(P("pirouette_speed"), math.sqrt(2 * P("pirouette_acc") * remaining))
                w_cmd = pr["dir"] * w_des
                self.w_f = w_cmd
                step = P("pirouette_acc") * self.dt
                self.w = float(np.clip(w_cmd, self.w - step, self.w + step))
                self.v = self.v_f = 0.0
                if remaining < math.radians(2) or now - pr["t_spin"] > P("pirouette_timeout_s"):
                    pr.update(phase="settle", t_settle=now)
            elif pr["phase"] == "settle":
                self.q_r_tgt, self.q_l_tgt = z.copy(), z.copy()
                self.yaw_tgt = self.pitch_tgt = 0.0
                self.ramp_base(0.0, 0.0)
                if now - pr["t_settle"] >= P("pirouette_settle_s") and abs(self.w) < 1e-3:
                    self.end_primitive("pirouette done, %.0f deg" % math.degrees(pr["turned"]))
        elif pr["name"] == "DROP":
            self.q_r_tgt, self.q_l_tgt = z.copy(), z.copy()
            self.pitch_tgt = P("drop_head_pitch") * P("pitch_sign")        # > 0 = head down
            self.ramp_base(0.0, 0.0)
            if now - pr["t0"] >= P("drop_hold_s"):
                self.end_primitive("drop done")

    def end_primitive(self, why):
        self.primitive = None
        self.grace_until = time.time() + self.P("pirouette_reacquire_s")
        self.t_last_visible = time.time()
        self.turning = False
        self.w = self.w_f = 0.0
        self.gest.reset()
        self.gest.cooldown_until = time.time() + self.P("gesture_cooldown_s")
        if not (self.keep_distance or self.turn_base):
            self.vel_releasing = True
        self.rospy.loginfo("person_follower: %s, back to mimicking", why)

    def gesture_label(self):
        if self.primitive is not None:
            pr = self.primitive
            if pr["name"] == "PIROUETTE":
                return "PIROUETTE %s %s" % (pr["phase"], "%.0f%%" % (100 * pr["turned"] / (2 * math.pi))
                                            if pr["phase"] == "spin" else "")
            return pr["name"]
        g = self.gest
        if not self.active or g.stale(time.time(), self.P("person_timeout")):
            return ""
        if self.gesture_freeze and g.freeze:
            return "FREEZE"
        if self.gesture_expand_attract and g.expand:
            return "EXPAND"
        if self.gesture_expand_attract and g.attract:
            return "ATTRACT"
        return ""

    def update_head(self):
        """Face tracking. camera_on_head (the RealSense is on the head, as in the URDF): the neck turns at
        k_head * error until the face is at the image center. Fixed camera: the head points at the face angle.
        Face lost: the head holds, after face_lost_return_s it goes slowly back to the center."""
        P = self.P
        with self.lock:
            face, t_face, t_seen = self.face, self.t_face, self.t_face_seen
        if face is not None and self.fresh(t_face):
            yaw_err = dead_band(face[0], P("head_dead"))      # > 0: face on the left -> turn left
            up_err = dead_band(face[1], P("head_dead"))       # > 0: face above -> look up (robot pitch < 0)
            if P("camera_on_head"):
                self.yaw_tgt += P("yaw_sign") * P("k_head") * yaw_err * self.dt
                self.pitch_tgt -= P("pitch_sign") * P("k_head") * up_err * self.dt
            else:
                self.yaw_tgt = P("yaw_sign") * face[0]
                self.pitch_tgt = -P("pitch_sign") * face[1]
        elif time.time() - t_seen > P("face_lost_return_s"):
            self.yaw_tgt = self.pitch_tgt = 0.0
        self.yaw_tgt = float(np.clip(self.yaw_tgt, -P("head_yaw_max"), P("head_yaw_max")))
        self.pitch_tgt = float(np.clip(self.pitch_tgt, -P("head_pitch_up_max"), P("head_pitch_down_max")))

    # ------------------------------------------------------------------------------------ commands / status
    def cb_command(self, m):
        try:
            ok, msg = self.apply_command(json.loads(m.data))
        except ValueError as e:
            ok, msg = False, "invalid JSON: %s" % e
        (self.rospy.loginfo if ok else self.rospy.logwarn)("person_follower: command %s: %s", m.data, msg)

    def apply_command(self, c):
        """Start/stop, options and parameters from the GUI or the command topic. Returns (ok, message)."""
        if "action" in c:
            if c["action"] in ("start", "stop"):
                self.pending = c["action"]
            elif c["action"] == "toggle":
                self.toggle_requested = True
            else:
                return False, "unknown action"
            return True, c["action"]
        if "option" in c:
            name, value = c["option"], bool(c.get("value"))
            if name not in OPTIONS:
                return False, "unknown option"
            if name in ("keep_distance", "turn_base"):
                if value and self.pub_vel is None:
                    if not self.cmd_vel_topic:
                        return False, "CMD_VEL_IN_topic not set (pilot.launch not running?)"
                    self.pub_vel = self.rospy.Publisher(self.cmd_vel_topic, self.Twist, queue_size=1)
                if not value and name == "keep_distance":
                    self.v = self.v_f = 0.0
                if not value and name == "turn_base":
                    self.w = self.w_f = 0.0
                    self.turning = False
                other = self.turn_base if name == "keep_distance" else self.keep_distance
                if not value and getattr(self, name) and not other:
                    self.vel_releasing = True   # nothing drives the base any more: one zero, then cmd_vel is free
            if name == "mimic_arms" and not value and self.mimic_arms:
                self.q_r_tgt[:] = 0.0; self.q_l_tgt[:] = 0.0
                self.arms_releasing = True
            if name == "track_head" and not value and self.track_head:
                self.yaw_tgt = self.pitch_tgt = 0.0
                self.head_releasing = True
            setattr(self, name, value)
            return True, "%s = %s" % (name, value)
        if "param" in c:
            name = c["param"]
            if name not in TUNABLE:
                return False, "not tunable from here: %s" % name
            lo, hi = TUNABLE[name]
            value = float(c["value"])
            if not lo <= value <= hi:
                return False, "%s must be in [%g, %g]" % (name, lo, hi)
            self.params[name] = value
            if name == "max_arm_elevation_deg":     # the arm lookup table depends on it
                self.arm = ArmModel(self.arm.m, self.P("q0_limits_deg"), self.P("q1_limits_deg"),
                                    self.P("elbow_limits_deg"), math.radians(self.P("max_arm_elevation_deg")),
                                    self.P("hand_above_shoulder_max"))
            return True, "%s = %g" % (name, self.params[name])
        return False, "empty command"

    def status(self):
        d = self.distance()
        f = lambda x: None if x is None else float(x)
        return dict(active=bool(self.active), d=f(d), d_ref=f(self.d_ref), v=float(self.v) if self.active else 0.0,
                    w=float(self.w) if self.active else 0.0,
                    yaw_cmd=float(self.yaw_cmd), pitch_cmd=float(self.pitch_cmd),
                    q_r=[float(x) for x in self.q_r], q_l=[float(x) for x in self.q_l],
                    person_visible=bool(self.person_visible()),
                    gesture=self.gesture_label(),
                    options={k: bool(getattr(self, k)) for k in OPTIONS},
                    params={k: float(self.P(k)) for k in TUNABLE})

    # ------------------------------------------------------------------------------------ outputs
    def step_commands(self, speed_deg_s):
        P = self.P
        a = self.dt / (P("joint_filter_tau") + self.dt)
        step = math.radians(speed_deg_s) * self.dt
        for q, tgt in ((self.q_r, self.q_r_tgt), (self.q_l, self.q_l_tgt)):
            f = (1 - a) * q + a * tgt if self.active else tgt
            q[:] = np.clip(f, q - step, q + step)
        hstep = math.radians(P("head_speed_deg_s")) * self.dt
        self.yaw_cmd = float(np.clip(self.yaw_tgt, self.yaw_cmd - hstep, self.yaw_cmd + hstep))
        self.pitch_cmd = float(np.clip(self.pitch_tgt, self.pitch_cmd - hstep, self.pitch_cmd + hstep))

    def publish(self):
        F, FA = self.Float64, self.Float64MultiArray
        P_sign = self.P("turn_sign")                     # -1 if the base turns away from the person
        prim = self.active and self.primitive is not None
        if self.pub_vel is not None and (self.keep_distance or self.turn_base or prim or self.vel_releasing):
            t = self.Twist()
            t.linear.x = self.v if (self.active and self.keep_distance and not prim) else 0.0
            t.angular.z = P_sign * self.w if (self.active and (self.turn_base or prim)) else 0.0
            self.pub_vel.publish(t)
            self.vel_releasing = False
        tol = math.radians(1.0)
        if self.arms_releasing and np.all(np.abs(self.q_r) < tol) and np.all(np.abs(self.q_l) < tol):
            self.arms_releasing = False
        if self.head_releasing and abs(self.yaw_cmd) < tol and abs(self.pitch_cmd) < tol:
            self.head_releasing = False
        if self.mimic_arms or self.arms_releasing:
            for s, q, k in (("right", self.q_r, self.stiff_r), ("left", self.q_l, self.stiff_l)):
                self.pub_eq[s].publish(FA(data=[float(x) for x in q]))
                self.pub_pr[s].publish(FA(data=[float(k[i]) if i < len(k) else 0.0 for i in range(self.n_cubes)]))
        if self.track_head or self.head_releasing:
            self.pub_yaw.publish(F(data=self.yaw_cmd))
            self.pub_pitch.publish(F(data=self.pitch_cmd))

    def at_rest(self):
        tol = math.radians(1.0)
        return (np.all(np.abs(self.q_r) < tol) and np.all(np.abs(self.q_l) < tol)
                and abs(self.yaw_cmd) < tol and abs(self.pitch_cmd) < tol)

    # ------------------------------------------------------------------------------------ checks
    def check_robot_side(self):
        rospy = self.rospy
        try:
            import rosnode
            nodes = rosnode.get_node_names()
        except Exception:  # noqa: BLE001
            return
        bad = [n for n in nodes if n.startswith("/" + self.robot + "/") and any(
            k in n for k in ("arm_inv_kin", "arm_inv_dyn", "head_inv_kin", "ego_dance", "ego_mirror"))]
        if bad and (self.mimic_arms or self.track_head):
            rospy.logfatal("person_follower: these nodes write the same cubes, stop them (body_movement?): %s", bad)
            raise SystemExit(1)
        if self.mimic_arms and "/" + self.robot + "/pitch_correction" not in nodes:
            rospy.logwarn("person_follower: pitch_correction NOT running: with the arms raised the LQR balances "
                          "around the wrong pitch. Start: roslaunch alterego_mimic pitch_correction.launch")

    def keyboard_thread(self):
        if not sys.stdin or not sys.stdin.isatty():
            self.rospy.loginfo("person_follower: no keyboard (roslaunch): toggle with the topic or the gesture")
            return
        self.rospy.loginfo("person_follower: press Enter to start / stop")
        while not self.stop_requested:
            line = sys.stdin.readline()
            if line == "":
                return
            self.request_toggle("Enter")

    # ------------------------------------------------------------------------------------ main loop
    def run(self):
        rospy = self.rospy
        rospy.loginfo("person_follower: waiting for alterego_state/upperbody")
        while self.meas_r is None and not rospy.is_shutdown() and not self.stop_requested:
            time.sleep(0.05)
        if self.meas_r is not None:
            self.q_r[:len(self.meas_r)] = self.meas_r
            self.q_l[:len(self.meas_l)] = self.meas_l
            self.yaw_cmd, self.pitch_cmd = self.meas_neck
        self.check_robot_side()
        threading.Thread(target=self.keyboard_thread, daemon=True).start()
        rospy.loginfo("person_follower: idle. Enter / toggle topic / both hands above the head for %.0f s to start",
                      self.P("gesture_hold"))

        rate = rospy.Rate(self.rate_hz)
        while not rospy.is_shutdown():
            if not self.cycle():
                rospy.loginfo("person_follower: arms and head at zero, exiting")
                break
            rate.sleep()

    def cycle(self):
        """One control period. False when the node has finished (Ctrl-C and everything back at zero).
        Called by run() and, with a simulated clock, by sim/simulate_follower.py."""
        if self.stop_requested:
            if self.active:
                self.stop("Ctrl-C")
            self.yaw_tgt = self.pitch_tgt = 0.0
            self.step_commands(self.P("descent_speed_deg_s"))
            self.publish()
            return not self.at_rest()

        if time.time() - self.t_status >= 0.1:
            self.t_status = time.time()
            self.pub_status.publish(self.String(data=json.dumps(self.status())))

        req, self.pending = self.pending, None
        if req == "start" and not self.active:
            self.last_toggle = time.time()
            self.start()
        elif req == "stop" and self.active:
            self.last_toggle = time.time()
            self.stop("command")

        toggle = self.toggle_requested or (time.time() - self.last_toggle > self.P("gesture_lockout") and self.gesture())
        if toggle:
            self.toggle_requested = False
            self.last_toggle = time.time()
            if self.active:
                self.stop("toggle")
            else:
                self.start()

        if self.active:
            if self.person_visible():
                self.t_last_visible = time.time()
            elif self.primitive is not None or time.time() < self.grace_until:
                pass        # a pirouette does not need the person; after it, time to find them again
            elif time.time() - self.t_last_visible > self.P("lost_stop_s"):
                # Person lost: stop, arms down slowly, head straight, back to idle. A new start is needed:
                # the robot does not resume by itself when someone appears again.
                self.stop("person lost for %.1f s" % self.P("lost_stop_s"))
                self.yaw_tgt = self.pitch_tgt = 0.0

        if self.active and abs(self.pitch) > self.P("max_pitch"):
            self.rospy.logerr("person_follower: pitch %.2f rad out of bound", self.pitch)
            self.stop("pitch out of bound")
            self.yaw_tgt = self.pitch_tgt = 0.0

        if self.active and self.primitive is None:
            self.update_gestures()

        if self.track_head and (self.active or self.P("head_track_idle")) and self.primitive is None:
            self.update_head()

        if self.active and self.primitive is not None:
            self.update_primitive()
        elif self.active:
            self.update_targets()
            d = self.distance()
            self.rospy.loginfo_throttle(1.0, "person_follower: d %s ref %.2f -> v %+.2f w %+.2f | head yaw %+.2f pitch %+.2f | "
                                        "R q0 %.0f q1 %.0f q3 %.0f  L q0 %.0f q1 %.0f q3 %.0f deg" % (
                                            "%.2f" % d if d is not None else "--", self.d_ref or 0, self.v, self.w,
                                            self.yaw_cmd, self.pitch_cmd, *np.degrees(self.q_r[[0, 1, 3]]),
                                            *np.degrees(self.q_l[[0, 1, 3]])))
        if self.active and self.primitive is not None and self.primitive["name"] == "DROP":
            speed = self.P("drop_arm_speed_deg_s")
        else:
            speed = self.P("joint_speed_deg_s") if self.active else self.P("descent_speed_deg_s")
        self.step_commands(speed)
        self.publish()
        return True

def main():
    import rospy
    rospy.init_node("person_follower", disable_signals=True)
    node = PersonFollower()

    def on_sigint(_sig, _frm):
        if node.stop_requested:
            rospy.logwarn("person_follower: already stopping, wait for the arms")
        node.stop_requested = True
    signal.signal(signal.SIGINT, on_sigint)
    signal.signal(signal.SIGTERM, on_sigint)
    node.run()
    rospy.signal_shutdown("done")


if __name__ == "__main__":
    main()
