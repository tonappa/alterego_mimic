#!/usr/bin/env python3
# Offline checks of person_follower.py, no ROS: python3 test/test_follower_offline.py
import math
import os
import sys
import types

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "scripts"))
import person_follower as pf  # noqa: E402

deg = np.degrees

# ------------------------------------------------------------------------------ arm model (robot right arm)
arm = pf.ArmModel(pf.V2_MODEL, [-20, 90], [-90, 10], [0, 110], math.radians(0), 0.0)
q = arm.solve([0, 0, -1], 0.0);          assert abs(deg(q[0])) <= 2 and abs(deg(q[1])) <= 2, q        # arm down
q = arm.solve([1, 0, 0], 0.0);           assert abs(deg(q[0]) - 90) <= 3.5 and abs(deg(q[1])) <= 3.5, q   # forward
q = arm.solve([0, -1, 0], 0.0);          assert abs(deg(q[1]) + 90) <= 2, q                           # out to the side
q = arm.solve([1, 0, 1], 0.0)                                                                           # 45 deg above
p = arm.points(q);                       assert p[2][2] <= arm.shoulder[2] + 1e-6 and abs(deg(q[0]) - 90) <= 2, q
arm.solve([0, -1, 0], 0.0, "k"); q = arm.solve([0, 0, 1], 0.0, "k")                                     # straight up
assert abs(deg(q[1]) + 90) <= 2, "straight up keeps the previous azimuth (side)"
q = arm.solve([0, 0, -1], math.radians(90))                                                             # elbow 90, arm down
assert abs(deg(q[3]) - 90) < 1e-6 and arm.points(q)[4][0] > 0.1, "forearm forward"
q = arm.solve([1, 0, 0], math.radians(90))                                                              # arm forward, elbow 90
assert arm.points(q)[4][2] <= arm.shoulder[2] + 1e-6, "hand kept at shoulder height"
# gimbal: arm out to the side, reached from arm forward (q0 = 90) or from arm down (q0 = 0): no q0 jump
a2 = pf.ArmModel(pf.V2_MODEL, [-20, 90], [-90, 10], [0, 110], 0.0, 0.0)
a2.solve([1, 0, 0], 0.0, "g"); q_side_from_fwd = a2.solve([0.05, -1, 0], 0.0, "g")
a2.solve([0, 0, -1], 0.0, "h"); q_side_from_down = a2.solve([0.05, -1, 0], 0.0, "h")
for q in (q_side_from_fwd, q_side_from_down):
    p = a2.points(q); d = (p[2] - p[0]) / np.linalg.norm(p[2] - p[0])
    assert d @ np.array([0.05, -1, 0]) / np.linalg.norm([0.05, -1, 0]) > 0.99
assert deg(q_side_from_fwd[0]) > 60 and deg(q_side_from_down[0]) < 30, (deg(q_side_from_fwd), deg(q_side_from_down))
print("arm model OK  (arm forward + elbow 90 -> elbow reduced to %.0f deg)" % deg(q[3]))

# ------------------------------------------------------------------------------ person facing the camera, 2 m
def person(right_elbow_offset=(0, 0.3, 0), left_elbow_offset=(0, 0.3, 0), wrist_extra=(0, 0.25, 0)):
    """3D keypoints in the camera optical frame (x right, y down, z forward). Person's left = image right."""
    k = np.zeros((17, 4))
    k[pf.L_SH] = [0.2, -0.3, 2.0, 1]; k[pf.R_SH] = [-0.2, -0.3, 2.0, 1]
    k[pf.L_HIP] = [0.15, 0.2, 2.0, 1]; k[pf.R_HIP] = [-0.15, 0.2, 2.0, 1]
    k[pf.R_EL, :3] = k[pf.R_SH, :3] + right_elbow_offset; k[pf.R_EL, 3] = 1
    k[pf.L_EL, :3] = k[pf.L_SH, :3] + left_elbow_offset; k[pf.L_EL, 3] = 1
    k[pf.R_WR, :3] = k[pf.R_EL, :3] + wrist_extra; k[pf.R_WR, 3] = 1
    k[pf.L_WR, :3] = k[pf.L_EL, :3] + wrist_extra; k[pf.L_WR, 3] = 1
    return k

F = pf.person_frame(person())
assert np.allclose(F[0], [0, 0, -1]) and np.allclose(F[1], [1, 0, 0]) and np.allclose(F[2], [0, -1, 0]), F
d, flex = pf.arm_in_body(person(), F, "right");            assert np.allclose(d, [0, 0, -1]) and flex < 1e-6
d, _ = pf.arm_in_body(person(right_elbow_offset=(0, 0, -0.3)), F, "right");  assert np.allclose(d, [1, 0, 0])   # toward the camera
d, _ = pf.arm_in_body(person(right_elbow_offset=(-0.3, 0, 0)), F, "right");  assert np.allclose(d, [0, -1, 0])  # to its right
_, flex = pf.arm_in_body(person(wrist_extra=(0, 0, -0.25)), F, "right");     assert abs(deg(flex) - 90) < 1e-6
print("person frame / arm direction OK")

# ------------------------------------------------------------------------------ node logic with fake ROS
pubs = {}
class Pub:
    def __init__(self, name, *a, **k): self.name = name; pubs[name] = self; self.last = None
    def publish(self, m): self.last = m
Msg = lambda **kw: types.SimpleNamespace(**kw)
class Twist:
    def __init__(self): self.linear = types.SimpleNamespace(x=0.0); self.angular = types.SimpleNamespace(z=0.0)
params = {}
get_calls = []
rospy = types.SimpleNamespace(
    Publisher=Pub, Subscriber=lambda *a, **k: None, get_param=lambda n, d=None: (get_calls.append(n), params.get(n, d))[1],
    loginfo=lambda *a: None, logwarn=lambda *a: None, logerr=lambda *a: None, logfatal=print,
    logwarn_throttle=lambda *a: None, loginfo_throttle=lambda *a: None)
sys.modules["rospy"] = rospy
for mod, names in [("std_msgs.msg", ["Float64", "Float64MultiArray", "Empty", "String"]),
                   ("geometry_msgs.msg", ["PointStamped"]), ("alterego_msgs.msg", ["UpperBodyState", "LowerBodyState"])]:
    m = types.ModuleType(mod)
    for n in names:
        setattr(m, n, lambda **kw: types.SimpleNamespace(**kw))
    sys.modules[mod] = m
sys.modules["geometry_msgs.msg"].Twist = Twist
os.environ["ROBOT_NAME"] = "rb"
params["~rate"] = 50                     # the loops below count 50 Hz cycles, whatever follower.yaml says
params["/rb/CMD_VEL_IN_topic"] = "cmd_vel"
params["/rb/left/stiffness_vec"] = [0.2] * 5; params["/rb/right/stiffness_vec"] = [0.2] * 5

def run(mirror):
    params["~mirror"] = mirror
    n = pf.PersonFollower()
    now = pf.time.time()
    n.found = True; n.kp2d = np.zeros((17, 3)); n.kp2d[pf.NOSE] = [0.50, 0.30, 0.9]; n.t_pose = now
    n.kp3d = person(); n.t_kp3d = now
    n.pos = np.array([0.0, 0.0, 2.0]); n.t_pos = now
    n.start()
    assert n.active and abs(n.d_ref - 2.0) < 1e-9
    # person 0.5 m closer, raises the RIGHT arm forward
    n.pos = np.array([0.0, 0.0, 1.5])
    n.kp3d = person(right_elbow_offset=(0, 0, -0.3), wrist_extra=(0, 0, -0.25))
    for _ in range(300):                       # 6 s at 50 Hz
        n.update_targets(); n.step_commands(60); n.publish()
    return n

n = run(True)
assert n.v < -0.1, "person closer -> backwards"
assert abs(deg(n.q_l[0]) + 90) <= 3 and abs(deg(n.q_r[0])) <= 3, "mirror: robot LEFT arm forward (left = -right)"
assert pubs["/rb/cmd_vel"].last.linear.x == n.v and len(pubs["/rb/left/ref_cubes_eq"].last.data) == 5
print("mirror    OK  v %+.2f  L q0 %.0f  R q0 %.0f" % (n.v, deg(n.q_l[0]), deg(n.q_r[0])))
n = run(False)
assert abs(deg(n.q_r[0]) - 90) <= 3 and abs(deg(n.q_l[0])) <= 3
print("imitation OK  R q0 %.0f  L q0 %.0f" % (deg(n.q_r[0]), deg(n.q_l[0])))

# person lost -> base stops; stop -> everything back to zero
n.t_pose = n.t_pos = n.t_kp3d = 0.0
for _ in range(100):
    n.update_targets(); n.step_commands(60)
assert abs(n.v) < 1e-3, "person lost -> v ramps to 0"
n.stop("test")
for _ in range(600):
    n.step_commands(20)
assert n.at_rest()
print("person lost / stop OK")

# ------------------------------------------------------------------------------ face tracking, camera on the head
# World: the face is at yaw F_YAW (left +) and F_UP (up +) from the robot. The camera turns with the head, so
# the tracker sees  yaw_left = F_YAW - yaw_cmd,  pitch_up = F_UP - (-pitch_cmd)  (robot pitch < 0 = up),
# with 0.2 s of camera + network delay and new data at 15 Hz.
def track(f_yaw, f_up, seconds=6.0, delay=0.2, lose_after=None):
    n = pf.PersonFollower()
    hist, t, last_frame = [], 0.0, -1.0
    for k in range(int(seconds * 50)):
        t = k * 0.02
        hist.append((n.yaw_cmd, n.pitch_cmd))
        if t - last_frame >= 1 / 15:
            last_frame = t
            y, p = hist[max(0, len(hist) - 1 - int(delay / 0.02))]
            seen = lose_after is None or t < lose_after
            n.face = (f_yaw - y, f_up + p) if seen else None
            n.t_face = pf.time.time()
            if seen:
                n.t_face_seen = pf.time.time()
        n.update_head(); n.step_commands(60)
    return n

n = track(0.4, 0.2)
assert abs(n.yaw_cmd - 0.4) < 0.04 and abs(n.pitch_cmd + 0.2) < 0.04, (n.yaw_cmd, n.pitch_cmd)
print("face tracking OK  face yaw 0.40 up 0.20 -> neck yaw %+.2f pitch %+.2f" % (n.yaw_cmd, n.pitch_cmd))
n = track(1.2, -0.8)
assert abs(n.yaw_cmd - 0.6) < 1e-6 and abs(n.pitch_cmd - 0.3) < 1e-6
print("face tracking limits OK  -> yaw %+.2f pitch %+.2f" % (n.yaw_cmd, n.pitch_cmd))
n = track(0.4, 0.0, seconds=4.0, lose_after=3.0)
assert abs(n.yaw_cmd - 0.4) < 0.05, "face lost for 1 s: head holds"
print("face lost: head holds OK")
# ------------------------------------------------------------------------------ parameters and commands
params["~mirror"] = True
n = pf.PersonFollower()
get_calls.clear()
for _ in range(50):
    n.cycle()
assert not get_calls, "no get_param after the constructor (each one would be an XMLRPC call): %s" % get_calls[:3]
ok, _ = n.apply_command({"param": "k_lin", "value": 1.5}); assert ok and n.P("k_lin") == 1.5
ok, msg = n.apply_command({"param": "k_lin", "value": 9.0}); assert not ok, "out of range refused"
ok, msg = n.apply_command({"param": "rate", "value": 10}); assert not ok, "not tunable refused"
# keep_distance switched off: exactly one zero cmd_vel, then nothing (the pilot is free again)
vel = pubs["/rb/cmd_vel"]; count = []
vel.publish = lambda m, _c=count: _c.append(m.linear.x)
n.cycle(); assert len(count) == 1
n.apply_command({"option": "keep_distance", "value": False})
for _ in range(20):
    n.cycle()
assert len(count) == 21, "turn_base still on: cmd_vel keeps being published"
n.apply_command({"option": "turn_base", "value": False})
for _ in range(20):
    n.cycle()
assert count[21:] == [0.0], "both off: one zero, then cmd_vel is free"
n.apply_command({"option": "keep_distance", "value": True}); n.cycle(); assert len(count) == 23
n.apply_command({"option": "turn_base", "value": True})
# start / stop by command
n.apply_command({"action": "start"}); n.cycle(); assert not n.active, "nobody in view: start refused"
now = pf.time.time()
n.found = True; n.kp2d = np.zeros((17, 3)); n.kp2d[pf.NOSE] = [0.5, 0.3, 0.9]; n.t_pose = now
n.pos = np.array([0.0, 0.0, 2.0]); n.t_pos = now
n.apply_command({"action": "start"}); n.cycle(); assert n.active
n.apply_command({"action": "stop"}); n.cycle(); assert not n.active
print("parameters read once, commands, cmd_vel released  OK")
# ------------------------------------------------------------------------------ base rotation (turn_base)
n = pf.PersonFollower()
n.apply_command({"option": "track_head", "value": False})          # hold the neck where the test puts it
def see(n):
    now = pf.time.time()
    n.found = True; n.kp2d = np.zeros((17, 3)); n.kp2d[pf.NOSE] = [0.5, 0.3, 0.9]; n.t_pose = now
    n.pos = np.array([0.0, 0.0, 2.0]); n.t_pos = now
def run_neck(n, yaw, cycles=100):
    for _ in range(cycles):
        see(n); n.yaw_cmd = n.yaw_tgt = yaw; n.cycle()
    return n.w, pubs["/rb/cmd_vel"].last.angular.z
w, wz = run_neck(n, 0.5);  assert w == 0.0 and wz == 0.0, "idle: the base never turns"
run_neck(n, 0.0, 5)
see(n); n.apply_command({"action": "start"}); n.cycle(); assert n.active
w, wz = run_neck(n, 0.2);  assert w == 0.0, "neck below turn_start: no rotation"
w, wz = run_neck(n, 0.5);  assert 0.3 < w <= n.P("turn_max") + 1e-9 and wz == w, (w, wz)       # neck left -> turn left
w, _ = run_neck(n, 0.2);   assert 0.05 < w < 0.15, "hysteresis: still turning between turn_stop and turn_start"
w, _ = run_neck(n, 0.05);  assert abs(w) < 1e-3, "neck back within turn_stop: base stops"
w, wz = run_neck(n, -0.5); assert w < -0.3 and wz == w, "neck right -> turn right"
n.found = False; n.t_pose = 0.0
for _ in range(100):
    n.yaw_cmd = n.yaw_tgt = -0.5; n.cycle()
assert abs(n.w) < 1e-3, "person lost: rotation stops"
print("base rotation: idle never, threshold, hysteresis, both sides, person lost  OK")
# ------------------------------------------------------------------------------ elbow hold and person lost
clock = [1000.0]
real_time = pf.time
pf.time = types.SimpleNamespace(time=lambda: clock[0], sleep=lambda x: None)
n = pf.PersonFollower()
def seen(n, kp3d):
    n.found = True; n.kp2d = np.zeros((17, 3)); n.kp2d[pf.NOSE] = [0.5, 0.3, 0.9]; n.t_pose = clock[0]
    n.pos = np.array([0.0, 0.0, 2.0]); n.t_pos = clock[0]; n.kp3d = kp3d; n.t_kp3d = clock[0]
def cycles(n, seconds, kp3d=None):
    for _ in range(int(seconds * 50)):
        clock[0] += 0.02
        if kp3d is not None:
            seen(n, kp3d)
        n.cycle()
bent = person(wrist_extra=(0, 0, -0.25))                     # arms down, elbows bent 90 deg, wrists visible
seen(n, bent); n.apply_command({"action": "start"}); cycles(n, 3.0, bent)
assert n.active and abs(deg(n.q_r[3]) - 90) < 3 and abs(deg(n.q_l[3]) + 90) < 3, (deg(n.q_r), deg(n.q_l))
no_wrist = bent.copy(); no_wrist[[pf.L_WR, pf.R_WR], 3] = 0
cycles(n, 2.0, no_wrist)
assert abs(deg(n.q_r[3]) - 90) < 3 and abs(deg(n.q_l[3]) + 90) < 3, "wrist not seen: the elbow keeps its flexion"
print("elbow: wrist not seen -> flexion kept (R %.0f, L %.0f deg)  OK" % (deg(n.q_r[3]), deg(n.q_l[3])))

fwd = person(right_elbow_offset=(0, 0, -0.3), left_elbow_offset=(0, 0, -0.3))   # both arms forward
cycles(n, 3.0, fwd)
assert abs(deg(n.q_r[0]) - 90) < 3 and n.active
n.found = False                                               # the person disappears
cycles(n, 1.0)
assert n.active and n.v == 0.0 and abs(deg(n.q_r[0]) - 90) < 3, "within lost_stop_s: base stopped, arms hold"
cycles(n, 1.2)
assert not n.active, "lost for more than lost_stop_s: back to idle"
cycles(n, 6.0)
assert n.at_rest(), "arms down and head straight"
cycles(n, 2.0, fwd)
assert not n.active, "the person reappears: no automatic restart"
print("person lost: base stops, then idle with arms down and head straight, no automatic restart  OK")
pf.time = real_time
print("ALL OK")
