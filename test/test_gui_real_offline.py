#!/usr/bin/env python3
# Offline check of the GUI real-robot backend with a fake ROS (no robot): python3 test/test_gui_real_offline.py
import json, math, os, sys, types
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "gui")); sys.path.insert(0, os.path.join(HERE, "..", "sim"))

subs, pubs = {}, {}
class Pub:
    def __init__(self, name, *a, **k): self.name, self.msgs = name, []; pubs[name] = self
    def publish(self, m): self.msgs.append(m)
M = lambda **kw: types.SimpleNamespace(**kw)
rospy = types.SimpleNamespace(Subscriber=lambda name, cls, cb, **k: subs.__setitem__(name, cb), Publisher=Pub,
                              core=types.SimpleNamespace(is_initialized=lambda: False), init_node=lambda *a, **k: None,
                              get_param=lambda n, d=None: d)
master_up = [True]
class Master:
    def __init__(self, n): pass
    def getPid(self):
        if not master_up[0]: raise OSError("refused")
        return 1
fake = {"rospy": rospy, "rosgraph": types.SimpleNamespace(Master=Master)}
for mod, names in [("std_msgs.msg", ["Float64MultiArray", "String"]), ("geometry_msgs.msg", ["PointStamped"]),
                   ("sensor_msgs.msg", ["CompressedImage"]), ("rosgraph_msgs.msg", ["Log"]),
                   ("alterego_msgs.msg", ["UpperBodyState", "LowerBodyState"])]:
    m = types.ModuleType(mod)
    for n in names: setattr(m, n, M)
    fake[mod] = m
sys.modules.update(fake)

import sim_core, real_bridge   # noqa: E402
kin = sim_core.RobotKinematics()

# master down -> clear error
master_up[0] = False
try:
    real_bridge.RealBackend(kin, "rb"); raise SystemExit("should fail")
except real_bridge.RealUnavailable as e:
    print("master down ->", e)
master_up[0] = True
try:
    real_bridge.RealBackend(kin, ""); raise SystemExit("should fail")
except real_bridge.RealUnavailable as e:
    print("no robot name ->", e)

rb = real_bridge.RealBackend(kin, "rb")
ns = "/rb/"
assert set(subs) >= {ns + "alterego_state/upperbody", ns + "person_keypoints_3d", ns + "person_follower/status", "/rosout"}
# robot: right arm forward, neck yaw 0.3; wheels: 1 rad forward then turn 90 deg, then 1 rad
subs[ns + "alterego_state/upperbody"](M(right_meas_arm_shaft=[math.pi / 2, 0, 0, 0, 0], left_meas_arm_shaft=[0] * 5,
                                        left_meas_neck_shaft=0.3, right_meas_neck_shaft=-0.2))
for wheel, yaw in ((0.0, 0.1), (1.0, 0.1), (1.0, 0.1 + math.pi / 2), (2.0, 0.1 + math.pi / 2)):
    subs[ns + "alterego_state/lowerbody"](M(wheels_angular_pos=wheel, yaw_angle=yaw, pitch_angle=0.0))
# person: torso 2 m in front of the camera
kp = [1.0] + [0.0] * 68
kp[1 + 4 * 5:5 + 4 * 5] = [0.2, 0.0, 2.0, 1.0]; kp[1 + 4 * 6:5 + 4 * 6] = [-0.2, 0.0, 2.0, 1.0]
subs[ns + "person_keypoints_3d"](M(data=kp))
subs[ns + "person_pose"](M(data=[1.0, 640, 480] + [0.5, 0.5, 0.9] * 17))
subs[ns + "person_tracker/image/compressed"](M(data=b"\xff\xd8jpeg"))
subs[ns + "person_face"](M(data=[1.0, 0.1, 0.2, 1.8]))
subs[ns + "person_follower/status"](M(data=json.dumps(dict(active=True, d=2.0, d_ref=1.9, v=0.1, options={"mirror": True}, params={"k_lin": 0.8}))))
subs["/rosout"](M(name="/rb/person_follower", msg="person_follower: STARTED", header=M(stamp=M(to_sec=lambda: 0.0))))
subs["/rosout"](M(name="/other_node", msg="ignored", header=M(stamp=M(to_sec=lambda: 0.0))))

s = rb.state(["base", "right_hand", "fake_realsense"])
print("odometry x %.3f y %.3f yaw %.2f" % (s["base_x"], s["base_y"], s["base_yaw"]))
# yaw_angle grows by pi/2 = the base turned RIGHT (the LQR uses des_yaw_rate = -angular.z): heading -pi/2, y < 0
assert abs(s["base_x"] - 0.125) < 1e-6 and abs(s["base_y"] + 0.125) < 1e-6 and abs(s["base_yaw"] + math.pi / 2) < 1e-9
assert s["kw_valid"][5] and s["kw_valid"][6] and not s["kw_valid"][0]
T = np.array(s["T_cam"]).reshape(4, 4)
mid = (np.array(s["kw"][5]) + np.array(s["kw"][6])) / 2
assert abs(np.linalg.norm(mid - T[:3, 3]) - 2.0) < 1e-3, "person 2 m from the camera, in the world"
assert s["active"] is True and s["follower_ok"] and s["image"] and s["events"] == [s["events"][0]] and "STARTED" in s["events"][0]
assert all(c["ok"] for c in s["connections"]), s["connections"]
assert abs(s["yaw"] - math.degrees(0.3)) < 1e-6 and abs(s["q_r"][0] - 90) < 1e-6
fu, fv = s["view"]["face_uv"]
assert abs(fu - (320 - 615 * math.tan(0.1))) < 1e-9 and abs(fv - (240 - 615 * math.tan(0.2))) < 1e-9 and not s["view"]["face_est"]
print("state OK: person %.2f m from the camera, connections all live, log filtered" % np.linalg.norm(mid - T[:3, 3]))

rb.control({"action": "start"}); rb.control({"action": "param", "name": "k_lin", "value": 1.2})
sent = [json.loads(m.data) for m in pubs[ns + "person_follower/command"].msgs]
assert sent == [{"action": "start"}, {"param": "k_lin", "value": 1.2}], sent
try:
    rb.control({"action": "person", "values": {"x": 1}}); raise SystemExit("should fail")
except ValueError as e:
    print("sim-only action ->", e)
rb.stamp["status"] -= 10
try:
    rb.control({"action": "start"}); raise SystemExit("should fail")
except ValueError as e:
    print("follower down ->", e)
print("ALL OK")
