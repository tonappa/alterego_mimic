#!/usr/bin/env python3
# Offline checks, no ROS, no camera, no ultralytics: python3 test/test_offline.py
# 1) pure functions (depth window, torso pixel, person selection, message layout)
# 2) PersonTracker.process() end to end with fake rospy / pyrealsense2 / YOLO result
import math
import os
import sys
import types

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "scripts"))
import person_tracker as pt  # noqa: E402


# ------------------------------------------------------------------------------ 1) pure functions
d = np.zeros((480, 640), np.float32)
d[200:260, 300:340] = 2.0
d[230, 320] = 0.0                                    # hole in the middle: ignored by the median
assert abs(pt.patch_depth(d, 320, 230, 3, 0.2, 6.0) - 2.0) < 1e-6
assert pt.patch_depth(d, 10, 10, 3, 0.2, 6.0) == 0.0                 # no valid depth
assert pt.patch_depth(d, 700, 10, 3, 0.2, 6.0) == 0.0                # outside the image
d2 = d.copy(); d2[200:260, 300:340] = 9.0
assert pt.patch_depth(d2, 320, 230, 3, 0.2, 6.0) == 0.0              # beyond max_depth

kp = np.zeros((17, 2)); conf = np.zeros(17)
kp[5] = [300, 200]; kp[6] = [340, 200]; kp[11] = [305, 280]; kp[12] = [335, 280]
conf[[5, 6, 11, 12]] = 0.9
assert pt.torso_pixel(kp, conf, (0, 0, 100, 100), 0.3) == (320.0, 240.0)
conf[:] = 0
assert pt.torso_pixel(kp, conf, (0, 0, 100, 50), 0.3) == (50.0, 25.0)  # falls back to the box center

boxes = [(0, 0, 100, 100), (0, 0, 50, 50)]
assert pt.select_person(boxes, [3.0, 1.0], "largest") == 0
assert pt.select_person(boxes, [3.0, 1.0], "nearest") == 1
assert pt.select_person(boxes, [0.0, 0.0], "nearest") == 0            # no depth: largest

data = pt.pose_message_data(False, 640, 480)
assert len(data) == 3 + 51 and data[:3] == [0.0, 640.0, 480.0] and not any(data[3:])
xyn = np.arange(34).reshape(17, 2) / 100.0; c = np.full(17, 0.5)
data = pt.pose_message_data(True, 640, 480, xyn, c)
assert data[0] == 1.0 and data[3 + 3 * 16] == xyn[16, 0] and data[5 + 3 * 16] == 0.5
kpf = np.zeros((17, 2)); cf = np.zeros(17)
kpf[1] = [100, 50]; kpf[2] = [120, 50]; cf[[1, 2]] = 0.9
assert pt.face_pixel(kpf, cf, 0.3) == ((110.0, 50.0), False)         # no nose: eyes
cf[0] = 0.9; kpf[0] = [110, 60]
assert pt.face_pixel(kpf, cf, 0.3) == ((110.0, 60.0), False)         # nose
assert pt.face_pixel(kpf, np.zeros(17), 0.3) == (None, False)
cs = np.zeros(17); kps = np.zeros((17, 2)); kps[5] = [300, 100]; kps[6] = [200, 100]; cs[[5, 6]] = 0.9
assert pt.face_pixel(kps, cs, 0.3) == ((250.0, 40.0), True)          # face out of view: above the shoulders
print("pure functions OK")


# ------------------------------------------------------------------------------ 2) process() end to end
published = {}


class FakePub:
    def __init__(self, name, *_a, **_k):
        self.name = name

    def publish(self, msg):
        published.setdefault(self.name, []).append(msg)

    def get_num_connections(self):
        return 0


class Msg:
    def __init__(self, data=None):
        self.data = data
        self.header = types.SimpleNamespace(stamp=None, frame_id="")
        self.point = types.SimpleNamespace(x=0.0, y=0.0, z=0.0)


fake_rospy = types.SimpleNamespace(
    Publisher=FakePub, get_param=lambda name, default: default, Time=types.SimpleNamespace(now=lambda: 0.0),
    loginfo=print, logwarn=print, logerr=print, logfatal=print)
sys.modules["rospy"] = fake_rospy
for mod, names in [("std_msgs.msg", ["Float64MultiArray"]), ("geometry_msgs.msg", ["PointStamped"]),
                   ("sensor_msgs.msg", ["CompressedImage"])]:
    m = types.ModuleType(mod)
    for n in names:
        setattr(m, n, Msg)
    sys.modules[mod] = m
    sys.modules.setdefault(mod.split(".")[0], types.ModuleType(mod.split(".")[0]))

os.environ["ROBOT_NAME"] = "robot_test"
pt.PersonTracker.load_model = lambda self, w: None      # no ultralytics
tr = pt.PersonTracker()


# pinhole deprojection like librealsense (no distortion)
class Intr:
    fx = fy = 600.0; ppx = 320.0; ppy = 240.0


tr.rs = types.SimpleNamespace(rs2_deproject_pixel_to_point=lambda i, uv, z: [
    (uv[0] - i.ppx) / i.fx * z, (uv[1] - i.ppy) / i.fy * z, z])


class T:  # tensor-like
    def __init__(self, a):
        self.a = np.asarray(a, dtype=float)

    def cpu(self):
        return self

    def numpy(self):
        return self.a


def person(cx, cy, half):
    k = np.zeros((17, 2))
    k[0] = [cx, cy - 2 * half]                         # nose
    k[5] = [cx - half, cy - half]; k[6] = [cx + half, cy - half]
    k[11] = [cx - half, cy + half]; k[12] = [cx + half, cy + half]
    return k


# two people: A big at the center (far), B small on the right (near)
kpA, kpB = person(320, 240, 40), person(500, 240, 20)
boxesA = [320 - 60, 240 - 120, 320 + 60, 240 + 150]; boxesB = [500 - 30, 240 - 60, 500 + 30, 240 + 70]
confs = np.zeros((2, 17)); confs[:, [0, 5, 6, 11, 12]] = 0.9
result = types.SimpleNamespace(
    boxes=types.SimpleNamespace(xyxy=T([boxesA, boxesB]), __len__=None),
    keypoints=types.SimpleNamespace(xy=T([kpA, kpB]), xyn=T(np.stack([kpA, kpB]) / [640, 480]), conf=T(confs)),
    plot=lambda: np.zeros((480, 640, 3), np.uint8))
result.boxes = type("B", (), {"xyxy": T([boxesA, boxesB]), "__len__": lambda s: 2})()

depth = np.zeros((480, 640), np.float32)
depth[:, 200:440] = 3.0                               # A at 3 m
depth[:, 460:540] = 1.5                               # B at 1.5 m
color = np.zeros((480, 640, 3), np.uint8)
tr.model = lambda img, **kw: [result]

for mode, expect_u, expect_z in [("largest", 320, 3.0), ("nearest", 500, 1.5)]:
    published.clear()
    tr.select = mode
    tr.process(color, depth, Intr())
    pose = published["/robot_test/person_pose"][-1].data
    pos = published["/robot_test/person_position"][-1].point
    k3 = published["/robot_test/person_keypoints_3d"][-1].data
    assert pose[0] == 1.0 and abs(pose[3 + 3 * 5] * 640 - (expect_u - (40 if mode == "largest" else 20))) < 1e-6
    assert abs(pos.z - expect_z) < 1e-6 and abs(pos.x - (expect_u - 320) / 600.0 * expect_z) < 1e-6
    assert k3[0] == 1.0 and k3[4 * 5 + 4] == 1.0 and abs(k3[4 * 5 + 3] - expect_z) < 1e-6   # left shoulder valid
    assert k3[4 * 9 + 4] == 0.0                                                            # wrist not visible
    face = published["/robot_test/person_face"][-1].data                                    # nose at (u, 240 - 2 half)
    half = 40 if mode == "largest" else 20
    assert face[0] == 1.0 and abs(face[1] - math.atan2(-(expect_u - 320) / 600.0, 1)) < 1e-9
    assert abs(face[2] - math.atan2(2 * half / 600.0, 1)) < 1e-9 and face[2] > 0            # above the center
    print("process() %-7s -> torso X %.2f Z %.2f  OK" % (mode, pos.x, pos.z))

# nobody in the image
published.clear()
empty = types.SimpleNamespace(boxes=type("B", (), {"__len__": lambda s: 0})(), keypoints=None,
                              plot=lambda: np.zeros((480, 640, 3), np.uint8))
tr.model = lambda img, **kw: [empty]
tr.process(color, depth, Intr())
assert published["/robot_test/person_pose"][-1].data[0] == 0.0
assert "/robot_test/person_position" not in published
assert published["/robot_test/person_keypoints_3d"][-1].data[0] == 0.0
assert published["/robot_test/person_face"][-1].data[0] == 0.0
print("process() no person -> found 0, no position  OK")
print("ALL OK")
