"""Real-robot backend of the GUI: a ROS node that observes the robot and commands person_follower.

Reads   /$ROBOT_NAME/alterego_state/upperbody   measured arm cubes and neck (left = yaw, right = pitch)
        /$ROBOT_NAME/alterego_state/lowerbody   wheel angle and yaw -> odometry for the 3D view
        /$ROBOT_NAME/person_pose, person_keypoints_3d, person_face, person_tracker/image/compressed  (tracker)
        /$ROBOT_NAME/person_follower/status     (follower state, JSON)
        /rosout                                 (log lines of the follower)
Writes  /$ROBOT_NAME/person_follower/command    (start / stop / toggle, options, parameters, JSON)

The GUI never publishes robot commands itself: base, arms and neck are driven only by person_follower.
"""
import collections
import json
import math
import os
import socket
import threading
import time

import numpy as np

_INSTANCE = None


class RealUnavailable(RuntimeError):
    pass


def get_backend(kin, robot_name):
    """One ROS node per process: created on the first switch to "Real robot", reused afterwards."""
    global _INSTANCE
    if _INSTANCE is None:
        _INSTANCE = RealBackend(kin, robot_name)
    else:
        _INSTANCE.check_master()
    return _INSTANCE


class RealBackend:
    def __init__(self, kin, robot_name):
        try:
            import rospy
            import rosgraph
            from std_msgs.msg import Float64MultiArray, String
            from geometry_msgs.msg import PointStamped
            from sensor_msgs.msg import CompressedImage
            from rosgraph_msgs.msg import Log
        except ImportError as e:
            raise RealUnavailable("ROS Python packages not found (%s). Source the workspace setup.bash." % e)
        try:
            from alterego_msgs.msg import UpperBodyState, LowerBodyState
        except ImportError:
            raise RealUnavailable("alterego_msgs not found. Source the workspace that contains AlterEGO_v2.")
        if not robot_name:
            raise RealUnavailable("Robot name missing: export ROBOT_NAME=robot_alterego6 or pass --robot.")
        self.rospy, self.rosgraph, self.String = rospy, rosgraph, String
        self.robot = robot_name
        self.kin = kin
        self.check_master()
        if not rospy.core.is_initialized():
            rospy.init_node("person_follower_gui", anonymous=True, disable_signals=True)
        ns = "/%s/" % robot_name
        self.R = float(rospy.get_param(ns + "wheels/R_", 0.125))

        self.lock = threading.Lock()
        self.t0 = time.time()
        self.stamp = {}                                  # topic -> wall time of the last message
        self.q_r = np.zeros(5); self.q_l = np.zeros(5); self.neck = (0.0, 0.0)
        self.odo = None                                  # (x, y, yaw) of the base, from the wheels
        self._last_wheel = None; self._yaw0 = None
        self.pose2d = None; self.kp3d = None; self.face = None
        self.jpeg = None
        self.status = None
        self.events = collections.deque(maxlen=12)

        S = rospy.Subscriber
        S(ns + "alterego_state/upperbody", UpperBodyState, self.cb_upper, queue_size=1)
        S(ns + "alterego_state/lowerbody", LowerBodyState, self.cb_lower, queue_size=1)
        S(ns + "person_pose", Float64MultiArray, self._store("pose2d"), queue_size=1)
        S(ns + "person_keypoints_3d", Float64MultiArray, self._store("kp3d"), queue_size=1)
        S(ns + "person_face", Float64MultiArray, self._store("face"), queue_size=1)
        S(ns + "person_tracker/image/compressed", CompressedImage, self.cb_image, queue_size=1, buff_size=2 ** 22)
        S(ns + "person_follower/status", String, self.cb_status, queue_size=1)
        S("/rosout", Log, self.cb_log, queue_size=50)
        self.pub_cmd = rospy.Publisher(ns + "person_follower/command", String, queue_size=10)

    # ------------------------------------------------------------------------------------ connection
    def check_master(self):
        uri = os.environ.get("ROS_MASTER_URI", "http://localhost:11311")
        old = socket.getdefaulttimeout()
        socket.setdefaulttimeout(2.0)
        try:
            self.rosgraph.Master("/person_follower_gui_probe").getPid()
        except Exception:
            raise RealUnavailable("ROS master %s not reachable. Check ROS_MASTER_URI, ROS_IP and the robot WiFi." % uri)
        finally:
            socket.setdefaulttimeout(old)

    # ------------------------------------------------------------------------------------ callbacks
    def _store(self, attr):
        def cb(m):
            with self.lock:
                setattr(self, attr, list(m.data))
                self.stamp[attr] = time.time()
        return cb

    def cb_upper(self, m):
        with self.lock:
            self.q_r = np.array(list(m.right_meas_arm_shaft)[:5] + [0] * (5 - len(m.right_meas_arm_shaft[:5])), float)
            self.q_l = np.array(list(m.left_meas_arm_shaft)[:5] + [0] * (5 - len(m.left_meas_arm_shaft[:5])), float)
            self.neck = (float(m.left_meas_neck_shaft), float(m.right_meas_neck_shaft))
            self.stamp["upper"] = time.time()

    def cb_lower(self, m):
        with self.lock:
            wheel, yaw = float(m.wheels_angular_pos), float(m.yaw_angle)
            if self.odo is None:
                self.odo, self._last_wheel, self._yaw0 = [0.0, 0.0, 0.0], wheel, yaw
            ds = (wheel - self._last_wheel) * self.R
            self._last_wheel = wheel
            # The LQR tracks des_yaw with des_yaw_rate = -cmd_vel.angular.z: yaw_angle DEcreases when the base turns
            # left (angular.z > 0, ROS and ego_dance convention). The 3D view uses x forward, y left, yaw > 0 = left.
            self.odo[2] = -(yaw - self._yaw0)
            self.odo[0] += ds * math.cos(self.odo[2])
            self.odo[1] += ds * math.sin(self.odo[2])
            self.stamp["lower"] = time.time()

    def cb_image(self, m):
        with self.lock:
            self.jpeg = bytes(m.data)
            self.stamp["image"] = time.time()

    def cb_status(self, m):
        try:
            st = json.loads(m.data)
        except ValueError:
            return
        with self.lock:
            self.status = st
            self.stamp["status"] = time.time()

    def cb_log(self, m):
        if "person_follower" not in m.name or "gui" in m.name:
            return
        with self.lock:
            self.events.append("%s  %s" % (time.strftime("%H:%M:%S", time.localtime(m.header.stamp.to_sec())), m.msg))

    # ------------------------------------------------------------------------------------ GUI side
    def age(self, key):
        t = self.stamp.get(key)
        return None if t is None else time.time() - t

    def camera_jpeg(self):
        with self.lock:
            return self.jpeg if self.jpeg is not None and self.age("image") < 2.0 else None

    def reset_odometry(self):
        with self.lock:
            self.odo = None

    def control(self, c):
        a = c.get("action")
        if a in ("start", "stop", "toggle"):
            cmd = {"action": a}
        elif a == "option":
            cmd = {"option": c["name"], "value": bool(c["value"])}
        elif a == "param":
            cmd = {"param": c["name"], "value": float(c["value"])}
        elif a == "reset_odometry":
            self.reset_odometry()
            return
        else:
            raise ValueError("'%s' is not available on the real robot" % a)
        if self.age("status") is None or self.age("status") > 2.0:
            raise ValueError("person_follower is not running on %s" % self.robot)
        self.pub_cmd.publish(self.String(data=json.dumps(cmd)))

    def state(self, keep_links):
        with self.lock:
            q_r, q_l, (yaw, pitch) = self.q_r.copy(), self.q_l.copy(), self.neck
            x, y, th = self.odo if self.odo is not None else (0.0, 0.0, 0.0)
            pose2d = self.pose2d if (self.age("pose2d") or 99) < 1.0 else None
            kp3d = self.kp3d if (self.age("kp3d") or 99) < 1.0 else None
            face = self.face if (self.age("face") or 99) < 1.0 else None
            status = self.status if (self.age("status") or 99) < 2.0 else None
            events = list(self.events)[-8:]
        links = self.kin.link_poses(q_r, q_l, yaw, pitch, (x, y), th)
        T_cam = self.kin.camera_pose(x, yaw, pitch, y, th)

        # person in the world, from the 3D keypoints in the camera frame
        kw, kv = np.zeros((17, 3)), [False] * 17
        if kp3d is not None and kp3d[0] > 0.5:
            for i in range(17):
                X, Y, Z, valid = kp3d[1 + 4 * i:5 + 4 * i]
                if valid > 0.5:
                    kw[i] = (T_cam @ np.array([X, Y, Z, 1.0]))[:3]
                    kv[i] = True

        view = None
        if pose2d is not None:
            # normalized image coordinates -> the 640x480 canvas of the page
            u = [pose2d[3 + 3 * i] * 640.0 for i in range(17)]
            v = [pose2d[4 + 3 * i] * 480.0 for i in range(17)]
            vis = [pose2d[5 + 3 * i] >= 0.3 for i in range(17)]
            view = dict(u=u, v=v, vis=vis, found=pose2d[0] > 0.5, face_uv=None, face_est=False)
            if face is not None and face[0] > 0.5:
                # person_face angles -> pixel, same pinhole as the page (fx = fy = 615, center 320, 240).
                # Estimated when no face keypoint is visible (the tracker then puts it above the shoulders).
                view["face_uv"] = [320.0 - 615.0 * math.tan(face[1]), 240.0 - 615.0 * math.tan(face[2])]
                view["face_est"] = not any(vis[:5])

        d, d_ref, v_cmd, w_cmd, active = (None, None, 0.0, 0.0, None)
        options, params = {}, {}
        if status is not None:
            d, d_ref, v_cmd, active = status.get("d"), status.get("d_ref"), status.get("v", 0.0), status.get("active")
            w_cmd = status.get("w", 0.0)
            options, params = status.get("options", {}), status.get("params", {})

        def conn(key, label, limit=1.0):
            a = self.age(key)
            return dict(name=label, age=a, ok=a is not None and a < limit)

        connections = [conn("upper", "Robot state"), conn("lower", "Wheels and IMU"), conn("pose2d", "Person tracker"),
                       conn("image", "Camera image", 2.0), conn("status", "Person follower", 2.0)]
        return dict(
            t=time.time() - self.t0, active=active, d=d, d_ref=d_ref, v=v_cmd, w=w_cmd,
            yaw=math.degrees(yaw), pitch=math.degrees(pitch),
            q_r=[math.degrees(a) for a in q_r], q_l=[math.degrees(a) for a in q_l],
            base_x=x, base_y=y, base_yaw=th,
            links={l: [round(float(e), 5) for e in links[l].ravel()] for l in keep_links if l in links},
            kw=np.round(kw, 4).tolist(), kw_valid=kv, T_cam=[round(float(e), 5) for e in T_cam.ravel()],
            view=view, image=self.camera_jpeg() is not None, options=options, params=params,
            events=events, follower_ok=status is not None, connections=connections, robot=self.robot,
            master=os.environ.get("ROS_MASTER_URI", ""))
