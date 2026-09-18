#!/usr/bin/env python3
# Person tracker for AlterEgo. Runs on the robot base PC, where the RealSense is plugged in.
#
# Same camera access as alterego_object_detection: pyrealsense2 opens the camera directly
# (no realsense2_camera driver), depth aligned to color. YOLOv8-pose finds the person, the aligned
# depth gives the 3D position.
#
# Published topics (all under /<ROBOT_NAME>/):
#   person_pose           std_msgs/Float64MultiArray  2D pose, layout:
#                           data[0]   1 if a person was found, 0 otherwise (the rest is then all zeros)
#                           data[1:3] image width, height [px]
#                           data[3 + 3*i + 0..2] keypoint i: x, y normalized to [0, 1] (0,0 = top-left), confidence
#   person_position       geometry_msgs/PointStamped  torso center in the camera color optical frame [m]
#                           (x right, y down, z forward). Only when a person is found and the depth is valid.
#   person_keypoints_3d   std_msgs/Float64MultiArray  data[0] found, then 17 x (X, Y, Z, valid) [m], same frame
#   person_face           std_msgs/Float64MultiArray  where the face is seen from the camera, layout:
#                           [found, yaw_left, pitch_up, depth]: angles [rad] of the face from the optical axis,
#                           yaw_left > 0 = face left of the image center, pitch_up > 0 = above; depth [m], 0 = unknown.
#                           Face point = nose, else the mean of the visible eyes and ears, else estimated
#                           above the shoulders (face out of the image), with depth 0.
#   person_tracker/image/compressed  sensor_msgs/CompressedImage  annotated image (only if someone subscribes)
# Keypoint order (COCO): 0 nose, 1 left eye, 2 right eye, 3 left ear, 4 right ear, 5 left shoulder,
#   6 right shoulder, 7 left elbow, 8 right elbow, 9 left wrist, 10 right wrist, 11 left hip, 12 right hip,
#   13 left knee, 14 right knee, 15 left ankle, 16 right ankle. "Left" is the person's left.
#
# Only standard messages and no cv_bridge: the node runs in the conda Python (3.10), where the compiled
# ROS Python modules of Noetic (3.8) do not load.
import math
import os
import threading
import time

import numpy as np

N_KEYPOINTS = 17
SHOULDERS_HIPS = (5, 6, 11, 12)


# ------------------------------------------------------------------------------------------------------
# Pure functions (no ROS, no camera): tested in test/test_geometry.py
# ------------------------------------------------------------------------------------------------------
def patch_depth(depth_m, u, v, radius, min_depth, max_depth):
    """Median of the valid depths [m] in a (2r+1)^2 window around pixel (u, v). 0 if none is valid."""
    h, w = depth_m.shape
    u, v = int(round(u)), int(round(v))
    if u < 0 or v < 0 or u >= w or v >= h:
        return 0.0
    patch = depth_m[max(0, v - radius):min(h, v + radius + 1), max(0, u - radius):min(w, u + radius + 1)]
    valid = patch[(patch > min_depth) & (patch < max_depth)]
    return float(np.median(valid)) if valid.size else 0.0


def torso_pixel(kp_xy, kp_conf, box_xyxy, min_conf):
    """Torso center [px]: mean of the visible shoulders and hips, else the box center."""
    pts = [kp_xy[i] for i in SHOULDERS_HIPS if kp_conf[i] >= min_conf]
    if pts:
        p = np.mean(np.asarray(pts, dtype=float), axis=0)
        return float(p[0]), float(p[1])
    x1, y1, x2, y2 = box_xyxy
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


FACE_POINTS = (1, 2, 3, 4)   # eyes, ears


def face_pixel(kp_xy, kp_conf, min_conf):
    """Face point [px] and whether it is estimated: the nose, else the mean of the visible eyes and ears.
    Face out of the image (person close, camera low): estimated above the shoulders, 0.6 shoulder widths
    over their midpoint, so the head can look up and find it. (None, False) without face and shoulders."""
    if kp_conf[0] >= min_conf:
        return (float(kp_xy[0][0]), float(kp_xy[0][1])), False
    pts = [kp_xy[i] for i in FACE_POINTS if kp_conf[i] >= min_conf]
    if pts:
        p = np.mean(np.asarray(pts, dtype=float), axis=0)
        return (float(p[0]), float(p[1])), False
    if kp_conf[5] >= min_conf and kp_conf[6] >= min_conf:
        mid = (np.asarray(kp_xy[5], dtype=float) + np.asarray(kp_xy[6], dtype=float)) / 2
        width = float(np.linalg.norm(np.asarray(kp_xy[5], dtype=float) - np.asarray(kp_xy[6], dtype=float)))
        return (float(mid[0]), float(mid[1] - 0.6 * width)), True
    return None, False


def select_person(boxes_xyxy, torso_depths, mode):
    """Index of the person to track. mode 'largest': biggest box. 'nearest': smallest valid torso depth,
    falling back to the biggest box when no depth is valid."""
    boxes = np.asarray(boxes_xyxy, dtype=float)
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    if mode == "nearest":
        d = np.asarray(torso_depths, dtype=float)
        valid = d > 0
        if valid.any():
            idx = np.where(valid)[0]
            return int(idx[np.argmin(d[valid])])
    return int(np.argmax(areas))


def pose_message_data(found, width, height, kp_xyn=None, kp_conf=None):
    """Layout of person_pose (same as ego_dance/pose_tracker.py)."""
    data = [0.0, float(width), float(height)] + [0.0] * (3 * N_KEYPOINTS)
    if found:
        data[0] = 1.0
        for k in range(N_KEYPOINTS):
            data[3 + 3 * k] = float(kp_xyn[k][0])
            data[4 + 3 * k] = float(kp_xyn[k][1])
            data[5 + 3 * k] = float(kp_conf[k])
    return data


# ------------------------------------------------------------------------------------------------------
# Node
# ------------------------------------------------------------------------------------------------------
class PersonTracker:
    def __init__(self):
        import rospy
        from std_msgs.msg import Float64MultiArray
        from geometry_msgs.msg import PointStamped
        from sensor_msgs.msg import CompressedImage
        self.rospy = rospy
        self.Float64MultiArray, self.PointStamped, self.CompressedImage = Float64MultiArray, PointStamped, CompressedImage

        robot_name = os.environ.get("ROBOT_NAME", "")
        if not robot_name:
            rospy.logfatal("person_tracker: ROBOT_NAME is not set")
            raise SystemExit(1)
        prefix = "/" + robot_name + "/"

        p = lambda name, default: rospy.get_param("~" + name, default)
        self.rate_hz = float(p("rate", 15))
        self.conf = float(p("conf", 0.5))                 # person detection threshold
        self.kp_conf = float(p("kp_conf", 0.3))           # keypoint threshold for the 3D points
        self.imgsz = int(p("imgsz", 640))
        self.device = str(p("device", ""))                # "" = ultralytics default, "cpu", "0" = first GPU
        self.select = str(p("select", "largest"))         # largest | nearest
        self.width = int(p("width", 640))
        self.height = int(p("height", 480))
        self.fps = int(p("fps", 30))
        self.serial = str(p("serial", ""))                # RealSense serial, "" = first found
        self.depth_patch = int(p("depth_patch", 3))       # median window radius [px]
        self.min_depth = float(p("min_depth", 0.2))       # [m]
        self.max_depth = float(p("max_depth", 6.0))       # [m]
        self.frame_id = str(p("frame_id", "camera_color_optical_frame"))
        show = str(p("show_window", "auto")).lower()      # auto = only if a display is available
        self.show_window = (show == "true") or (show == "auto" and bool(os.environ.get("DISPLAY")))
        weights = str(p("weights", ""))

        self.pub_pose = rospy.Publisher(prefix + "person_pose", Float64MultiArray, queue_size=1)
        self.pub_position = rospy.Publisher(prefix + "person_position", PointStamped, queue_size=1)
        self.pub_kp3d = rospy.Publisher(prefix + "person_keypoints_3d", Float64MultiArray, queue_size=1)
        self.pub_face = rospy.Publisher(prefix + "person_face", Float64MultiArray, queue_size=1)
        self.pub_image = rospy.Publisher(prefix + "person_tracker/image/compressed", CompressedImage, queue_size=1)

        self.model = self.load_model(weights)

        self.rs = None
        self.pipeline = None
        self.align = None
        self.depth_scale = 0.001
        self.camera_ok = False
        self.consecutive_errors = 0

    # --------------------------------------------------------------------------------------- model
    def load_model(self, weights):
        from ultralytics import YOLO
        rospy = self.rospy
        if not weights:
            # Next to the package; ultralytics downloads it there on the first run (needs internet once)
            weights = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models", "yolov8n-pose.pt")
        weights = os.path.abspath(weights)
        os.makedirs(os.path.dirname(weights), exist_ok=True)
        if not os.path.isfile(weights):
            rospy.logwarn("person_tracker: %s not found, ultralytics will try to download it", weights)
        rospy.loginfo("person_tracker: loading %s", weights)
        model = YOLO(weights)
        rospy.loginfo("person_tracker: model loaded")
        return model

    # --------------------------------------------------------------------------------------- camera
    def start_camera(self, timeout_s=5.0):
        """Open the RealSense like alterego_object_detection. In a thread: pipeline.start() can hang."""
        import pyrealsense2 as rs
        self.rs = rs
        rospy = self.rospy
        result = {}

        def _start():
            try:
                pipeline = rs.pipeline()
                config = rs.config()
                if self.serial:
                    config.enable_device(self.serial)
                config.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, self.fps)
                config.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps)
                profile = pipeline.start(config)
                result["pipeline"] = pipeline
                result["scale"] = profile.get_device().first_depth_sensor().get_depth_scale()
                result["name"] = profile.get_device().get_info(rs.camera_info.name)
                result["serial"] = profile.get_device().get_info(rs.camera_info.serial_number)
            except Exception as e:  # noqa: BLE001 - report whatever librealsense raises
                result["error"] = e

        t = threading.Thread(target=_start, daemon=True)
        t.start()
        t.join(timeout=timeout_s)
        if t.is_alive():
            rospy.logerr("person_tracker: RealSense start timed out after %.0f s", timeout_s)
            return False
        if "error" in result:
            rospy.logerr("person_tracker: RealSense start failed: %s", result["error"])
            rospy.logerr("person_tracker: is the camera used by another process (realsense2_camera, "
                         "alterego_object_detection, realsense-viewer)? Only one can open it")
            return False
        self.pipeline = result["pipeline"]
        self.depth_scale = result["scale"]
        self.align = rs.align(rs.stream.color)
        self.camera_ok = True
        self.consecutive_errors = 0
        rospy.loginfo("person_tracker: %s (serial %s) started, %dx%d @ %d fps, depth scale %.5f",
                      result["name"], result["serial"], self.width, self.height, self.fps, self.depth_scale)
        return True

    def stop_camera(self):
        if self.pipeline is not None:
            try:
                self.pipeline.stop()
            except Exception:  # noqa: BLE001
                pass
        self.pipeline = None
        self.camera_ok = False

    def grab(self):
        """Newest aligned (color BGR, depth [m], intrinsics). Queued older framesets are dropped."""
        frames = self.pipeline.wait_for_frames(1000)
        while True:
            newer = self.pipeline.poll_for_frames()
            if not newer:          # empty frameset (same truth test as alterego_object_detection)
                break
            frames = newer
        aligned = self.align.process(frames)
        depth_frame = aligned.get_depth_frame()
        color_frame = aligned.get_color_frame()
        if not depth_frame or not color_frame:
            return None
        intr = color_frame.profile.as_video_stream_profile().get_intrinsics()
        color = np.asanyarray(color_frame.get_data())
        depth_m = np.asanyarray(depth_frame.get_data()).astype(np.float32) * self.depth_scale
        return color, depth_m, intr

    # --------------------------------------------------------------------------------------- processing
    def process(self, color, depth_m, intr):
        rospy = self.rospy
        h, w = color.shape[:2]
        stamp = rospy.Time.now()
        kwargs = dict(imgsz=self.imgsz, conf=self.conf, verbose=False)
        if self.device:
            kwargs["device"] = self.device
        result = self.model(color, **kwargs)[0]

        found = result.boxes is not None and len(result.boxes) > 0 and result.keypoints is not None
        kp3d = [0.0] + [0.0] * (4 * N_KEYPOINTS)
        face = [0.0, 0.0, 0.0, 0.0]
        face_uv = None
        torso_xyz = None
        text = "no person"

        if found:
            boxes = result.boxes.xyxy.cpu().numpy()
            kp_xy_all = result.keypoints.xy.cpu().numpy()      # (n, 17, 2) pixels
            kp_xyn_all = result.keypoints.xyn.cpu().numpy()    # (n, 17, 2) normalized
            kp_conf_all = result.keypoints.conf
            kp_conf_all = kp_conf_all.cpu().numpy() if kp_conf_all is not None else np.ones(kp_xy_all.shape[:2])

            torso_px = [torso_pixel(kp_xy_all[i], kp_conf_all[i], boxes[i], self.kp_conf) for i in range(len(boxes))]
            torso_d = [patch_depth(depth_m, u, v, self.depth_patch, self.min_depth, self.max_depth) for u, v in torso_px]
            i = select_person(boxes, torso_d, self.select)

            self.pub_pose.publish(self.Float64MultiArray(
                data=pose_message_data(True, w, h, kp_xyn_all[i], kp_conf_all[i])))

            # 3D keypoints
            kp3d[0] = 1.0
            for k in range(N_KEYPOINTS):
                if kp_conf_all[i][k] < self.kp_conf:
                    continue
                u, v = kp_xy_all[i][k]
                z = patch_depth(depth_m, u, v, self.depth_patch, self.min_depth, self.max_depth)
                if z > 0:
                    X, Y, Z = self.rs.rs2_deproject_pixel_to_point(intr, [float(u), float(v)], z)
                    kp3d[1 + 4 * k:5 + 4 * k] = [X, Y, Z, 1.0]

            # Face direction: ray through the face pixel (z = 1), so it works also without depth
            face_uv, estimated = face_pixel(kp_xy_all[i], kp_conf_all[i], self.kp_conf)
            if face_uv is not None:
                rx, ry, _ = self.rs.rs2_deproject_pixel_to_point(intr, [face_uv[0], face_uv[1]], 1.0)
                fd = 0.0 if estimated else patch_depth(depth_m, face_uv[0], face_uv[1], self.depth_patch,
                                                       self.min_depth, self.max_depth)
                face = [1.0, math.atan2(-rx, 1.0), math.atan2(-ry, 1.0), fd]

            # Torso position
            if torso_d[i] > 0:
                u, v = torso_px[i]
                torso_xyz = self.rs.rs2_deproject_pixel_to_point(intr, [float(u), float(v)], torso_d[i])
                msg = self.PointStamped()
                msg.header.stamp = stamp
                msg.header.frame_id = self.frame_id
                msg.point.x, msg.point.y, msg.point.z = torso_xyz
                self.pub_position.publish(msg)
                text = "person  X %+.2f  Y %+.2f  Z %.2f m" % tuple(torso_xyz)
            else:
                text = "person, no valid depth on the torso"
        else:
            self.pub_pose.publish(self.Float64MultiArray(data=pose_message_data(False, w, h)))

        self.pub_kp3d.publish(self.Float64MultiArray(data=kp3d))
        self.pub_face.publish(self.Float64MultiArray(data=face))
        self.publish_image(result, torso_px[i] if found else None, text, stamp, face_uv)
        return text

    def publish_image(self, result, torso_uv, text, stamp, face_uv=None):
        want_topic = self.pub_image.get_num_connections() > 0
        if not (want_topic or self.show_window):
            return
        import cv2
        img = result.plot()
        if torso_uv is not None:
            cv2.circle(img, (int(torso_uv[0]), int(torso_uv[1])), 6, (0, 255, 0), -1)
        if face_uv is not None:
            cv2.drawMarker(img, (int(face_uv[0]), int(face_uv[1])), (0, 255, 255), cv2.MARKER_CROSS, 20, 2)
        h, w = img.shape[:2]
        cv2.drawMarker(img, (w // 2, h // 2), (255, 255, 255), cv2.MARKER_CROSS, 14, 1)   # image center
        cv2.rectangle(img, (5, 5), (470, 35), (0, 0, 0), -1)
        cv2.putText(img, text, (10, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        if want_topic:
            ok, jpg = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 70])
            if ok:
                msg = self.CompressedImage()
                msg.header.stamp = stamp
                msg.header.frame_id = self.frame_id
                msg.format = "jpeg"
                msg.data = jpg.tobytes()
                self.pub_image.publish(msg)
        if self.show_window:
            cv2.imshow("AlterEgo person tracker", img)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                self.rospy.signal_shutdown("q pressed")

    # --------------------------------------------------------------------------------------- loop
    def run(self):
        rospy = self.rospy
        rate = rospy.Rate(self.rate_hz)
        next_retry = 0.0
        n = 0
        while not rospy.is_shutdown():
            if not self.camera_ok:
                if time.time() >= next_retry:
                    rospy.loginfo("person_tracker: opening the RealSense")
                    if not self.start_camera():
                        next_retry = time.time() + 5.0
                        rospy.logwarn("person_tracker: camera not available, retrying in 5 s")
                rate.sleep()
                continue
            try:
                frame = self.grab()
                self.consecutive_errors = 0
            except RuntimeError as e:
                self.consecutive_errors += 1
                rospy.logwarn_throttle(5, "person_tracker: frame error: %s" % e)
                if self.consecutive_errors >= 5:
                    rospy.logerr("person_tracker: camera lost, restarting the pipeline")
                    self.stop_camera()
                rate.sleep()
                continue
            if frame is None:
                rate.sleep()
                continue
            text = self.process(*frame)
            n += 1
            rospy.loginfo_throttle(5, "person_tracker: %d frames, %s" % (n, text))
            rate.sleep()

    def shutdown(self):
        self.stop_camera()
        if self.show_window:
            try:
                import cv2
                cv2.destroyAllWindows()
            except Exception:  # noqa: BLE001
                pass
        self.rospy.loginfo("person_tracker: stopped")


def main():
    import rospy
    rospy.init_node("person_tracker")
    tracker = PersonTracker()
    rospy.on_shutdown(tracker.shutdown)
    tracker.run()


if __name__ == "__main__":
    main()
