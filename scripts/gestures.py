"""Dance vocabulary V1 for person_follower: gestures recognized on the person's 3D keypoints.

No ROS here: the follower calls GestureDetector.update() once per new camera frame and reads the result.
All coordinates are in the person's body frame (rows forward, left, up of person_follower.person_frame),
relative to the shoulder midpoint, in metres. So the rules do not depend on where the camera looks.

Poses (held) and gestures:
  EXPAND     both arms open to the side at about shoulder height, elbows straight   -> continuous while held
  ATTRACT    both hands on the chest, elbows bent                                    -> continuous while held
  ARM_UP_*   one hand above the head, the other arm below the shoulder, held         -> PIROUETTE event
  FREEZE     arms crossed in front of the chest (optional, off by default)          -> continuous while held
  DROP       both hands moved down fast, from shoulder height to below it (optional) -> DROP event
The start/stop gesture (both hands above the head) stays in person_follower.
"""
import math
from collections import deque

import numpy as np

NOSE = 0
L_SH, R_SH, L_EL, R_EL, L_WR, R_WR, L_HIP, R_HIP = 5, 6, 7, 8, 9, 10, 11, 12
HEAD_ABOVE_SHOULDERS = 0.22        # [m] nose height over the shoulders when the nose depth is not valid


def _unit(v):
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-9 else None


def body_features(kp3d, frame):
    """Keypoints of interest in the body frame, relative to the shoulder midpoint. None without shoulders.
    kp3d: (17, 4) X, Y, Z, valid in the camera frame. frame: rows forward, left, up (person_frame)."""
    if frame is None or not (kp3d[L_SH, 3] and kp3d[R_SH, 3]):
        return None
    origin = (kp3d[L_SH, :3] + kp3d[R_SH, :3]) / 2
    f = {}
    for name, i in (("l_sh", L_SH), ("r_sh", R_SH), ("l_el", L_EL), ("r_el", R_EL), ("l_wr", L_WR), ("r_wr", R_WR),
                    ("nose", NOSE)):
        f[name] = frame @ (kp3d[i, :3] - origin) if kp3d[i, 3] else None
    f["head_up"] = f["nose"][2] if f["nose"] is not None else HEAD_ABOVE_SHOULDERS
    f["shoulder_width"] = float(np.linalg.norm(kp3d[L_SH, :3] - kp3d[R_SH, :3]))
    return f


def _arm(f, side):
    """(upper unit, forearm unit, wrist) for one arm of the person, or None if a point is missing."""
    sh, el, wr = f[side + "_sh"], f[side + "_el"], f[side + "_wr"]
    if sh is None or el is None or wr is None:
        return None
    u, fo = _unit(el - sh), _unit(wr - el)
    if u is None or fo is None:
        return None
    return u, fo, wr


def _angle(a, b):
    return math.degrees(math.acos(float(np.clip(np.dot(a, b), -1.0, 1.0))))


def pose_expand(f, P):
    """Both arms open to the side, about horizontal, elbows about straight."""
    s_elev = math.sin(math.radians(P("expand_elevation_tol_deg")))
    for side, out in (("l", 1.0), ("r", -1.0)):
        a = _arm(f, side)
        if a is None:
            return False
        u, fo, _ = a
        if abs(u[2]) > s_elev or abs(fo[2]) > s_elev + 0.15:           # upper arm and forearm about horizontal
            return False
        if out * u[1] < math.cos(math.radians(45)):                      # mostly to the side, outward
            return False
        if _angle(u, fo) > P("expand_elbow_max_deg"):                    # elbow about straight
            return False
    return True


def pose_attract(f, P):
    """Both hands on the chest: wrists within the torso width, between shoulder height and 0.35 m below, close to
    the chest, elbows bent, and the forearms pointing INWARD, toward the body midline. The inward forearms tell
    it apart from elbows bent with the forearms forward, a common pose while mimicking. Crossed wrists are
    FREEZE, not ATTRACT."""
    for side, inward in (("l", -1.0), ("r", 1.0)):
        a = _arm(f, side)
        if a is None:
            return False
        u, fo, wr = a
        if abs(wr[1]) > P("attract_hand_lateral_max") or not -0.35 <= wr[2] <= 0.05 or wr[0] > 0.25:
            return False
        if _angle(u, fo) < P("attract_elbow_min_deg"):
            return False
        if inward * fo[1] < P("attract_forearm_inward_min"):
            return False
    return not crossed(f)


def crossed(f):
    """Wrists on the opposite side of the body midline (left wrist on the right side and vice versa)."""
    return f["l_wr"] is not None and f["r_wr"] is not None and f["l_wr"][1] < -0.03 and f["r_wr"][1] > 0.03


def pose_freeze(f, P):
    """Arms crossed in front of the chest."""
    if not crossed(f):
        return False
    return all(-0.45 <= f[k][2] <= 0.10 and abs(f[k][1]) <= 0.35 for k in ("l_wr", "r_wr"))


def other_arm_down(f, other):
    """The other arm is below the shoulder. The camera looks up at the face, so a hanging arm often leaves the
    image at the bottom: a missing wrist counts as down unless its elbow is seen above the shoulder. Both hands
    above the head (the start / stop gesture) keep the other wrist high and visible, so they never pass."""
    wr, el = f[other + "_wr"], f[other + "_el"]
    if wr is not None:
        return wr[2] < 0.0
    return el is None or el[2] < 0.0


def pose_arm_up(f, P, side):
    """This arm's hand above the head with the upper arm raised, the other arm below the shoulder."""
    other = "r" if side == "l" else "l"
    a = _arm(f, side)
    if a is None:
        return False
    u, _, wr = a
    if wr[2] < f["head_up"] + P("pirouette_above_head"):
        return False
    if math.degrees(math.asin(float(np.clip(u[2], -1, 1)))) < 45.0:     # upper arm clearly raised
        return False
    return other_arm_down(f, other)


class GestureDetector:
    """Hold timers with a short grace for missed frames, one event per hold, cooldown after events."""

    POSES = ("EXPAND", "ATTRACT", "FREEZE", "ARM_UP_L", "ARM_UP_R")

    def __init__(self, P):
        self.P = P
        self.reset()

    def reset(self):
        self.since = {p: None for p in self.POSES}      # time the pose started
        self.last_seen = {p: -1e9 for p in self.POSES}
        self.fired = {p: False for p in self.POSES}
        self.history = deque()                          # (t, left wrist up, right wrist up) for DROP
        self.cooldown_until = -1e9
        self.freeze_until = -1e9
        self.t_last = -1e9
        self.expand = self.attract = self.freeze = False

    def hold_s(self, pose):
        return {"EXPAND": self.P("expand_hold"), "ATTRACT": self.P("attract_hold"), "FREEZE": self.P("freeze_hold"),
                "ARM_UP_L": self.P("pirouette_hold"), "ARM_UP_R": self.P("pirouette_hold")}[pose]

    def update(self, t, f, enabled):
        """f: body_features() or None. enabled: dict expand_attract, pirouette, freeze, drop.
        Returns a list of events: ("PIROUETTE", "left"|"right") or ("DROP", None). The continuous states are in
        self.expand, self.attract, self.freeze."""
        P, events = self.P, []
        self.t_last = t
        now = set()
        if f is not None:
            if enabled.get("expand_attract"):
                if pose_expand(f, P):
                    now.add("EXPAND")
                if pose_attract(f, P):
                    now.add("ATTRACT")
            if enabled.get("freeze") and pose_freeze(f, P):
                now.add("FREEZE")
            if enabled.get("pirouette"):
                if pose_arm_up(f, P, "l"):
                    now.add("ARM_UP_L")
                if pose_arm_up(f, P, "r"):
                    now.add("ARM_UP_R")
        grace = P("gesture_grace_s")
        for p in self.POSES:
            if p in now:
                if self.since[p] is None:
                    self.since[p] = t
                self.last_seen[p] = t
            elif t - self.last_seen[p] > grace:
                self.since[p] = None
                self.fired[p] = False
        held = {p: self.since[p] is not None and t - self.since[p] >= self.hold_s(p) for p in self.POSES}

        self.expand = held["EXPAND"] and not held["ATTRACT"]
        self.attract = held["ATTRACT"] and not held["EXPAND"]
        if held["FREEZE"]:
            self.freeze_until = t + P("freeze_release_s")
        self.freeze = t < self.freeze_until

        if t >= self.cooldown_until:
            for p, side in (("ARM_UP_L", "left"), ("ARM_UP_R", "right")):
                if held[p] and not self.fired[p]:
                    self.fired[p] = True
                    self.cooldown_until = t + P("gesture_cooldown_s")
                    events.append(("PIROUETTE", side))
                    break

        # DROP: both wrists from shoulder height or above to drop_amplitude below it within drop_window_s
        if enabled.get("drop") and f is not None and f["l_wr"] is not None and f["r_wr"] is not None:
            self.history.append((t, f["l_wr"][2], f["r_wr"][2]))
        while self.history and t - self.history[0][0] > P("drop_window_s"):
            self.history.popleft()
        if enabled.get("drop") and len(self.history) >= 3 and t >= self.cooldown_until:
            t0, l0, r0 = self.history[0]
            _, l1, r1 = self.history[-1]
            if min(l0, r0) >= -0.05 and l0 - l1 >= P("drop_amplitude") and r0 - r1 >= P("drop_amplitude"):
                events.append(("DROP", None))
                self.history.clear()
                self.cooldown_until = t + P("gesture_cooldown_s")
        return events

    def stale(self, t, timeout):
        """No camera frame for longer than timeout: the continuous states are not valid any more."""
        return t - self.t_last > timeout
