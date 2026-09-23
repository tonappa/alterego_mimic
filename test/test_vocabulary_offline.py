#!/usr/bin/env python3
# Dance vocabulary V1, offline (no ROS): python3 test/test_vocabulary_offline.py
# 1) the detector on synthetic poses (also FREEZE and DROP, off by default)
# 2) closed loop in the simulator: EXPAND, ATTRACT, PIROUETTE (both sides, too close), start/stop not confused,
#    and one minute of ordinary arm movements to count false detections
import math
import os
import sys

import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "sim"))
sys.path.insert(0, os.path.join(HERE, "..", "scripts"))
import gestures   # noqa: E402
import sim_core   # noqa: E402

cfg = yaml.safe_load(open(os.path.join(HERE, "..", "config", "follower.yaml")))
P = lambda k: cfg[k]
DOWN, T_POSE, CHEST, RAISED, UP = sim_core.DOWN_P, sim_core.T_P, sim_core.CHEST_P, sim_core.RAISED_P, sim_core.UP_P


def feats(r, l):
    """Body features of a person 1.9 m in front of a camera, arm poses as (elev, azim, elbow, twist)."""
    arms = (*sim_core.arm_dirs(*r[:3], "right", r[3]), *sim_core.arm_dirs(*l[:3], "left", l[3]))
    kw = sim_core.build_keypoints(np.array([1.9, 0]), 0.0, 0.0, arms, np.zeros(2), 0.0)
    cam = np.stack([-kw[:, 1], -kw[:, 2], kw[:, 0]], 1)
    kp3d = np.c_[cam, np.ones(17)]
    return gestures.body_features(kp3d, sim_core.pf.person_frame(kp3d))


# ------------------------------------------------------------------------------ 1) detector
assert gestures.pose_expand(feats(T_POSE, T_POSE), P)
assert not gestures.pose_expand(feats((0, 0, 0, 0), (0, 0, 0, 0)), P), "arms forward are not EXPAND"
assert not gestures.pose_expand(feats(T_POSE, DOWN), P), "one arm open is not EXPAND"
assert not gestures.pose_expand(feats((45, 88, 0, 0), (45, 88, 0, 0)), P), "arms up in a V are not EXPAND"
assert gestures.pose_attract(feats(CHEST, CHEST), P)
assert not gestures.pose_attract(feats((-90, 0, 90, 0), (-90, 0, 90, 0)), P), "elbows bent, forearms forward"
assert not gestures.pose_attract(feats(CHEST, DOWN), P), "one hand on the chest"
assert gestures.pose_arm_up(feats(RAISED, DOWN), P, "r") and gestures.pose_arm_up(feats(DOWN, RAISED), P, "l")
assert not gestures.pose_arm_up(feats(UP, UP), P, "r"), "both hands up is start/stop, not a pirouette"
assert not gestures.pose_arm_up(feats((15, 60, 0, 0), DOWN), P, "r"), "arm a bit above the shoulder is mimicked"
f = feats(RAISED, DOWN); f["l_wr"] = None
assert gestures.pose_arm_up(f, P, "r"), "hanging arm out of the image counts as down"
f = feats(RAISED, UP); f["l_wr"] = None
assert not gestures.pose_arm_up(f, P, "r"), "other elbow seen above the shoulder: not down"

# hold, grace, one event per hold, cooldown
det = gestures.GestureDetector(P)
on = dict(expand_attract=True, pirouette=True, freeze=True, drop=True)
t, events = 0.0, []
for k in range(40):                       # 2.6 s of right arm up at 15 Hz, with one missed frame
    t = k / 15.0
    events += det.update(t, None if k == 10 else feats(RAISED, DOWN), on)
assert events == [("PIROUETTE", "right")], events
for k in range(40, 60):                   # still up: no second event
    events += det.update(k / 15.0, feats(RAISED, DOWN), on)
assert len(events) == 1
det.reset(); t = 0.0
for k in range(10):                       # EXPAND needs expand_hold
    det.update(k / 15.0, feats(T_POSE, T_POSE), on)
    assert det.expand == (k / 15.0 >= P("expand_hold")), (k, det.expand)
det.reset()
for k in range(5):                        # a quick opening (0.3 s) is only mimicked
    det.update(k / 15.0, feats(T_POSE, T_POSE), on)
for k in range(5, 20):
    det.update(k / 15.0, feats(DOWN, DOWN), on)
assert not det.expand

# FREEZE: arms crossed in front of the chest (synthetic body features)
det.reset()
cross = feats(CHEST, CHEST)
cross["l_wr"] = np.array([0.15, -0.10, -0.15]); cross["r_wr"] = np.array([0.15, 0.10, -0.15])
assert gestures.pose_freeze(cross, P) and not gestures.pose_attract(cross, P)
for k in range(12):
    det.update(k / 15.0, cross, on)
assert det.freeze
for k in range(12, 20):                   # uncrossed: frozen for freeze_release_s more
    det.update(k / 15.0, feats(DOWN, DOWN), on)
assert det.freeze
for k in range(20, 40):
    det.update(k / 15.0, feats(DOWN, DOWN), on)
assert not det.freeze

# DROP: both hands from shoulder height to 0.5 m lower in 0.4 s
det.reset(); drops = []
for k in range(8):
    fk = feats(T_POSE, T_POSE)
    z = -0.08 * k
    fk["l_wr"] = fk["l_wr"] * np.array([1, 1, 0]) + np.array([0, 0, z])
    fk["r_wr"] = fk["r_wr"] * np.array([1, 1, 0]) + np.array([0, 0, z])
    drops += det.update(k / 15.0, fk, on)
assert ("DROP", None) in drops, drops
det.reset(); drops = []
for k in range(30):                       # the same lowering in 2 s is not a DROP
    fk = feats(T_POSE, T_POSE)
    z = -0.02 * k
    fk["l_wr"] = fk["l_wr"] * np.array([1, 1, 0]) + np.array([0, 0, z])
    fk["r_wr"] = fk["r_wr"] * np.array([1, 1, 0]) + np.array([0, 0, z])
    drops += det.update(k / 15.0, fk, on)
assert not drops
print("detector: EXPAND, ATTRACT, PIROUETTE, FREEZE, DROP rules, hold, grace, cooldown  OK")


# ------------------------------------------------------------------------------ 2) closed loop in the simulator
def run(sim, seconds):
    for _ in range(int(round(seconds / sim.dt))):
        sim.step()


def pose(sim, r, l):
    sim.person.target.update(r_elev=r[0], r_azim=r[1], r_elbow=r[2], r_twist=r[3],
                             l_elev=l[0], l_azim=l[1], l_elbow=l[2], l_twist=l[3])


def started(x=1.9):
    sim = sim_core.Sim(mode="interactive")
    sim.person.target.update(x=x); sim.person.state.update(x=x)
    run(sim, 3.0)
    sim.node.apply_command({"action": "start"})
    run(sim, 1.0)
    assert sim.node.active
    return sim


# EXPAND: the target distance grows while held, the robot moves away, it stays after the release
sim = started(); n = sim.node; d0, x0 = n.d_ref, sim.base_x
pose(sim, T_POSE, T_POSE); run(sim, 4.0)
pose(sim, DOWN, DOWN); run(sim, 4.0)
grow = n.d_ref - d0
print("EXPAND  4 s held -> target distance %+.2f m (%.2f -> %.2f), base moved %+.2f m" % (grow, d0, n.d_ref, sim.base_x - x0))
assert 0.3 < grow < 0.6 and sim.base_x < x0 - 0.2 and n.active

# ATTRACT: robot arms reach out while held, the distance shrinks down to attract_min_distance, then arms mimic again
sim = started(2.6); n = sim.node; d0 = n.d_ref
pose(sim, CHEST, CHEST); run(sim, 3.0)
reach_q0 = (math.degrees(n.q_r[0]), math.degrees(n.q_l[0]))
run(sim, 11.0)
print("ATTRACT arms q0 R %.0f L %.0f deg while held, target distance %.2f -> %.2f m (min %.1f)" % (
    *reach_q0, d0, n.d_ref, P("attract_min_distance")))
assert abs(reach_q0[0] - P("attract_q0_deg")) < 5 and abs(reach_q0[1] + P("attract_q0_deg")) < 5
assert abs(n.d_ref - P("attract_min_distance")) < 1e-6
pose(sim, DOWN, DOWN); run(sim, 4.0)
assert abs(math.degrees(n.q_r[0])) < 10 and abs(math.degrees(n.q_l[0])) < 10, "released: arms mimic again"


# PIROUETTE: right arm up -> turn left 360 deg, neck toward the turn, arms open; then back to mimicking
def pirouette(side):
    sim = started(); n = sim.node; yaw0 = sim.base_yaw
    pose(sim, RAISED, DOWN) if side == "right" else pose(sim, DOWN, RAISED)
    neck, spread, t0, t_end = [], [], sim.t, None
    for _ in range(int(round(16.0 / sim.dt))):
        sim.step()
        if sim.t - t0 > 3.0:
            pose(sim, DOWN, DOWN)
        if n.primitive is not None and n.primitive["phase"] == "spin":
            neck.append(n.yaw_cmd); spread.append((math.degrees(n.q_r[1]), math.degrees(n.q_l[1])))
        if t_end is None and n.primitive is None and neck:
            t_end = sim.t
    turned = sim.base_yaw - yaw0
    print("PIROUETTE %s arm up -> turned %+.0f deg, neck during the turn %+.2f rad, arms q1 %s deg, "
          "done %.1f s after the gesture, following %s" % (side, math.degrees(turned), np.median(neck),
                                                           np.round(np.median(spread, 0)), t_end - t0, n.active))
    return turned, np.median(neck), np.median(spread, 0), n
turned, neck, spread, n = pirouette("right")
assert abs(math.degrees(turned) - 360) < 10 and neck > 0.4 and spread[0] < -70 and spread[1] > 70 and n.active
turned, neck, _, n = pirouette("left")
assert abs(math.degrees(turned) + 360) < 10 and neck < -0.4 and n.active

# too close: ignored
sim = started(); n = sim.node
n.apply_command({"option": "keep_distance", "value": False})     # let the person come close
sim.person.target.update(x=1.0); run(sim, 4.0); yaw0 = sim.base_yaw
pose(sim, RAISED, DOWN); run(sim, 4.0)
assert abs(sim.base_yaw - yaw0) < 0.05 and n.primitive is None
print("PIROUETTE with the person at 1.0 m -> ignored  OK")

# both hands up = stop, not a pirouette
sim = started(); n = sim.node
pose(sim, UP, UP); run(sim, 5.0)          # gesture_lockout (3 s after the start) + gesture_hold
assert not n.active and abs(sim.base_yaw) < 0.2
print("both hands up -> STOP, no pirouette  OK")

# ordinary movements for 60 s: random arm poses within the usual mimic range, changing every 1.5 s
rng = np.random.default_rng(7)
sim = started(); n = sim.node
labels, pirouettes = set(), 0
d_start = n.d_ref
for step in range(40):
    r = (rng.uniform(-90, 15), rng.uniform(0, 80), rng.uniform(0, 100), 0)
    l = (rng.uniform(-90, 15), rng.uniform(0, 80), rng.uniform(0, 100), 0)
    pose(sim, r, l)
    for _ in range(int(round(1.5 / sim.dt))):
        sim.step()
        lab = n.gesture_label()
        if lab:
            labels.add(lab.split()[0])
        if n.primitive is not None and n.primitive["name"] == "PIROUETTE" and n.primitive["phase"] == "prepare":
            pirouettes += 1 if n.primitive.get("counted") is None else 0
            n.primitive["counted"] = True
print("60 s of ordinary movements -> pirouettes %d, gestures seen %s, target distance change %+.2f m" % (
    pirouettes, sorted(labels) or "none", n.d_ref - d_start))
assert pirouettes == 0 and abs(n.d_ref - d_start) < 0.25
print("ALL OK")
