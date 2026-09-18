#!/usr/bin/env python3
# Closed-loop check of the base rotation with the simulator (no ROS): python3 test/test_sim_offline.py
# The person walks beyond the neck limit and then around the robot. Before the start the base must not move;
# after the start the robot turns and keeps facing the person, with the neck back near the center.
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "sim"))
import sim_core  # noqa: E402


def run(sim, seconds):
    for _ in range(int(seconds / sim.dt)):
        sim.step()


def bearing(sim):
    """Angle of the person seen from the robot base, relative to the robot heading [rad]."""
    p = sim.person.state
    a = math.atan2(p["y"] - sim.base_y, p["x"] - sim.base_x) - sim.base_yaw
    return math.atan2(math.sin(a), math.cos(a))


def go_around(sim, angle, radius=1.9, seconds=8.0):
    """Walk along a circle around the robot start point up to `angle`."""
    a0 = math.atan2(sim.person.target["y"], sim.person.target["x"])
    steps = int(seconds / 0.1)
    for i in range(1, steps + 1):
        a = a0 + (angle - a0) * i / steps
        sim.person.target.update(x=radius * math.cos(a), y=radius * math.sin(a))
        run(sim, 0.1)


# ---- idle: the head follows up to its limit, the base never turns
sim = sim_core.Sim(mode="interactive")
run(sim, 3.0)
go_around(sim, math.radians(60))
run(sim, 4.0)
n = sim.node
print("idle      person at %+4.0f deg -> base yaw %+5.1f deg, neck %+5.1f deg" %
      (math.degrees(bearing(sim) + sim.base_yaw), math.degrees(sim.base_yaw), math.degrees(n.yaw_cmd)))
assert not n.active and sim.base_yaw == 0.0 and sim.base_x == 0.0, "idle: the base must not move"
assert abs(n.yaw_cmd - n.P("head_yaw_max")) < 1e-6, "idle: neck at its limit"

# ---- following: start in front, then walk to 60 deg and around to 120 deg
sim = sim_core.Sim(mode="interactive")
run(sim, 3.0)
sim.node.apply_command({"action": "start"})
run(sim, 1.0)
assert sim.node.active
for target in (60, 120, -45):
    go_around(sim, math.radians(target))
    run(sim, 6.0)
    n = sim.node
    b = bearing(sim)
    print("following person at %+4.0f deg -> base yaw %+6.1f deg, person %+5.1f deg off the robot axis, "
          "neck %+5.1f deg, distance %.2f m (start %.2f)" % (target, math.degrees(sim.base_yaw), math.degrees(b),
                                                            math.degrees(n.yaw_cmd), n.distance(), n.d_ref))
    assert abs(b) < math.radians(15), "the robot faces the person"
    assert abs(n.yaw_cmd) < n.P("turn_start"), "the neck is back below turn_start"
    assert abs(n.distance() - n.d_ref) < n.P("dead_lin") + 0.1, "distance kept"
# ---- lost: the person crosses in front of the robot, from one side to the other, faster than it can follow
#      (same as the GUI screenshot: from (0.9, 1.7) to (0.3, -1.9) at walking speed)
sim = sim_core.Sim(mode="interactive")
run(sim, 3.0)
sim.person.target.update(r_elev=0, l_elev=0)                  # both arms forward: the robot raises its arms
sim.node.apply_command({"action": "start"})
run(sim, 2.0)
sim.person.target.update(x=0.9, y=1.7)
run(sim, 8.0)
n = sim.node
assert n.active and max(abs(n.q_r[0]), abs(n.q_l[0])) > math.radians(70), "arms raised before the loss"
sim.person.target.update(x=0.3, y=-1.9)
t_cross, t_lost = sim.t, None
for _ in range(int(14.0 / sim.dt)):
    sim.step()
    if t_lost is None and not n.active:
        t_lost = sim.t
print("lost      stop %.1f s after the crossing started, arms %.0f / %.0f deg, neck %+.0f / %+.0f deg, "
      "base %+.2f m/s %+.2f rad/s" % (
    t_lost - t_cross if t_lost else -1, math.degrees(n.q_r[0]), math.degrees(n.q_l[0]),
    math.degrees(n.yaw_cmd), math.degrees(n.pitch_cmd), sim.v, sim.w))
assert t_lost is not None and not n.active, "back to idle"
assert max(abs(x) for x in list(n.q_r) + list(n.q_l)) < math.radians(1), "arms down"
assert sim.v == 0.0 and sim.w == 0.0, "base still"
print("ALL OK")
