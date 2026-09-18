"""Minimal URDF forward kinematics with numpy only (fixed, revolute, continuous, prismatic joints)."""
import math
import xml.etree.ElementTree as ET

import numpy as np


def _rpy(r, p, y):
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp, cp * sr, cp * cr]])


def _axis_angle(axis, q):
    a = np.asarray(axis, float)
    a = a / np.linalg.norm(a)
    x, y, z = a
    c, s, C = math.cos(q), math.sin(q), 1 - math.cos(q)
    return np.array([[c + x * x * C, x * y * C - z * s, x * z * C + y * s],
                     [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
                     [z * x * C - y * s, z * y * C + x * s, c + z * z * C]])


class URDFKinematics:
    def __init__(self, path):
        root = ET.parse(path).getroot()
        self.joints = {}
        children = set()
        for j in root.findall("joint"):
            o = j.find("origin")
            xyz = [float(v) for v in (o.get("xyz", "0 0 0").split() if o is not None else [0, 0, 0])]
            rpy = [float(v) for v in (o.get("rpy", "0 0 0").split() if o is not None else [0, 0, 0])]
            ax = j.find("axis")
            axis = [float(v) for v in ax.get("xyz").split()] if ax is not None else [1, 0, 0]
            T = np.eye(4)
            T[:3, :3] = _rpy(*rpy)
            T[:3, 3] = xyz
            parent, child = j.find("parent").get("link"), j.find("child").get("link")
            self.joints[j.get("name")] = dict(type=j.get("type"), origin=T, axis=axis, parent=parent, child=child)
            children.add(child)
        self.links = [l.get("name") for l in root.findall("link")]
        self.child_joints = {}
        for name, j in self.joints.items():
            self.child_joints.setdefault(j["parent"], []).append(name)
        roots = [l for l in self.links if l not in children]
        self.root = "world" if "world" in roots else roots[0]

    def link_poses(self, q=None):
        """World (root) transform of every link reachable from the root. q: {joint name: value}."""
        q = q or {}
        out = {self.root: np.eye(4)}
        stack = [self.root]
        while stack:
            parent = stack.pop()
            for jn in self.child_joints.get(parent, []):
                j = self.joints[jn]
                if j["child"] in out:            # URDFs with two parents (world_to_base): first one wins
                    continue
                M = np.eye(4)
                v = float(q.get(jn, 0.0))
                if j["type"] in ("revolute", "continuous"):
                    M[:3, :3] = _axis_angle(j["axis"], v)
                elif j["type"] == "prismatic":
                    M[:3, 3] = np.asarray(j["axis"], float) * v
                out[j["child"]] = out[parent] @ j["origin"] @ M
                stack.append(j["child"])
        return out
