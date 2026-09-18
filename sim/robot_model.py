"""AlterEgo v2 URDF (utils/ego_dance/urdf/ego_robot_gazebo_v2.urdf) for rendering: meshes per link, decimated.
The neck is rotated as head_inv_kin does it: yaw (left cube) about the vertical axis, then pitch (right cube),
about the neck pivot (see the note in the answer: the URDF neck joints are both pitch axes)."""
import os
import re
import numpy as np
import trimesh
import yourdfpy

HERE = os.path.dirname(os.path.abspath(__file__))
SRC_URDF = os.path.join(HERE, "alterego_v2.urdf")      # copy of the expanded AlterEgo v2 URDF


def find_description():
    """alterego_description folder: $ALTEREGO_DESCRIPTION, else rospack, else next to this package."""
    d = os.environ.get("ALTEREGO_DESCRIPTION")
    if d and os.path.isdir(d):
        return d
    try:
        import subprocess
        d = subprocess.check_output(["rospack", "find", "alterego_description"], text=True).strip()
        if os.path.isdir(d):
            return d
    except Exception:
        pass
    for up in ("../..", "../../..", "../../../.."):
        cand = os.path.normpath(os.path.join(HERE, up, "alterego_description"))
        if os.path.isdir(cand):
            return cand
    raise SystemExit("alterego_description not found: export ALTEREGO_DESCRIPTION=<AlterEGO_v2>/alterego_description")


def prepared_urdf():
    """URDF with absolute mesh paths and without the gazebo tags, written to /tmp."""
    s = open(SRC_URDF).read().replace("package://alterego_description/", find_description().rstrip("/") + "/")
    s = re.sub(r"<gazebo.*?</gazebo>", "", s, flags=re.S)
    s = re.sub(r"<xacro:include[^>]*/>", "", s)
    out = "/tmp/alterego_v2_sim.urdf"
    open(out, "w").write(s)
    return out
HEAD_LINKS = {"neck", "neck_cube", "head", "camera", "camera_left", "camera_right", "realsense", "fake_realsense"}
NECK_PIVOT = np.array([0.0, 0.0, 0.7535])          # base_to_neck origin, in the base frame
RIGHT = ["base_to_right_shoulder_flange", "right_shoulder_cube_to_right_arm_flange",
         "right_arm_cube_to_right_elbow_flange", "right_elbow_cube_to_right_forearm_flange",
         "right_forearm_cube_to_right_wrist_flange"]
LEFT = [j.replace("right", "left") for j in RIGHT]


def rot_head(yaw, pitch):
    cy, sy, cp, sp = np.cos(yaw), np.sin(yaw), np.cos(pitch), np.sin(pitch)
    R = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]]) @ np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    T = np.eye(4); T[:3, :3] = R
    P = np.eye(4); P[:3, 3] = NECK_PIVOT
    Pi = np.eye(4); Pi[:3, 3] = -NECK_PIVOT
    return P @ T @ Pi


class Robot:
    def __init__(self, face_ratio=0.04, min_faces=250, max_faces=2500):
        self.urdf = yourdfpy.URDF.load(prepared_urdf(), load_meshes=True, build_scene_graph=True)
        self.zero = {j: 0.0 for j in self.urdf.actuated_joint_names}
        self.urdf.update_cfg(self.zero)
        self.T_world_base0 = self.urdf.get_transform("base", "world")
        # meshes in their link frame
        self.parts = []   # (link, vertices (n,3), faces (m,3), color)
        scene = self.urdf.scene
        for node in scene.graph.nodes_geometry:
            T_world_node, geom_name = scene.graph[node]
            mesh = scene.geometry[geom_name]
            if not isinstance(mesh, trimesh.Trimesh) or len(mesh.faces) == 0:
                continue
            link = self._link_of(node)
            if link is None or link in ("world", "trans_x", "trans_y", "trans_z", "rot_z", "rot_y", "laser", "imu",
                                        "left_hand_ik", "right_hand_ik"):
                continue
            m = self._lowpoly(mesh, face_ratio, min_faces, max_faces)
            T_link = self.urdf.get_transform(link, "world")
            m.apply_transform(np.linalg.inv(T_link) @ T_world_node)      # into the link frame
            self.parts.append((link, np.asarray(m.vertices), np.asarray(m.faces), self._color(link)))
        self.n_faces = sum(len(p[2]) for p in self.parts)
        self.T_cam0 = np.linalg.inv(self.T_world_base0) @ self.urdf.get_transform("fake_realsense", "world")

    @staticmethod
    def _lowpoly(mesh, face_ratio, min_faces, max_faces):
        """CAD meshes are not manifold: decimating them directly tears them apart. Voxel remesh first
        (clean closed surface), then quadric decimation."""
        target = int(np.clip(len(mesh.faces) * face_ratio, min_faces, max_faces))
        if len(mesh.faces) <= target:
            return mesh.copy()
        pitch = max(float(max(mesh.extents)) / 45.0, 0.003)
        try:
            v = mesh.voxelized(pitch=pitch).fill()
            m = v.marching_cubes
            m.apply_transform(v.transform)
            if len(m.faces) > target:
                m = m.simplify_quadric_decimation(face_count=target)
            return m
        except Exception:
            return mesh.convex_hull

    def _link_of(self, node):
        # yourdfpy names the geometry nodes after the link: walk up the scene graph to a link name
        g = self.urdf.scene.graph
        cur = node
        for _ in range(10):
            if cur in self.urdf.link_map:
                return cur
            parents = [e[0] for e in g.to_edgelist() if e[1] == cur]
            if not parents:
                return None
            cur = parents[0]
        return None

    @staticmethod
    def _color(link):
        if link in ("wheel_L", "wheel_R"):
            return np.array([0.15, 0.15, 0.15])
        if "hand" in link:
            return np.array([0.85, 0.85, 0.85])
        if link in HEAD_LINKS:
            return np.array([0.25, 0.25, 0.3])
        if "flange" in link:
            return np.array([0.2, 0.4, 0.8])
        if "cube" in link:
            return np.array([0.35, 0.35, 0.38])
        return np.array([0.9, 0.9, 0.92])

    def link_poses(self, q_r, q_l, yaw, pitch, base_xy, base_yaw=0.0):
        """World transform of every link. Base at base_xy on the floor, facing +x."""
        cfg = dict(self.zero)
        for i in range(5):
            cfg[RIGHT[i]] = float(q_r[i]); cfg[LEFT[i]] = float(q_l[i])
        self.urdf.update_cfg(cfg)
        c, sn = np.cos(base_yaw), np.sin(base_yaw)
        Tz = np.eye(4); Tz[:2, :2] = [[c, -sn], [sn, c]]; Tz[0, 3], Tz[1, 3] = base_xy
        Tb = Tz @ self.T_world_base0
        Tb0_inv = np.linalg.inv(self.T_world_base0)
        H = rot_head(yaw, pitch)
        poses = {}
        for link in {p[0] for p in self.parts} | {"fake_realsense"}:
            T_base_link = Tb0_inv @ self.urdf.get_transform(link, "world")
            if link in HEAD_LINKS:
                T_base_link = H @ T_base_link
            poses[link] = Tb @ T_base_link
        return poses

    def triangles(self, poses):
        tris, cols = [], []
        for link, V, F, c in self.parts:
            T = poses[link]
            W = V @ T[:3, :3].T + T[:3, 3]
            tris.append(W[F]); cols.append(np.repeat(c[None], len(F), 0))
        return np.concatenate(tris), np.concatenate(cols)
