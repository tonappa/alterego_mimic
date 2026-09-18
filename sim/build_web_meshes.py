#!/usr/bin/env python3
"""One-off: low-poly meshes of the AlterEgo v2 URDF for the web GUI -> ../gui/web/robot_meshes.json.gz.
Needs yourdfpy, trimesh, fast_simplification, pycollada and the alterego_description meshes
(ALTEREGO_DESCRIPTION or rospack). The file is shipped with the package: rerun only if the meshes change."""
import gzip
import json
import os

import numpy as np

import robot_model

HERE = os.path.dirname(os.path.abspath(__file__))


def main(out=os.path.join(os.path.dirname(HERE), "gui", "web", "robot_meshes.json.gz")):
    r = robot_model.Robot()
    parts = []
    for link, V, F, c in r.parts:
        parts.append(dict(link=link, color=[round(float(x), 3) for x in c],
                          v=[round(float(x), 4) for x in np.asarray(V).ravel()],
                          f=[int(x) for x in np.asarray(F).ravel()]))
    with gzip.open(out, "wt") as f:
        json.dump(dict(parts=parts), f, separators=(",", ":"))
    print("%d parts, %d faces -> %s (%.0f kB)" % (len(parts), r.n_faces, out, os.path.getsize(out) / 1024))


if __name__ == "__main__":
    main()
