#!/usr/bin/env python3
"""AlterEgo person follower GUI, in the browser. Two modes, switched from the toggle at the top of the page:

  Simulation   person_follower.py runs inside this process on a simulated head camera (../sim/sim_core.py).
               You move the person from the page. Needs numpy and pyyaml only.
  Real robot   this process is a ROS node that shows the real robot, the tracker and the follower, and sends
               start / stop, options and parameters to person_follower. Needs the sourced ROS workspace,
               ROS_MASTER_URI of the robot and the robot name.

    python3 gui/web_gui.py                              # http://127.0.0.1:8765
    python3 gui/web_gui.py --robot robot_alterego6      # real-robot mode available (ROS sourced)
    python3 gui/web_gui.py --mode real --no-browser     # start directly in real-robot mode
"""
import argparse
import json
import math
import os
import sys
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "sim"))
sys.path.insert(0, HERE)
import sim_core  # noqa: E402

WEB = os.path.join(HERE, "web")
MESHES = os.path.join(WEB, "robot_meshes.json.gz")
TUNABLE = sim_core.pf.TUNABLE
OPTIONS = sim_core.pf.OPTIONS


# ================================================================================================ simulation
class SimBackend:
    def __init__(self, kin):
        self.kin = kin
        self.lock = threading.Lock()
        self.paused = False
        self.running = True           # False while the GUI shows the real robot
        self.speed = 1.0
        self.reset("interactive")
        threading.Thread(target=self.loop, daemon=True).start()

    def reset(self, mode):
        with self.lock:
            old = getattr(self, "sim", None)
            self.sim = sim_core.Sim(mode=mode, robot=self.kin)
            if old is not None and mode == "interactive" and old.mode == "interactive":
                self.sim.person.target = dict(old.person.target)
                self.sim.person.state = dict(old.person.target)
            self.mode = mode
            self.wall0, self.sim_t0 = time.monotonic(), 0.0

    def loop(self):
        while True:
            with self.lock:
                if self.paused or not self.running:
                    self.wall0, self.sim_t0 = time.monotonic(), self.sim.t
                else:
                    target = self.sim_t0 + (time.monotonic() - self.wall0) * self.speed
                    n = 0
                    while self.sim.t < target and n < 20:
                        self.sim.step()
                        n += 1
                    if self.mode == "demo" and self.sim.t >= sim_core.DEMO_LENGTH:
                        self.paused = True
            time.sleep(0.005)

    def control(self, c):
        a = c.get("action")
        if a == "mode":
            self.reset(c["mode"])
            return
        if a == "reset":
            self.reset(self.mode)
            return
        with self.lock:
            n = self.sim.node
            if a in ("start", "stop", "toggle"):
                n.apply_command({"action": a})
            elif a == "option":
                ok, msg = n.apply_command({"option": c["name"], "value": c["value"]})
                if not ok:
                    raise ValueError(msg)
            elif a == "param":
                ok, msg = n.apply_command({"param": c["name"], "value": c["value"]})
                if not ok:
                    raise ValueError(msg)
            elif a == "person" and self.mode == "interactive":
                for k, v in c.get("values", {}).items():
                    if k in self.sim.person.target:
                        self.sim.person.target[k] = float(v)
            elif a == "pause":
                self.paused = bool(c["paused"])
            elif a == "speed":
                self.speed = float(np.clip(c["speed"], 0.1, 4.0))
                self.wall0, self.sim_t0 = time.monotonic(), self.sim.t

    def state(self, keep_links):
        with self.lock:
            s = self.sim.state()
            st = self.sim.node.status()
            view = self.sim.last_view
            links = self.kin.link_poses(s["q_r"], s["q_l"], s["yaw"], s["pitch"], (s["base_x"], s["base_y"]), s["base_yaw"])
            out = dict(
                t=s["t"], active=st["active"], d=st["d"], d_ref=st["d_ref"], v=s["v"],
                yaw=math.degrees(s["yaw"]), pitch=math.degrees(s["pitch"]),
                q_r=[math.degrees(x) for x in s["q_r"]], q_l=[math.degrees(x) for x in s["q_l"]],
                base_x=s["base_x"], base_y=s["base_y"], base_yaw=s["base_yaw"], w=s["w"],
                links={l: [round(float(x), 5) for x in links[l].ravel()] for l in keep_links if l in links},
                kw=np.round(s["kw"], 4).tolist(), kw_valid=[True] * 17,
                T_cam=[round(float(x), 5) for x in s["T_cam"].ravel()],
                view=None, image=False, options=st["options"], params=st["params"],
                events=["%6.2f s  %s" % e for e in sim_core.EVENTS[-8:]], follower_ok=True,
                caption=sim_core.demo_caption(s["t"]) if self.mode == "demo" else "",
                sim=dict(mode=self.mode, paused=self.paused, speed=self.speed, person=dict(self.sim.person.target)))
            if view is not None:
                out["view"] = dict(u=np.round(view["u"], 1).tolist(), v=np.round(view["v"], 1).tolist(),
                                   vis=[bool(x) for x in view["vis"]], found=view["found"],
                                   face_uv=list(view["face_uv"]) if view["face_uv"] is not None else None,
                                   face_est=bool(view["face_est"]))
            return out


# ================================================================================================ app
class App:
    def __init__(self, robot_name):
        self.kin = sim_core.RobotKinematics()
        self.robot_name = robot_name
        self.mesh_links = None
        if os.path.exists(MESHES):
            import gzip
            with gzip.open(MESHES, "rt") as f:
                self.mesh_links = sorted({p["link"] for p in json.load(f)["parts"]})
        self.sim = SimBackend(self.kin)
        self.real = None
        self.backend = "sim"
        self.real_error = None

    def set_backend(self, which):
        if which == "real":
            try:
                import real_bridge
                self.real = real_bridge.get_backend(self.kin, self.robot_name)
            except Exception as e:   # noqa: BLE001 - shown on the page
                self.real_error = str(e)
                raise ValueError(self.real_error)
            self.real_error = None
            self.sim.running = False
            self.backend = "real"
        else:
            self.sim.running = True
            self.backend = "sim"

    def state(self):
        keep = self.mesh_links or []
        if self.backend == "real" and self.real is not None:
            s = self.real.state(keep or list(self.kin.fk.link_poses()))
        else:
            s = self.sim.state(keep or list(self.kin.fk.link_poses()))
        s["backend"] = self.backend
        s["robot_name"] = self.robot_name or ""
        s["real_error"] = self.real_error
        s["param_ranges"] = {k: list(v) for k, v in TUNABLE.items()}
        return s

    def control(self, c):
        if c.get("action") == "backend":
            self.set_backend(c["value"])
            return
        (self.real if self.backend == "real" else self.sim).control(c)

    def camera_jpeg(self):
        return self.real.camera_jpeg() if self.backend == "real" and self.real is not None else None


APP = None


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, body, ctype, extra=None):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Cache-Control", "no-store")
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = self.path.split("?")[0]
        try:
            if path in ("/", "/index.html"):
                with open(os.path.join(WEB, "index.html"), "rb") as f:
                    self._send(200, f.read(), "text/html; charset=utf-8")
            elif path == "/robot_meshes.json":
                if not os.path.exists(MESHES):
                    self._send(404, b"{}", "application/json")
                    return
                with open(MESHES, "rb") as f:
                    self._send(200, f.read(), "application/json", {"Content-Encoding": "gzip"})
            elif path == "/state":
                self._send(200, json.dumps(APP.state()).encode(), "application/json")
            elif path == "/camera.jpg":
                jpg = APP.camera_jpeg()
                if jpg is None:
                    self._send(204, b"", "image/jpeg")
                else:
                    self._send(200, jpg, "image/jpeg")
            else:
                self._send(404, b"not found", "text/plain")
        except (BrokenPipeError, ConnectionResetError):
            pass

    def do_POST(self):
        if self.path != "/control":
            self._send(404, b"not found", "text/plain")
            return
        n = int(self.headers.get("Content-Length", 0))
        try:
            APP.control(json.loads(self.rfile.read(n) or b"{}"))
            self._send(200, b'{"ok":true}', "application/json")
        except Exception as e:  # noqa: BLE001 - shown on the page
            self._send(400, json.dumps({"error": str(e)}).encode(), "application/json")


def main():
    global APP
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1", help="0.0.0.0 to open the GUI from another PC")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--robot", default=os.environ.get("ROBOT_NAME", ""), help="robot name, default $ROBOT_NAME")
    ap.add_argument("--mode", choices=("sim", "real"), default="sim", help="mode at start")
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args([a for a in sys.argv[1:] if not a.startswith("__")])   # ignore roslaunch args
    APP = App(args.robot)
    if args.mode == "real":
        try:
            APP.set_backend("real")
        except ValueError as e:
            print("Real robot not available, starting in simulation: %s" % e)
    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    url = "http://%s:%d" % ("127.0.0.1" if args.host == "0.0.0.0" else args.host, args.port)
    print("AlterEgo person follower GUI: %s  (Ctrl-C to quit)" % url)
    if not os.path.exists(MESHES):
        print("Robot meshes missing (%s): the robot is drawn as joint markers. Run sim/build_web_meshes.py" % MESHES)
    if not args.no_browser:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
