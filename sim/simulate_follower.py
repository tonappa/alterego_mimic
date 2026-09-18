#!/usr/bin/env python3
"""Scripted demo of person_follower.py (the video): runs sim_core.Sim in demo mode, saves frames.pkl for
render_video.py. The real follower code runs on a simulated head camera; see sim_core.py."""
import os
import pickle

import numpy as np

import sim_core

HERE = os.path.dirname(os.path.abspath(__file__))


def main(fps=12.5, frame_file=os.path.join(HERE, "frames.pkl")):
    sim = sim_core.Sim(mode="demo")
    every = int(round(1.0 / (fps * sim.dt)))
    frames = []
    k = 0
    while sim.t < sim_core.DEMO_LENGTH:
        if k % every == 0:
            s = sim.state()
            s["view"] = sim_core.observe(s["kw"], s["T_cam"], np.random.default_rng(k))
            s["caption"] = sim_core.demo_caption(s["t"])
            frames.append(s)
        sim.step()
        k += 1
    with open(frame_file, "wb") as f:
        pickle.dump(dict(frames=frames, events=list(sim_core.EVENTS), fps=fps, W=sim_core.W, H=sim_core.H), f)
    print("frames", len(frames))
    for t, msg in sim_core.EVENTS:
        if "STARTED" in msg or "STOPPED" in msg:
            print("  %.2f s  %s" % (t, msg))


if __name__ == "__main__":
    main()
