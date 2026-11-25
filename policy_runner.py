"""RUN_POLICY: drive the robot with a controller file written against evo.policy_api.

A policy file defines make_policy() (or class Policy) with reset(goal) and
act(obs) -> (v, w) or (v, w, done). The same file runs in a simulator, so the
observation here is built to match it exactly:

    obs.x, obs.y, obs.yaw   odometry
    obs.ball                latest 'ball' detection in the robot frame
                            {x_loc, y_loc, dist, age}; it stays after the ball
                            leaves view, with age growing; None until first seen
    obs.goal                the ball goal (marked in the UI, or given)
    obs.bump                front bumper pressed since the last step
    obs.v, obs.w            the last command actually sent (no wheel odometry twist here)

Commands are clamped to |v| <= 0.30 m/s and |w| <= 1.0 rad/s. The run ends when
the policy returns done, after `max_time`, when the robot has been commanded
still for 3 s (same rule as the simulator), or on cliff / wheel drop. Every tick
is logged to ~/policy_runs/<time>_<policy>.jsonl for replay and analysis.

This module has no ROS imports: NavRobot.run_policy passes in the accessors.
"""
from __future__ import annotations

import json
import math
import os
import time

from evo.policy_api import Obs, clamp_cmd, load_policy

V_MAX, W_MAX = 0.30, 1.0
TICK = 0.10            # s, 10 Hz like the simulator
SETTLE = 3.0           # s commanded still -> end
LOG_DIR = os.path.expanduser("~/policy_runs")


def run(path, get, set_, node, live_det, check, goal=None, max_time=90.0, name="ball",
        log_dir=LOG_DIR, sleep=time.sleep, clock=time.time):
    """Returns 'done' / 'settled' / 'timeout' / 'blocked' / 'error: ...' / 'no-goal' / 'no-robot'."""
    if node() is None:
        return "no-robot"
    if goal is None or goal[0] is None:
        g = get("ball_goal")
        goal = (float(g[0]), float(g[1])) if g else None
    if goal is None:
        set_(status="no ball goal set: tick “Ball goal” and click the BEV")
        return "no-goal"
    goal = (float(goal[0]), float(goal[1]))
    try:
        policy = load_policy(path)
        policy.reset(goal)
    except Exception as e:  # noqa: BLE001
        set_(status=f"policy rejected: {e}")
        return f"error: {e}"

    os.makedirs(log_dir, exist_ok=True)
    tag = os.path.basename(path).removesuffix(".py")
    logf = open(os.path.join(log_dir, time.strftime("%Y%m%d_%H%M%S_") + tag + ".jsonl"), "w")
    logf.write(json.dumps(dict(kind="start", policy=path, goal=goal, t=clock())) + "\n")

    t0 = clock()
    last_det, last_obj, arrived = None, None, None
    v_sent = w_sent = 0.0
    still_since = None
    result = "timeout"
    set_(ball_goal=goal, goal_mode="push")           # a bump here is the ball, not a wall
    try:
        while True:
            check()                                  # Stop button -> StopProgram
            now = clock()
            t = now - t0
            if t >= max_time:
                result = "timeout"
                break
            kinds = [h.get("type") for h in (get("hazards") or [])]
            if "cliff" in kinds or "wheel_drop" in kinds:
                result = "blocked"
                break
            det = live_det(name)
            if det is not None and det is not last_obj:      # a new frame saw the ball
                last_obj = det
                last_det = dict(x_loc=float(det["x_loc"]), y_loc=float(det["y_loc"]),
                                dist=float(det.get("dist", math.hypot(det["x_loc"], det["y_loc"]))))
                arrived = now
            ball = None
            if last_det is not None:
                ball = dict(last_det, age=round(now - arrived, 3))
            ox, oy, oyaw = float(get("odom_x")), float(get("odom_y")), float(get("odom_yaw"))
            obs = Obs(t=round(t, 3), x=ox, y=oy, yaw=oyaw, ball=ball, goal=goal,
                      bump="bump" in kinds, v=v_sent, w=w_sent)
            try:
                v, w, done = clamp_cmd(policy.act(obs), V_MAX, W_MAX)
            except Exception as e:  # noqa: BLE001
                result = f"error: {type(e).__name__}: {e}"
                break
            logf.write(json.dumps(dict(t=round(t, 3), x=round(ox, 3), y=round(oy, 3), yaw=round(oyaw, 3),
                                       ball=ball, bump=obs.bump, v=round(v, 3), w=round(w, 3),
                                       done=done)) + "\n")
            if done:
                result = "done"
                break
            if abs(v) < 0.01 and abs(w) < 0.02:
                still_since = now if still_since is None else still_since
                if now - still_since >= SETTLE:
                    result = "settled"
                    break
            else:
                still_since = None
            node().send_twist(v, w)
            v_sent, w_sent = v, w
            if ball is not None:
                bx = ox + ball["x_loc"] * math.cos(oyaw) - ball["y_loc"] * math.sin(oyaw)
                by = oy + ball["x_loc"] * math.sin(oyaw) + ball["y_loc"] * math.cos(oyaw)
                set_(ball_mark=(bx, by))
            set_(status=f"policy {tag}: v={v:+.2f} w={w:+.2f}"
                        + (f" ball {ball['dist']:.2f} m (age {ball['age']:.1f}s)" if ball else " ball not seen"))
            sleep(max(0.0, TICK - (clock() - now)))
    finally:
        n = node()
        if n is not None:
            n.send_twist(0.0, 0.0)
        set_(goal_mode="move")
        logf.write(json.dumps(dict(kind="end", result=result, t=round(clock() - t0, 3))) + "\n")
        logf.close()
    set_(status=f"policy {tag}: {result}")
    return result
