"""RUN_POLICY: drive the robot with a controller file written against evo.policy_api.

A policy file defines make_policy() (or class Policy) with reset(goal) and
act(obs) -> (v, w) or (v, w, done). The same file runs in a simulator, so the
observation here is built to match it exactly:

    obs.x, obs.y, obs.yaw   odometry
    obs.ball                latest 'ball' detection, as the simulator defines it:
                            {x_loc, y_loc} = the ball's CENTRE in the robot frame,
                            dist = range from the camera to that centre, age = s
                            since it arrived. The camera reports the visible surface
                            relative to the lens, so we add the ball radius along
                            the ray and the camera's offset (TB4_BALL_R, TB4_CAM_X).
                            It stays after the ball leaves view, with age growing;
                            None until first seen.
    obs.goal                the ball goal (marked in the UI, or given)
    obs.bump                a bump hazard message arrived since the last step (the
                            navigator keeps its hazard list for ~2 s for the UI, so
                            only fresh messages count)
    obs.v, obs.w            measured from odometry over the last step
    obs.scan, obs.scan_age  the RPLIDAR scan as LIDAR_BEAMS ranges in the robot frame
                            (beam i at bearing -pi + 2*pi*i/n, beam 0 straight back),
                            invalid or empty = LIDAR_MAX, as the simulator gives it

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
HAZARD_FRESH = 0.25    # s: a hazard message older than this is not "pressed now"
# Must match the simulator (sensing.cam_x, physics.ball_radius); set the measured
# values on the robot with these environment variables.
CAM_X = float(os.environ.get("TB4_CAM_X", "0.08"))    # m, camera ahead of the base centre
BALL_R = float(os.environ.get("TB4_BALL_R", "0.11"))  # m, ball radius


LIDAR_BEAMS = 36           # must match the simulator (sensing.lidar_beams, lidar_max, lidar_min)
LIDAR_MAX = 6.0
LIDAR_MIN = 0.15
LIDAR_WINDOW = math.radians(2.5)   # samples within this of a beam's bearing make up the beam


def scan_beams(scan, n=None, rmax=None, rmin=None):
    """A sensor_msgs/LaserScan from the navigator -> n ranges in the robot frame.

    The RPLIDAR is mounted yaw +90 deg from base_link (tb4_claude_nav build_grid:
    base_x = -r*sin(a), base_y = r*cos(a)), so base bearing b is lidar angle
    atan2(-cos b, sin b). Each beam is the median of the valid samples within
    LIDAR_WINDOW of it; none valid -> rmax (the simulator's 'nothing seen')."""
    n = LIDAR_BEAMS if n is None else n
    rmax = LIDAR_MAX if rmax is None else rmax
    rmin = LIDAR_MIN if rmin is None else rmin
    rng = list(scan.ranges)
    inc, a0 = float(scan.angle_increment), float(scan.angle_min)
    lo = max(rmin, float(getattr(scan, "range_min", 0.0) or 0.0))
    out = []
    for i in range(n):
        b = -math.pi + 2 * math.pi * i / n
        target = math.atan2(-math.cos(b), math.sin(b))
        vals = []
        k0 = int(round((target - a0) / inc)) if inc else 0
        span = int(math.ceil(LIDAR_WINDOW / abs(inc))) if inc else 0
        for k in range(k0 - span, k0 + span + 1):
            j = k % len(rng)
            ang = a0 + k * inc
            if abs(_wrap(ang - target)) > LIDAR_WINDOW + 1e-9:
                continue
            r = rng[j]
            if r is not None and math.isfinite(r) and lo <= r <= rmax:
                vals.append(float(r))
        if vals:
            vals.sort()
            m = len(vals) // 2
            out.append(round(vals[m] if len(vals) % 2 else 0.5 * (vals[m - 1] + vals[m]), 3))
        else:
            out.append(rmax)
    return tuple(out)


def ball_in_robot_frame(det, cam_x=None, ball_r=None):
    """A navigator detection -> the policy API's ball.

    The navigator gives x_loc/y_loc/dist relative to the camera, and its `dist`
    is the distance to the ball's visible SURFACE (stereo depth over the box).
    When it came from the p25 depth image it is depth along the optical axis,
    not range. The simulator, and the policy, use the ball's CENTRE in the robot
    frame and the range from the camera to it."""
    cam_x = CAM_X if cam_x is None else cam_x
    ball_r = BALL_R if ball_r is None else ball_r
    x, y = float(det["x_loc"]), float(det["y_loc"])
    brg = math.atan2(y, x)
    surf = float(det.get("dist", math.hypot(x, y)))
    p25 = det.get("p25_dist")
    if p25 is not None and abs(surf - float(p25)) < 1e-9:
        surf = surf / max(math.cos(brg), 0.5)          # depth along the axis -> range
    d = surf + ball_r
    return dict(x_loc=cam_x + d * math.cos(brg), y_loc=d * math.sin(brg), dist=d)
LOG_DIR = os.path.expanduser("~/policy_runs")


def _wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def goal_from_start(rel, ox, oy, oyaw):
    """(forward, left) of the robot's pose now -> an odom point."""
    f, l = float(rel[0]), float(rel[1])
    return (ox + f * math.cos(oyaw) - l * math.sin(oyaw), oy + f * math.sin(oyaw) + l * math.cos(oyaw))


def run(path, get, set_, node, live_det, check, goal=None, max_time=90.0, name="ball",
        log_dir=LOG_DIR, sleep=time.sleep, clock=time.time, goal_rel=None):
    """Returns 'done' / 'settled' / 'timeout' / 'blocked' / 'error: ...' / 'no-goal' / 'no-robot'.
    goal_rel = (forward, left) metres from the robot's pose at the start, instead of
    an odom goal: a study layout's goal measured from its taped start pose."""
    if node() is None:
        return "no-robot"
    if goal_rel is not None:
        goal = goal_from_start(goal_rel, float(get("odom_x")), float(get("odom_y")), float(get("odom_yaw")))
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
    logf.write(json.dumps(dict(kind="start", policy=path, goal=goal, goal_rel=goal_rel, t=clock(),
                               cam_x=CAM_X, ball_r=BALL_R)) + "\n")

    t0 = clock()
    last_det, last_obj, last_raw, arrived = None, None, None, None
    v_sent = w_sent = 0.0
    v_meas = w_meas = 0.0
    prev_od = None
    scan, last_scan_t = None, None
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
                last_det = ball_in_robot_frame(det)
                last_raw = dict(x_loc=round(float(det["x_loc"]), 3), y_loc=round(float(det["y_loc"]), 3),
                                dist=round(float(det.get("dist", 0.0)), 3))
                arrived = now
            ball = None
            if last_det is not None:
                ball = dict(last_det, age=round(now - arrived, 3))
            ox, oy, oyaw = float(get("odom_x")), float(get("odom_y")), float(get("odom_yaw"))
            t_od = get("odom_t") or now
            if prev_od is not None and t_od - prev_od[3] > 1e-3:        # a new odometry sample
                dt_od = t_od - prev_od[3]
                hd = prev_od[2] + 0.5 * _wrap(oyaw - prev_od[2])
                v_meas = ((ox - prev_od[0]) * math.cos(hd) + (oy - prev_od[1]) * math.sin(hd)) / dt_od
                w_meas = _wrap(oyaw - prev_od[2]) / dt_od
            if prev_od is None or t_od - prev_od[3] > 1e-3:
                prev_od = (ox, oy, oyaw, t_od)
            fresh_bump = "bump" in kinds and now - float(get("last_hazard_t") or 0.0) <= HAZARD_FRESH
            sc, sc_t = get("scan"), get("scan_t")
            if sc is not None and sc_t != last_scan_t:
                scan, last_scan_t = scan_beams(sc), sc_t
            scan_age = round(now - float(last_scan_t), 3) if scan is not None else 0.0
            obs = Obs(t=round(t, 3), x=ox, y=oy, yaw=oyaw, ball=ball, goal=goal,
                      bump=fresh_bump, v=v_meas, w=w_meas, scan=scan, scan_age=scan_age)
            try:
                v, w, done = clamp_cmd(policy.act(obs), V_MAX, W_MAX)
            except Exception as e:  # noqa: BLE001
                result = f"error: {type(e).__name__}: {e}"
                break
            logf.write(json.dumps(dict(t=round(t, 3), x=round(ox, 3), y=round(oy, 3), yaw=round(oyaw, 3),
                                       ball=ball, raw=last_raw, bump=obs.bump, v=round(v, 3), w=round(w, 3),
                                       v_meas=round(v_meas, 3), w_meas=round(w_meas, 3),
                                       scan=scan if scan is not None and scan_age < 0.05 else None,
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
