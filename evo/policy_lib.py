"""Helpers an evolved policy may import (``from evo import policy_lib as L``).

Pure geometry, no state, no I/O. The push-line functions are the ones behind
PUSH_TO_GOAL in hen-ml/turtlebot4-glassbox (tb4_claude_nav.py), copied so a
policy runs unchanged in the simulator and on the robot. Frames: odom (x, y),
robot/base_link (x forward, y left), angles in radians.
"""
import math


def wrap_angle(a):
    """Angle wrapped to (-pi, pi]."""
    return (a + math.pi) % (2 * math.pi) - math.pi


def dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def odom_to_local(pt, ox, oy, oyaw):
    """odom point -> robot frame (x forward, y left)."""
    dx, dy = pt[0] - ox, pt[1] - oy
    return (dx * math.cos(oyaw) + dy * math.sin(oyaw),
            -dx * math.sin(oyaw) + dy * math.cos(oyaw))


def local_to_odom(pt, ox, oy, oyaw):
    """robot-frame point -> odom."""
    return (ox + pt[0] * math.cos(oyaw) - pt[1] * math.sin(oyaw),
            oy + pt[0] * math.sin(oyaw) + pt[1] * math.cos(oyaw))


def ball_odom(obs):
    """Where the ball was last seen, in odom. obs.ball is already relative to the
    current pose (the runner placed the sighting with the pose at capture), so
    this is the same odom point every tick until a new frame: a static ball stays
    put. None if the ball has never been seen."""
    if obs.ball is None:
        return None
    return local_to_odom((obs.ball["x_loc"], obs.ball["y_loc"]), obs.x, obs.y, obs.yaw)


def push_line(ball, goal):
    """Unit vector ball -> goal and that distance."""
    dx, dy = goal[0] - ball[0], goal[1] - ball[1]
    d = math.hypot(dx, dy)
    if d < 1e-6:
        return (1.0, 0.0), 0.0
    return (dx / d, dy / d), d


def push_stance(ball, goal, standoff=0.55):
    """Point `standoff` behind the ball on the goal-ball line (robot, ball, goal collinear)."""
    (ux, uy), _ = push_line(ball, goal)
    return (ball[0] - ux * standoff, ball[1] - uy * standoff)


def stance_error(robot, ball, goal):
    """(along, lateral): along > 0 means the robot is on the goal side of the ball
    (wrong side to push from); lateral is the unsigned offset from the push line."""
    (ux, uy), _ = push_line(ball, goal)
    rx, ry = robot[0] - ball[0], robot[1] - ball[1]
    return rx * ux + ry * uy, abs(-rx * uy + ry * ux)


def line_signed_offset(point, ball, goal):
    """Signed distance of `point` from the line through ball -> goal (+ = left)."""
    (ux, uy), _ = push_line(ball, goal)
    px, py = point[0] - ball[0], point[1] - ball[1]
    return -px * uy + py * ux


def heading_to(ox, oy, oyaw, target):
    """Heading error (rad) from the robot's yaw to `target` (odom point)."""
    return wrap_angle(math.atan2(target[1] - oy, target[0] - ox) - oyaw)


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def scan_bearing(i, n):
    """Robot-frame bearing of lidar beam i of n (beam 0 straight back, n/2 ahead)."""
    return wrap_angle(-math.pi + 2 * math.pi * i / n)


def scan_min(scan, bearing, half_width):
    """Closest lidar range within +-half_width of a robot-frame bearing (rad), or
    None without a scan. Walls and obstacles; the ball only if it reaches the scan plane."""
    if not scan:
        return None
    n = len(scan)
    best = None
    for i, r in enumerate(scan):
        if abs(wrap_angle(scan_bearing(i, n) - bearing)) <= half_width + 1e-9:
            best = r if best is None else min(best, r)
    return best
