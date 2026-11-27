"""Put each camera sighting on the map once, with the pose the robot had when
the frame was taken.

The OAK-D runs at ~2 fps and its detections reach the navigator ~0.3 s after
the frame was captured. Placing a detection with the robot's pose *now* makes a
static ball slide around the map whenever the robot moves: by the pipeline
delay on every frame, and by the whole time since the last frame if the
detection is re-used until the next one arrives. So the navigator keeps a short
odometry history, places each sighting with the pose at its capture time, and
from then on that odom point stays put; consumers re-express it in the robot's
current frame when they need it relative to the robot.

Capture time comes from the detection message's header stamp (depthai_ros_driver
stamps it from the device's frame timestamp, on the same clock as time.time()
on the Pi). If the stamp is missing or implausible, a fixed delay is assumed
(TB4_CAM_LAG). No ROS imports: tested offline.
"""
from __future__ import annotations

import bisect
import math
import os
from collections import deque

DEFAULT_LAG = float(os.environ.get("TB4_CAM_LAG", "0.35"))   # s, assumed capture->arrival delay
MAX_LAG = 1.5             # s: a stamp further behind its arrival than this is not believed
MAX_AHEAD = 0.05          # s: nor one ahead of it by more than clock jitter


def _wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def local_to_odom(pt, pose):
    """Robot-frame point (x forward, y left) at `pose` (x, y, yaw) -> odom point."""
    x, y, yaw = pose
    c, s = math.cos(yaw), math.sin(yaw)
    return (x + pt[0] * c - pt[1] * s, y + pt[0] * s + pt[1] * c)


def odom_to_local(pt, pose):
    """Odom point -> robot frame at `pose`."""
    x, y, yaw = pose
    dx, dy = pt[0] - x, pt[1] - y
    c, s = math.cos(yaw), math.sin(yaw)
    return (dx * c + dy * s, -dx * s + dy * c)


def capture_time(stamp, arrival, default_lag=None, max_lag=MAX_LAG):
    """(t_capture, source): the header stamp when it is plausible ('stamp'),
    else arrival minus the assumed delay ('assumed')."""
    default_lag = DEFAULT_LAG if default_lag is None else default_lag
    if stamp is not None:
        try:
            stamp = float(stamp)
        except (TypeError, ValueError):
            stamp = None
    if stamp is not None and stamp > 0 and -MAX_AHEAD <= arrival - stamp <= max_lag:
        return min(stamp, arrival), "stamp"
    return arrival - default_lag, "assumed"


class PoseHistory:
    """The last `maxlen` odometry samples; pose at any recent time, interpolated."""

    def __init__(self, maxlen=600):           # ~30 s of 20 Hz odometry
        self._q = deque(maxlen=maxlen)

    def add(self, t, x, y, yaw):
        s = (float(t), float(x), float(y), float(yaw))
        if self._q and s[0] <= self._q[-1][0]:
            if s[0] == self._q[-1][0]:
                self._q[-1] = s              # same instant: keep the latest reading
            return                           # out of order: ignore
        self._q.append(s)

    def __len__(self):
        return len(self._q)

    def at(self, t):
        """(x, y, yaw) at time t; clamped to the oldest/newest sample; None if empty."""
        q = list(self._q)
        if not q:
            return None
        if t <= q[0][0]:
            return q[0][1:]
        if t >= q[-1][0]:
            return q[-1][1:]
        i = bisect.bisect_left([p[0] for p in q], t)
        t0, x0, y0, a0 = q[i - 1]
        t1, x1, y1, a1 = q[i]
        f = (t - t0) / (t1 - t0)
        return (x0 + f * (x1 - x0), y0 + f * (y1 - y0), _wrap(a0 + f * _wrap(a1 - a0)))


def place(dets, pose_cap, t_cap, cap_src):
    """Annotate detections (dicts with x_loc/y_loc in the robot frame at capture)
    in place with where they were seen: t_cap, cap_src, pose_cap and `odom`, the
    sighting's odom point. Returns dets."""
    pc = [round(float(v), 4) for v in pose_cap]
    for d in dets:
        if d.get("x_loc") is None or d.get("y_loc") is None:
            continue
        ox, oy = local_to_odom((d["x_loc"], d["y_loc"]), pose_cap)
        d.update(t_cap=round(float(t_cap), 3), cap_src=cap_src, pose_cap=pc,
                 odom=[round(ox, 4), round(oy, 4)])
    return dets


def dets_now(dets, pose):
    """The detections as seen from `pose` (the robot's pose now): x_loc/y_loc
    (and dist) re-expressed from each sighting's fixed odom point, so a static
    object stays put on the map however the robot has moved since the frame.
    The capture-frame values are kept as x_cap/y_cap/dist_cap. Detections that
    were never placed (no `odom`) are returned as they are, same object."""
    out = []
    for d in dets or []:
        if d.get("odom") is None:
            out.append(d)
            continue
        lx, ly = odom_to_local(d["odom"], pose)
        out.append(dict(d, x_cap=d["x_loc"], y_cap=d["y_loc"], dist_cap=d.get("dist"),
                        x_loc=lx, y_loc=ly, dist=math.hypot(lx, ly)))
    return out


class Stamped:
    """The last few sensor messages by time, so a detection can be paired with
    the depth frame / lidar scan from its own capture instead of the newest one
    (while the robot turns, the newest depth frame looks somewhere else)."""

    def __init__(self, maxlen=8):
        self._q = deque(maxlen=maxlen)

    def add(self, t, item):
        self._q.append((float(t), item))

    def nearest(self, t, max_dt=0.3):
        """(stamp, item) nearest t if within max_dt, else the latest (as before
        this existed); (None, None) if empty."""
        q = list(self._q)
        if not q:
            return None, None
        if t is not None:
            best = min(q, key=lambda p: abs(p[0] - t))
            if abs(best[0] - t) <= max_dt:
                return best
        return q[-1]


def msg_time(msg, now):
    """A message's header stamp if plausible against its arrival `now`, else `now`."""
    st = getattr(getattr(msg, "header", None), "stamp", None)
    stamp = (st.sec + st.nanosec * 1e-9) if st is not None and hasattr(st, "sec") else None
    return capture_time(stamp, now, default_lag=0.0)[0]
