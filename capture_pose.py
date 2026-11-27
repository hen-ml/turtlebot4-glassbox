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
# Camera (OAK-D Lite RGB) position ahead of base_link, from the TurtleBot 4 Lite
# description (turtlebot4_description/urdf/lite: camera bracket at x -9.0 cm, the
# OAK-D Lite +5.29 cm on it -> oakd_rgb_camera_frame at x -0.0371 m, i.e. BEHIND
# the base centre; 18.6 cm above the floor). Override with TB4_CAM_X.
CAM_X = float(os.environ.get("TB4_CAM_X", "-0.037"))
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


def place(dets, pose_cap, t_cap, cap_src, cam_x=0.0):
    """Annotate detections (dicts with x_loc/y_loc relative to the camera, which is
    `cam_x` ahead of the base centre, at capture) in place with where they were
    seen: t_cap, cap_src, pose_cap and `odom`, the sighting's odom point. Returns dets."""
    pc = [round(float(v), 4) for v in pose_cap]
    for d in dets:
        if d.get("x_loc") is None or d.get("y_loc") is None:
            continue
        ox, oy = local_to_odom((d["x_loc"] + cam_x, d["y_loc"]), pose_cap)
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


# Focal length (px) of the 416 preview YOLO runs on: the centre 720x720 of the
# 1280x720 RGB frame (fx 1002.91) resized to 416 (see tb4_claude_nav CAM_FX).
PREVIEW_FX = 579.45
# YOLO box width / the ball's true angular diameter. Calibrated 2026-11-26 on a ball
# 54 cm round (diameter 0.172 m): box width x surface distance ~= 104 px m over 7
# whole sightings at 0.5-1.6 m, i.e. 1.0435 x fx x diameter.
BOX_K = 1.0435


class BallSizer:
    """The ball's radius, estimated from the camera, so it need not be set per ball.

    Each whole sighting gives diameter ~= w_px * distance / (BOX_K * PREVIEW_FX), with
    `distance` the stereo distance to the ball's visible surface (the navigator's
    `dist`). Sightings cut off by the frame edge, closer than `rng[0]` (stereo and
    boxes degrade) or farther than `rng[1]`, or smaller than `min_w` px are skipped.
    radius() is the override if given (TB4_BALL_R); else the median of the
    estimates once there are `min_n`, clamped to `clamp`; else `default`. The median
    keeps the occasional stereo reading of the background (seen on a dark ball)
    from dominating."""

    def __init__(self, default=0.11, override=None, min_n=3, rng=(0.5, 2.5), min_w=15.0,
                 clamp=(0.03, 0.15), fx=None, box_k=None):
        self.default, self.override = float(default), (None if override is None else float(override))
        self.min_n, self.rng, self.min_w, self.clamp = min_n, rng, min_w, clamp
        self.fx = PREVIEW_FX if fx is None else fx
        self.box_k = BOX_K if box_k is None else box_k
        self.estimates = []

    def add(self, det):
        """Take one sighting (needs w_px and dist; skips edge boxes). Returns the
        diameter estimate used, or None if the sighting was skipped."""
        try:
            w, d = det.get("w_px"), det.get("dist")
            if w is None or d is None or det.get("edge"):
                return None
            w, d = float(w), float(d)
        except (TypeError, ValueError, AttributeError):
            return None
        if not (self.rng[0] <= d <= self.rng[1]) or w < self.min_w:
            return None
        dia = w * d / (self.box_k * self.fx)
        self.estimates.append(dia)
        return dia

    @property
    def n(self):
        return len(self.estimates)

    def estimate(self):
        """Median-based radius from the sightings so far, or None if too few."""
        if len(self.estimates) < self.min_n:
            return None
        e = sorted(self.estimates)
        m = len(e) // 2
        med = e[m] if len(e) % 2 else 0.5 * (e[m - 1] + e[m])
        return min(self.clamp[1], max(self.clamp[0], med / 2.0))

    def radius(self):
        if self.override is not None:
            return self.override
        r = self.estimate()
        return self.default if r is None else r
