#!/usr/bin/env python3
"""A static ball must stay put on the map while the robot moves.

The camera runs at ~2 fps and its detections arrive ~0.3 s after capture. If a
sighting is placed with the robot's pose when it is *used* rather than when the
frame was *taken*, a static ball slides around the map as the robot turns and
drives. These tests pin the fix (capture_pose.py, used by the navigator and by
RUN_POLICY): each sighting is placed once with the pose at capture and stays put.

Pure Python (no ROS, numpy or OpenCV):  python3 tests/test_capture_pose.py
"""
import json
import math
import os
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

import capture_pose as CP     # noqa: E402
import policy_runner as PR    # noqa: E402

POLICY = os.path.join(HERE, "fixtures", "simple_push_policy.py")


class CaptureTime(unittest.TestCase):
    def test_uses_a_plausible_stamp(self):
        self.assertEqual(CP.capture_time(99.7, 100.0), (99.7, "stamp"))

    def test_falls_back_to_the_assumed_delay(self):
        for stamp in (None, 0, "junk", 100.5, 90.0):       # missing, zero, garbage, future, stale
            t, src = CP.capture_time(stamp, 100.0, default_lag=0.3)
            self.assertEqual(src, "assumed", stamp)
            self.assertAlmostEqual(t, 99.7)


class History(unittest.TestCase):
    def test_interpolates_and_clamps(self):
        h = CP.PoseHistory()
        h.add(0.0, 0.0, 0.0, 0.0)
        h.add(1.0, 1.0, 2.0, 0.5)
        h.add(0.5, 9.0, 9.0, 9.0)                          # out of order: ignored
        x, y, a = h.at(0.25)
        self.assertAlmostEqual(x, 0.25); self.assertAlmostEqual(y, 0.5); self.assertAlmostEqual(a, 0.125)
        self.assertEqual(h.at(-5.0), (0.0, 0.0, 0.0))
        self.assertEqual(h.at(5.0), (1.0, 2.0, 0.5))
        self.assertIsNone(CP.PoseHistory().at(1.0))

    def test_yaw_interpolates_across_pi(self):
        h = CP.PoseHistory()
        h.add(0.0, 0.0, 0.0, math.pi - 0.1)
        h.add(1.0, 0.0, 0.0, -math.pi + 0.1)               # turned 0.2 rad left through +-pi
        self.assertAlmostEqual(abs(h.at(0.5)[2]), math.pi, places=6)


class PlaceAndReexpress(unittest.TestCase):
    def test_static_point_stays_put(self):
        seen_from = (1.0, 1.0, math.pi / 2)                # facing +y
        det = dict(label="sports ball", x_loc=2.0, y_loc=0.0, dist=2.0)
        CP.place([det], seen_from, 10.0, "stamp")
        self.assertEqual(det["odom"], [1.0, 3.0])
        self.assertEqual(det["pose_cap"], [1.0, 1.0, round(math.pi / 2, 4)])
        now = (0.0, 3.0, 0.0)                              # moved and turned since
        (d,) = CP.dets_now([det], now)
        self.assertAlmostEqual(d["x_loc"], 1.0, places=3)  # 1 m ahead of where it is now
        self.assertAlmostEqual(d["y_loc"], 0.0, places=3)
        self.assertAlmostEqual(d["dist"], 1.0, places=3)
        self.assertEqual((d["x_cap"], d["y_cap"], d["dist_cap"]), (2.0, 0.0, 2.0))
        self.assertEqual(det["x_loc"], 2.0, "the stored detection must not change")

    def test_unplaced_detections_pass_through_unchanged(self):
        det = dict(label="sports ball", x_loc=1.0, y_loc=0.0, dist=1.0)
        self.assertIs(CP.dets_now([det], (5.0, 5.0, 1.0))[0], det)


class StampedFrames(unittest.TestCase):
    def test_pairs_a_detection_with_the_frame_from_its_capture(self):
        h = CP.Stamped(6)
        for k in range(5):
            h.add(100.0 + 0.5 * k, f"depth{k}")            # 2 fps
        self.assertEqual(h.nearest(101.05), (101.0, "depth2"))
        self.assertEqual(h.nearest(None), (102.0, "depth4"))
        self.assertEqual(h.nearest(50.0), (102.0, "depth4"), "nothing near: fall back to the newest")
        self.assertEqual(CP.Stamped().nearest(1.0), (None, None))

    def test_message_time_from_its_header(self):
        NS = __import__("types").SimpleNamespace
        msg = NS(header=NS(stamp=NS(sec=99, nanosec=500_000_000)))
        self.assertAlmostEqual(CP.msg_time(msg, 100.0), 99.5)
        self.assertEqual(CP.msg_time(NS(), 100.0), 100.0)     # no header: arrival
        self.assertEqual(CP.msg_time(NS(header=NS(stamp=NS(sec=0, nanosec=0))), 100.0), 100.0)


class Navigator:
    """The camera -> navigator -> RUN_POLICY path, as on the robot: frames every
    1/hz s, delivered `lag` s later, placed with capture_pose exactly as
    tb4_claude_nav._nn_cb does (header stamp or the assumed delay), and served
    re-expressed for the current pose, as tb4_claude_nav._live_det does."""
    HALF_FOV = math.atan(208 / 325.95)                     # OAK-D 416 preview

    def __init__(self, ball, hz=2.0, lag=0.3, stamped=True, assumed_lag=0.3, at_arrival=False):
        self.ball, self.hz, self.lag, self.stamped, self.assumed = ball, hz, lag, stamped, assumed_lag
        self.at_arrival = at_arrival        # the old behaviour: place with the pose when it arrived
        t0 = 1000.0                                        # a real-looking clock: stamps > 0
        self.st = dict(t=t0, x=0.0, y=0.0, yaw=0.0, v=0.0, w=0.0)
        self.hist = CP.PoseHistory()
        self.hist.add(t0, 0.0, 0.0, 0.0)
        self.pending, self.dets, self.next_cap = [], [], t0

    def _camera(self):
        t = self.st["t"]
        while self.next_cap <= t + 1e-9:
            tc = self.next_cap
            x, y, yaw = self.hist.at(tc)                   # where the robot truly was
            cx, cy = x + PR.CAM_X * math.cos(yaw), y + PR.CAM_X * math.sin(yaw)
            lx, ly = CP.odom_to_local(self.ball, (cx, cy, yaw))
            brg, rc = math.atan2(ly, lx), math.hypot(lx, ly)
            if lx > 0 and abs(brg) <= self.HALF_FOV:
                surf = rc - PR.BALL_R                       # the navigator reports the visible surface
                d = dict(label="sports ball", x_loc=surf * math.cos(brg), y_loc=surf * math.sin(brg), dist=surf)
                self.pending.append((tc + self.lag, tc, d))
            else:
                self.pending.append((tc + self.lag, tc, None))
            self.next_cap += 1.0 / self.hz
        for item in [p for p in self.pending if p[0] <= t + 1e-9]:
            self.pending.remove(item)
            arrival, tc, d = item
            if d is None:
                self.dets = []
                continue
            t_cap, src = CP.capture_time(tc if self.stamped else None, arrival, default_lag=self.assumed)
            if self.at_arrival:
                t_cap = arrival
            d["t"] = arrival
            self.dets = CP.place([d], self.hist.at(t_cap), t_cap, src)

    def live_det(self, name):
        s = self.st
        found = CP.dets_now(self.dets, (s["x"], s["y"], s["yaw"]))
        return found[0] if found else None

    def get(self, k):
        s = self.st
        return {"odom_x": s["x"], "odom_y": s["y"], "odom_yaw": s["yaw"], "odom_t": s["t"],
                "hazards": [], "last_hazard_t": 0.0, "scan": None, "scan_t": None, "ball_goal": None}.get(k)

    def send_twist(self, v, w):
        self.st["v"], self.st["w"] = v, w

    def sleep(self, dt):
        s = self.st
        for _ in range(max(1, int(round(dt / 0.01)))):
            s["yaw"] += s["w"] * 0.01
            s["x"] += s["v"] * math.cos(s["yaw"]) * 0.01
            s["y"] += s["v"] * math.sin(s["yaw"]) * 0.01
            s["t"] = round(s["t"] + 0.01, 6)
            self.hist.add(s["t"], s["x"], s["y"], s["yaw"])
            self._camera()

    def run(self, goal, max_time=20.0):
        marks = []

        def set_(**kw):
            if "ball_mark" in kw:
                marks.append((self.st["x"], self.st["y"], kw["ball_mark"]))
        self._camera()
        with tempfile.TemporaryDirectory() as d:
            PR.run(POLICY, self.get, set_, lambda: self, self.live_det, lambda: None, goal=goal,
                   max_time=max_time, log_dir=d, sleep=self.sleep, clock=lambda: self.st["t"])
            with open(os.path.join(d, os.listdir(d)[0])) as f:
                rows = [json.loads(line) for line in f]
        return [r for r in rows if "kind" not in r], marks


LAYOUTS = [  # ball, goal: every one but the first makes the seed policy turn with the ball out of view
    ((1.2, 0.0), (2.4, 0.0)),
    ((1.0, 0.6), (2.0, 1.4)),
    ((1.2, 0.0), (1.4, 1.2)),
    ((-1.0, 0.3), (-2.2, 0.3)),
]


class StaticBallUnderRunPolicy(unittest.TestCase):
    CONTACT = 0.18 + PR.BALL_R + 0.05      # robot radius + ball radius + margin: after this the ball would move

    def _errors(self, nav, goal):
        ticks, marks = nav.run(goal)
        errs, seen = [], 0
        for r in ticks:
            if math.hypot(r["x"] - nav.ball[0], r["y"] - nav.ball[1]) < self.CONTACT:
                break
            if r["ball"] is None:
                continue
            seen += 1
            # what the controller was handed, put back on the map with the pose it was handed at
            bx, by = CP.local_to_odom((r["ball"]["x_loc"], r["ball"]["y_loc"]), (r["x"], r["y"], r["yaw"]))
            errs.append(math.hypot(bx - nav.ball[0], by - nav.ball[1]))
            b = (r.get("belief") or {}).get("ball")
            if b:
                errs.append(math.hypot(b[0] - nav.ball[0], b[1] - nav.ball[1]))
            errs.append(math.hypot(r["ball_odom"][0] - nav.ball[0], r["ball_odom"][1] - nav.ball[1]))
        for x, y, m in marks:                # the map's BALL marker, until the robot reaches the ball
            if math.hypot(x - nav.ball[0], y - nav.ball[1]) < self.CONTACT:
                break
            errs.append(math.hypot(m[0] - nav.ball[0], m[1] - nav.ball[1]))
        return errs, seen

    def test_stamped_frames_static_ball_stays_put(self):
        for ball, goal in LAYOUTS:
            with self.subTest(ball=ball, goal=goal):
                errs, seen = self._errors(Navigator(ball, lag=0.3, stamped=True), goal)
                self.assertGreater(seen, 10)
                self.assertLess(max(errs), 0.01, f"static ball wandered {100 * max(errs):.1f} cm")

    def test_unstamped_frames_with_the_right_assumed_delay(self):
        errs, _ = self._errors(Navigator(*LAYOUTS[2][:1], lag=0.3, stamped=False, assumed_lag=0.3), LAYOUTS[2][1])
        self.assertLess(max(errs), 0.01)

    def test_placing_with_the_pose_at_arrival_wanders(self):
        """The control: the same robot and camera, placed the old way. If this ever
        passes the 1 cm bar, the tests above no longer measure anything."""
        worst = max(max(self._errors(Navigator(b, lag=0.3, at_arrival=True), g)[0]) for b, g in LAYOUTS)
        self.assertGreater(worst, 0.05)

    def test_seed_policy_still_reaches_and_pushes_the_ball(self):
        ball, goal = LAYOUTS[1]
        nav = Navigator(ball, lag=0.3)
        ticks, _ = nav.run(goal)
        closest = min(math.hypot(r["x"] - ball[0], r["y"] - ball[1]) for r in ticks)
        self.assertLess(closest, self.CONTACT, "the seed policy never got to the ball")


if __name__ == "__main__":
    unittest.main(verbosity=2)
