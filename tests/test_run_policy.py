#!/usr/bin/env python3
"""Offline tests for RUN_POLICY (policy_runner + NavRobot.run_policy).

Reuses the closed-loop FakeBase from test_push_to_goal (stubbed ROS, the real
navigator module, a base that integrates cmd_vel and rolls the ball) and drives
it with a small controller file written against evo.policy_api.

Run:  python3 tests/test_run_policy.py
"""
import json
import math
import os
import sys
import tempfile
import time
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))

from test_push_to_goal import FakeBase, tb4          # noqa: E402  (stubs ROS on import)
import policy_runner                                 # noqa: E402
from evo.policy_api import PolicyRejected, check_source   # noqa: E402

POLICY = os.path.join(HERE, "fixtures", "simple_push_policy.py")


class TestPolicyCheck(unittest.TestCase):
    def test_rejects_file_system_access(self):
        with self.assertRaises(PolicyRejected):
            check_source("import os\ndef make_policy():\n    return None\n")
        with self.assertRaises(PolicyRejected):
            check_source("class Policy:\n    def act(self, o):\n        return open('x')\n")

    def test_accepts_the_fixture(self):
        with open(POLICY) as f:
            check_source(f.read())


class TestRunPolicy(unittest.TestCase):
    def setUp(self):
        self.logdir = tempfile.mkdtemp()
        policy_runner.LOG_DIR = self.logdir
        # FakeBase publishes the ball centre relative to the base, so no camera
        # offset or radius to add in this world.
        self._saved = (policy_runner.CAM_X, policy_runner.BALL_R)
        policy_runner.CAM_X, policy_runner.BALL_R = 0.0, 0.0

    def tearDown(self):
        policy_runner.CAM_X, policy_runner.BALL_R = self._saved

    def _run(self, ball, goal, max_time=60.0):
        tb4._set(ball_goal=goal)
        with FakeBase(ball) as base:
            r = policy_runner.run(POLICY, get=tb4._get, set_=tb4._set, node=lambda: tb4.ros_node,
                                  live_det=tb4.NavRobot()._live_det, check=lambda: None,
                                  max_time=max_time, log_dir=self.logdir)
            return r, base

    def test_pushes_the_ball_toward_the_goal(self):
        goal = (2.6, 0.4)
        r, base = self._run(ball=(1.0, 0.2), goal=goal)
        d0 = math.hypot(1.0 - goal[0], 0.2 - goal[1])
        self.assertIn(r, ("done", "settled", "timeout"))
        self.assertLess(base.ball_dist_to(goal), d0 - 0.5, f"ball barely moved ({r})")

    def test_no_goal(self):
        tb4._set(ball_goal=None)
        with FakeBase((1.0, 0.0)):
            r = policy_runner.run(POLICY, get=tb4._get, set_=tb4._set, node=lambda: tb4.ros_node,
                                  live_det=tb4.NavRobot()._live_det, check=lambda: None,
                                  log_dir=self.logdir)
        self.assertEqual(r, "no-goal")

    def test_logs_every_tick_and_stops_the_base(self):
        r, base = self._run(ball=(1.0, 0.0), goal=(2.0, 0.0), max_time=3.0)
        files = os.listdir(self.logdir)
        self.assertEqual(len(files), 1)
        with open(os.path.join(self.logdir, files[0])) as f:
            lines = [json.loads(x) for x in f]
        self.assertEqual(lines[0]["kind"], "start")
        self.assertEqual(lines[-1]["kind"], "end")
        self.assertGreater(len(lines), 10)
        self.assertEqual(base.twists[-1], (0.0, 0.0))

    def test_bad_policy_never_moves_the_robot(self):
        bad = os.path.join(self.logdir, "bad.py")
        with open(bad, "w") as f:
            f.write("import subprocess\ndef make_policy():\n    return None\n")
        tb4._set(ball_goal=(2.0, 0.0))
        with FakeBase((1.0, 0.0)) as base:
            r = policy_runner.run(bad, get=tb4._get, set_=tb4._set, node=lambda: tb4.ros_node,
                                  live_det=tb4.NavRobot()._live_det, check=lambda: None,
                                  log_dir=self.logdir)
        self.assertTrue(r.startswith("error"))
        self.assertEqual(base.twists, [])

    def test_navrobot_method_resolves_relative_paths(self):
        tb4._set(ball_goal=(2.0, 0.0))
        rel = os.path.relpath(POLICY, os.path.dirname(HERE))
        with FakeBase((1.0, 0.0)):
            t0 = time.time()
            r = tb4.NavRobot().run_policy(rel, max_time=2.0)
        self.assertIn(r, ("done", "settled", "timeout"))
        self.assertLess(time.time() - t0, 10)


def _write_policy(dirname, body):
    path = os.path.join(dirname, "p.py")
    with open(path, "w") as f:
        f.write("class Policy:\n    def reset(self, goal):\n        self.goal = goal\n"
                "    def act(self, obs):\n" + body)
    return path


class TestObservationMatchesTheSimulator(unittest.TestCase):
    """The robot must hand the policy what the simulator does (evo.policy_api)."""

    def setUp(self):
        self.logdir = tempfile.mkdtemp()

    def _log(self):
        f = [x for x in os.listdir(self.logdir) if x.endswith(".jsonl")][-1]
        with open(os.path.join(self.logdir, f)) as fh:
            return [json.loads(x) for x in fh]

    def _run(self, policy, ball=(3.0, 3.0), goal=(2.0, 0.0), state=None, **kw):
        tb4._set(ball_goal=goal)
        with FakeBase(ball) as base:
            if state:
                tb4._set(**state)                 # after FakeBase, which resets the world
            r = policy_runner.run(policy, get=tb4._get, set_=tb4._set, node=lambda: tb4.ros_node,
                                  live_det=tb4.NavRobot()._live_det, check=lambda: None,
                                  log_dir=self.logdir, **kw)
        return r, base

    def test_ball_is_its_centre_in_the_robot_frame(self):
        b = policy_runner.ball_in_robot_frame
        # p25 depth (axial) 1.0 m straight ahead -> centre 1.0 + r ahead of the lens
        d = b(dict(x_loc=1.0, y_loc=0.0, dist=1.0, p25_dist=1.0), cam_x=0.08, ball_r=0.11)
        self.assertAlmostEqual(d["x_loc"], 0.08 + 1.11, places=6)
        self.assertAlmostEqual(d["dist"], 1.11, places=6)
        # p25 depth at 30 deg is axial: range = 1 / cos 30
        a = math.radians(30)
        d = b(dict(x_loc=math.cos(a), y_loc=math.sin(a), dist=1.0, p25_dist=1.0), cam_x=0.0, ball_r=0.0)
        self.assertAlmostEqual(d["dist"], 1.0 / math.cos(a), places=6)
        # raw OAK spatial distance is already a range
        d = b(dict(x_loc=math.cos(a), y_loc=math.sin(a), dist=1.0, p25_dist=None), cam_x=0.0, ball_r=0.1)
        self.assertAlmostEqual(d["dist"], 1.1, places=6)
        self.assertAlmostEqual(math.atan2(d["y_loc"], d["x_loc"]), a, places=6)

    def test_bump_is_only_a_fresh_hazard(self):
        pol = _write_policy(self.logdir, "        return (0.0, 0.0, bool(obs.bump))\n")
        bump = [{"type": "bump", "where": "front"}]
        r, _ = self._run(pol, max_time=4.0, state=dict(hazards=bump, last_hazard_t=time.time() - 1.5))
        self.assertEqual(r, "settled", "a 1.5 s old bump must not count")
        r, _ = self._run(pol, max_time=4.0, state=dict(hazards=bump, last_hazard_t=time.time() + 60))
        self.assertEqual(r, "done", "a fresh bump must count")
        tb4._set(hazards=[], last_hazard_t=0.0)

    def test_v_and_w_are_measured_from_odometry(self):
        pol = _write_policy(self.logdir, "        return (0.2, 0.5, obs.t > 2.0)\n")
        r, _ = self._run(pol, max_time=5.0)
        ticks = [x for x in self._log() if "v_meas" in x and x["t"] > 1.0]
        self.assertTrue(ticks)
        self.assertAlmostEqual(sum(x["v_meas"] for x in ticks) / len(ticks), 0.2, delta=0.04)
        self.assertAlmostEqual(sum(x["w_meas"] for x in ticks) / len(ticks), 0.5, delta=0.08)

    def test_goal_relative_to_the_start_pose(self):
        pol = _write_policy(self.logdir, "        return (0.0, 0.0, True)\n")
        tb4._set(ball_goal=None)
        with FakeBase((3.0, 3.0), pose=(1.0, 2.0, math.pi / 2)):
            policy_runner.run(pol, get=tb4._get, set_=tb4._set, node=lambda: tb4.ros_node,
                              live_det=tb4.NavRobot()._live_det, check=lambda: None,
                              log_dir=self.logdir, goal_rel=(1.0, 0.5))
        g = self._log()[0]["goal"]
        self.assertAlmostEqual(g[0], 1.0 - 0.5, places=3)
        self.assertAlmostEqual(g[1], 2.0 + 1.0, places=3)

    def test_lidar_beams_in_the_robot_frame(self):
        """A wall 1 m straight ahead of the robot, in the RPLIDAR's own frame (mounted
        yaw +90 deg), must come out as the forward beam; nothing elsewhere = max."""
        class Scan:
            pass
        sc = Scan()
        sc.angle_min, sc.angle_increment, sc.range_min = -math.pi, math.radians(1.0), 0.15
        sc.ranges = [float("inf")] * 360
        fwd_lidar = math.atan2(-math.cos(0.0), math.sin(0.0))        # base bearing 0 in lidar angles
        k = int(round((fwd_lidar - sc.angle_min) / sc.angle_increment)) % 360
        for j in (k - 1, k, k + 1):
            sc.ranges[j % 360] = 1.0
        beams = policy_runner.scan_beams(sc)
        self.assertEqual(len(beams), policy_runner.LIDAR_BEAMS)
        self.assertAlmostEqual(beams[policy_runner.LIDAR_BEAMS // 2], 1.0)   # straight ahead
        self.assertEqual(beams[0], policy_runner.LIDAR_MAX)                    # behind: nothing
        self.assertEqual(sum(b < policy_runner.LIDAR_MAX for b in beams), 1)

    def test_toyscript_verb(self):
        import toyscript
        out = []
        it = toyscript.Interpreter(toyscript.MockRobot(log=out.append), log=out.append)
        it.run('SET r = RUN_POLICY_REL("policies/current.py", 1.5, -0.25)\n')
        self.assertEqual(it.env["r"], "done")
        self.assertTrue(any("RUN_POLICY_REL" in x and "fwd=1.50" in x for x in out), out)
        it.run('SET b = PUSH_TO_GOAL_REL(1.5, -0.25)\n')
        self.assertEqual(it.env["b"], "at goal")

    def test_baseline_goal_relative_to_the_start_pose(self):
        got = {}
        nav = tb4.NavRobot()
        nav.push_to_goal = lambda gx, gy, max_time=None: got.update(g=(gx, gy), t=max_time) or "at goal"
        with FakeBase((3.0, 3.0), pose=(1.0, 2.0, math.pi / 2)):
            r = nav.push_to_goal_rel(1.0, 0.5)
        self.assertEqual(r, "at goal")
        self.assertAlmostEqual(got["g"][0], 0.5, places=3)
        self.assertAlmostEqual(got["g"][1], 3.0, places=3)
        self.assertEqual(got["t"], 90.0)

if __name__ == "__main__":
    unittest.main(verbosity=2)
