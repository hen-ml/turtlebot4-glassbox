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


if __name__ == "__main__":
    unittest.main(verbosity=2)
