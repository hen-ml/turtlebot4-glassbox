#!/usr/bin/env python3
"""Tests for the laptop harness. No robot, no model, no network —
both boundaries are faked via dependency injection. Stdlib only.

Run:  python3 tests/test_harness.py
"""
import builtins
import json
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from harness import plan as planmod
from harness.llm_client import LLMClient
from harness.mission import MissionHarness


# ── fakes ────────────────────────────────────────────────────────────────────
def make_llm(replies):
    it = iter(replies)

    def fetcher(url, payload, headers, timeout):
        content = next(it)
        return 200, json.dumps({"choices": [{"message": {"content": content}}]})

    return LLMClient("http://fake/v1", "fake-model", fetcher=fetcher)


class FakeRobot:
    """In-memory stand-in for the TB4 HTTP adapter."""

    def __init__(self, docked=False, run_error=None, program_status="done"):
        self.state = {"cam_ok": True, "lidar_ok": True, "base_ok": True,
                      "docked": docked, "dock_busy": False, "battery_percentage": 87,
                      "nav_active": False, "run_active": False, "run_error": None,
                      "status": "idle", "odom_x": 0.0, "odom_y": 0.0,
                      "run_program": None}
        self.calls = []
        self.run_error = run_error
        self.program_status = program_status

    def health(self):
        return True, dict(self.state)

    def list_programs(self):
        return ["push_ball_to_wall", "push_ball_to_goal", "follow", "open_explore"]

    def get_state(self):
        return dict(self.state)

    def run_program(self, name):
        self.calls.append(("run_program", name))
        self.state["run_active"] = False          # finishes instantly in-test
        self.state["run_error"] = self.run_error
        self.state["status"] = self.program_status
        if self.run_error:
            self.state["status"] = "blocked"

    def goto_xy(self, x, y, mode="move"):
        self.calls.append(("goto_xy", x, y, mode))
        self.state["nav_active"] = False
        self.state["status"] = "arrived (delta=0.10m)"

    def run_stop(self):
        self.calls.append(("run_stop",))
        self.state["run_active"] = False
        self.state["nav_active"] = False

    def dock(self):
        self.calls.append(("dock",))
        self.state["docked"] = True
        self.state["dock_busy"] = False

    def undock(self):
        self.calls.append(("undock",))
        self.state["docked"] = False
        self.state["dock_busy"] = False


def make_harness(llm, robot):
    return MissionHarness(robot, llm, out=lambda *a, **k: None,
                          repo_root=ROOT, sleep=lambda *_: None)


HAPPY = json.dumps({"goal": "push ball to wall",
                    "rationale": "one push program suffices",
                    "steps": [{"kind": "run_program", "name": "push_ball_to_wall"}]})

GOOD_REPLAN = json.dumps({"goal": "remainder",
                          "rationale": "retry with the goal variant",
                          "steps": [{"kind": "run_program", "name": "push_ball_to_goal"}]})


class TestPlan(unittest.TestCase):
    def test_extract_json_tolerates_fences_and_prose(self):
        out = planmod.extract_json('Sure!\n```json\n{"goal":"g","steps":[{"kind":"wait","secs":2}]}\n```')
        self.assertEqual(out["goal"], "g")

    def test_extract_json_rejects_garbage(self):
        with self.assertRaises(planmod.PlanError):
            planmod.extract_json("no json here at all")

    def test_validate_accepts_catalog_program_and_goto(self):
        plan = planmod.validate_plan(
            {"goal": "g", "rationale": "r",
             "steps": [{"kind": "undock"}, {"kind": "run_program", "name": "push_ball_to_wall"},
                       {"kind": "goto", "x": 1.5, "y": -2, "mode": "move"}]},
            ["push_ball_to_wall", "follow"])
        self.assertEqual([s.kind for s in plan.steps], ["undock", "run_program", "goto"])

    def test_validate_rejects_unknown_program(self):
        with self.assertRaises(planmod.PlanError):
            planmod.validate_plan({"goal": "g",
                                   "steps": [{"kind": "run_program", "name": "launch_rocket"}]},
                                  ["follow"])

    def test_validate_rejects_out_of_range_and_unknown_kinds(self):
        with self.assertRaises(planmod.PlanError):
            planmod.validate_plan({"goal": "g", "steps": [{"kind": "goto", "x": 999, "y": 0}]},
                                   ["follow"])
        with self.assertRaises(planmod.PlanError):
            planmod.validate_plan({"goal": "g", "steps": [{"kind": "self_destruct"}]},
                                   ["follow"])
        with self.assertRaises(planmod.PlanError):
            planmod.validate_plan({"goal": "g", "steps": [{"kind": "wait", "secs": 5}] * 13},
                                   ["follow"])


class TestMission(unittest.TestCase):
    def test_happy_path(self):
        robot = FakeRobot()
        rep = make_harness(make_llm([HAPPY]), robot).run("push the ball to the wall",
                                                         assume_yes=True)
        self.assertTrue(rep["ok"])
        self.assertIn(("run_program", "push_ball_to_wall"), robot.calls)
        self.assertNotIn(("run_stop",), robot.calls)

    def test_failure_triggers_one_replan_then_succeeds(self):
        robot = FakeRobot(run_error="stalled")
        llm = make_llm([HAPPY, GOOD_REPLAN])
        harness = make_harness(llm, robot)
        original_leg = harness._leg_run_program
        calls = {"n": 0}

        def leg(step):
            calls["n"] += 1
            if calls["n"] == 2:            # re-planned leg runs clean
                robot.run_error = None
            return original_leg(step)

        harness._leg_run_program = leg
        rep = harness.run("push the ball to the wall", assume_yes=True)
        self.assertEqual(rep["replans"], 1)
        self.assertTrue(rep["ok"])
        self.assertIn(("run_program", "push_ball_to_goal"), robot.calls)

    def test_repeated_failure_stops_robot_and_reports(self):
        robot = FakeRobot(run_error="lost")
        rep = make_harness(make_llm([HAPPY, GOOD_REPLAN, HAPPY]), robot).run(
            "push the ball to the wall", assume_yes=True)
        self.assertFalse(rep["ok"])
        self.assertEqual(rep["replans"], 2)
        self.assertIn(("run_stop",), robot.calls)
        self.assertTrue(rep["error"])

    def test_docked_robot_refuses_movement(self):
        robot = FakeRobot(docked=True)
        rep = make_harness(make_llm([HAPPY]), robot).run("push the ball to the wall",
                                                         assume_yes=True)
        self.assertFalse(rep["ok"])
        self.assertIn("docked", rep["error"])
        self.assertNotIn(("run_program", "push_ball_to_wall"), robot.calls)

    def test_operator_abort_at_gate(self):
        robot = FakeRobot()
        orig_input = builtins.input
        builtins.input = lambda *a: "n"
        try:
            rep = make_harness(make_llm([HAPPY]), robot).run("push the ball to the wall",
                                                             assume_yes=False)
        finally:
            builtins.input = orig_input
        self.assertEqual(rep["error"], "aborted by operator")
        self.assertEqual(robot.calls, [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
