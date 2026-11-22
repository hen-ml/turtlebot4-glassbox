"""Mission loop: compile → human gate → execute leg-by-leg → verify →
bounded re-plan. Robot-agnostic except for the adapter it is handed.

Everything here talks to the robot only through the adapter in robot_api.py.
"""
import json
import os
import time

from . import plan as planmod
from .llm_client import LLMError
from .robot_api import RobotError

DEFAULT_DEADLINE_S = 900.0
PROGRAM_TIMEOUT_S = 300.0
GOTO_TIMEOUT_S = 90.0
DOCK_TIMEOUT_S = 90.0
MAX_REPLANS_TOTAL = 2


def build_catalog(repo_root):
    """'name — description' lines for every program/*.toy, from the headers."""
    lines = []
    pdir = os.path.join(repo_root, "programs")
    for f in sorted(os.listdir(pdir)):
        if not f.endswith(".toy"):
            continue
        name = f[:-4]
        desc = ""
        with open(os.path.join(pdir, f)) as fh:
            for line in fh:
                low = line.lower()
                if low.startswith("# description:"):
                    desc = line.split(":", 1)[1].strip()
                    break
        lines.append("  - %s — %s" % (name, desc or "(no description)"))
    return "\n".join(lines)


def _compile_messages(mission_text, catalog, prior=None):
    if prior is None:
        user = ("Mission: %s\n\nProgram catalog:\n%s\n\n"
                "Compile the mission into the JSON target described above. "
                "Prefer composing catalog programs; only use goto for explicit "
                "coordinate requests; end with dock only if the user asked. "
                "Keep steps minimal." % (mission_text, catalog))
    else:
        user = ("Original mission: %s\n\nProgram catalog:\n%s\n\n"
                "Progress so far:\n%s\n\nThe leg that just failed: %s\nError: %s\n\n"
                "Current robot state: %s\n\n"
                "Re-plan ONLY the remaining work, starting from the current state. "
                "Return the same JSON shape for the revised remainder of the "
                "mission." % (prior["mission"], catalog, prior["progress"],
                              prior["failed_leg"], prior["error"], prior["state"]))
    system = (
        "You are the mission compiler for a small indoor TurtleBot 4. You output "
        "ONE JSON object and nothing else, matching this schema exactly:\n"
        + planmod.MISSION_SCHEMA_HINT +
        "\nRules:\n"
        "- steps[i].kind is one of: run_program, goto, dock, undock, wait.\n"
        "- run_program.name must be copied verbatim from the program catalog.\n"
        "- Never invent coordinates; goto only for explicit coordinate missions.\n"
        "- No prose outside the JSON object.\n"
        "- If the mission is physically impossible with the catalog, return steps: [] "
        "and explain in rationale (the harness will refuse to run)."
    )
    return [{"role": "system", "content": system},
            {"role": "user", "content": user}]


def _state_brief(st):
    return json.dumps({k: st.get(k) for k in
                       ("status", "nav_active", "run_active", "run_error",
                        "odom_x", "odom_y", "docked", "battery_percentage",
                        "run_program")}, sort_keys=True)


class LegResult(object):
    def __init__(self, step, ok, detail):
        self.step = step
        self.ok = ok
        self.detail = detail

    def line(self):
        return "%s %s — %s" % ("✓" if self.ok else "✗", self.step.describe(), self.detail)


class MissionHarness(object):
    def __init__(self, robot, llm, out=print, repo_root=None, sleep=time.sleep):
        self.robot = robot
        self.llm = llm
        self.out = out
        self.repo_root = repo_root or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.sleep = sleep
        self.allowed = planmod.known_programs(self.repo_root)
        try:
            live = self.robot.list_programs()
            if live:
                self.allowed = sorted(set(self.allowed) | set(live))
        except Exception:
            pass  # offline / dry-run: repo catalog is enough
        self._abort = False

    # ── phases ─────────────────────────────────────────────────────────────
    def compile(self, mission_text, prior=None):
        catalog = build_catalog(self.repo_root)
        raw = self.llm.chat(_compile_messages(mission_text, catalog, prior))
        obj = planmod.extract_json(raw)
        return planmod.validate_plan(obj, self.allowed)

    def confirm(self, plan):
        self.out("\n" + plan.describe() + "\n")

    # ── legs ───────────────────────────────────────────────────────────────
    def _poll(self, predicate, timeout_s):
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            if self._abort:
                return None
            st = self.robot.get_state()
            if predicate(st):
                return st
            self.sleep(1.0)
        return None

    def _leg_wait(self, step):
        remaining = step.args["secs"]
        while remaining > 0 and not self._abort:
            chunk = min(0.5, remaining)
            self.sleep(chunk)
            remaining -= chunk
        return LegResult(step, True, "dwelled %.0fs" % step.args["secs"])

    def _leg_goto(self, step):
        self.robot.goto_xy(step.args["x"], step.args["y"], step.args["mode"])
        st = self._poll(lambda s: not s.get("nav_active", True), GOTO_TIMEOUT_S)
        if st is None:
            return LegResult(step, False, "goto timed out / aborted")
        status = str(st.get("status", ""))
        if status.lower().startswith("blocked"):
            return LegResult(step, False, status)
        return LegResult(step, True, status or "arrived")

    def _leg_run_program(self, step):
        name = step.args["name"]
        self.robot.run_program(name)
        st = self._poll(lambda s: not s.get("run_active", True), PROGRAM_TIMEOUT_S)
        if st is None:
            self.robot.run_stop()
            return LegResult(step, False, "program %s timed out; stopped" % name)
        err = st.get("run_error")
        if err:
            return LegResult(step, False, str(err))
        return LegResult(step, True, str(st.get("status") or "done"))

    def _leg_dock(self, want_docked, step):
        (self.robot.dock if want_docked else self.robot.undock)()
        st = self._poll(lambda s: not s.get("dock_busy", False)
                        and bool(s.get("docked")) == want_docked, DOCK_TIMEOUT_S)
        if st is None:
            return LegResult(step, False, "dock state not reached")
        return LegResult(step, True, "docked" if want_docked else "undocked")

    def _run_leg(self, step):
        try:
            if step.kind == "wait":
                return self._leg_wait(step)
            if step.kind == "goto":
                return self._leg_goto(step)
            if step.kind == "run_program":
                return self._leg_run_program(step)
            if step.kind == "dock":
                return self._leg_dock(True, step)
            if step.kind == "undock":
                return self._leg_dock(False, step)
            return LegResult(step, False, "unknown leg kind %r" % step.kind)
        except RobotError as e:
            return LegResult(step, False, str(e))

    # ── main entry ───────────────────────────────────────────────────────────
    def run(self, mission_text, assume_yes=False, deadline_s=DEFAULT_DEADLINE_S):
        report = {"mission": mission_text, "legs": [], "replans": 0, "ok": False, "error": None}
        t0 = time.time()
        try:
            preflight, st = self.robot.health()
            self.out("pre-flight: cam=%s lidar=%s base=%s | docked=%s | battery=%s%%" % (
                "OK" if st.get("cam_ok") else "--",
                "OK" if st.get("lidar_ok") else "--",
                "OK" if st.get("base_ok") else "--",
                st.get("docked"), st.get("battery_percentage")))
            if not preflight:
                report["error"] = "pre-flight failed (need cam+lidar+base green) — fix the stack first"
                self.out(report["error"]); return report

            plan = self.compile(mission_text)
            if not plan.steps:
                report["error"] = "compiler produced no steps: " + plan.rationale
                self.out(report["error"]); return report

            if st.get("docked") and any(s.kind != "undock" and s.kind != "wait"
                                        for s in plan.steps):
                report["error"] = "robot is docked — undock first (wheels are disabled)"
                self.out(report["error"]); return report

            self.confirm(plan)
            if not assume_yes:
                try:
                    ans = input("\nProceed? [y/N] ").strip().lower()
                except EOFError:
                    ans = ""
                if ans not in ("y", "yes"):
                    report["error"] = "aborted by operator"
                    self.out(report["error"]); return report

            pending = list(plan.steps)
            done = []
            while pending:
                if time.time() - t0 > deadline_s:
                    self.robot.run_stop()
                    report["error"] = "mission deadline exceeded — robot stopped"
                    self.out(report["error"]); return report
                step = pending.pop(0)
                self.out("→ " + step.describe())
                res = self._run_leg(step)
                self.out("   " + res.line())
                report["legs"].append({"step": step.describe(), "ok": res.ok,
                                       "detail": res.detail})
                if res.ok:
                    done.append(res)
                    continue
                if report["replans"] >= MAX_REPLANS_TOTAL:
                    break
                prior = {
                    "mission": mission_text,
                    "progress": "\n".join("- %s: ok — %s" % (d.step.describe(), d.detail)
                                          for d in done) or "(nothing completed)",
                    "failed_leg": step.describe(),
                    "error": res.detail,
                    "state": _state_brief(self.robot.get_state()),
                }
                self.out("   re-planning (%d/%d)…" % (report["replans"] + 1, MAX_REPLANS_TOTAL))
                try:
                    plan2 = self.compile(mission_text, prior=prior)
                except (LLMError, planmod.PlanError) as e:
                    self.out("   re-plan failed: %s" % e)
                    break
                report["replans"] += 1
                if not plan2.steps:
                    report["error"] = "re-plan produced no steps: " + plan2.rationale
                    break
                pending = plan2.steps  # the re-plan returns the *remainder*
            else:   # pending drained without break/error → mission recovered
                report["ok"] = True
                return report

            if not report["error"]:
                report["error"] = "leg failed after re-plan budget (%d used) — robot stopped" % report["replans"]
            self.robot.run_stop()
            self.out(report["error"])
            return report
        except KeyboardInterrupt:
            self.out("interrupted — stopping robot")
            self.robot.run_stop()
            report["error"] = "interrupted by operator"
            return report
        except (LLMError, planmod.PlanError) as e:
            report["error"] = f"compile failed: {e}"
            self.out(report["error"]); return report
        except RobotError as e:
            report["error"] = f"robot unreachable: {e}"
            self.out(report["error"]); return report
