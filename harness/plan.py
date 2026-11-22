"""Plan schema, tolerant JSON extraction, and hard validation.

The LLM's output is *untrusted input*. Everything here is pure and offline so
it can be tested without a robot or a model. The executor in mission.py only
ever sees a Plan that passed validate_plan().
"""
import glob
import json
import math
import os
import re

MISSION_SCHEMA_HINT = """{
  "goal": "<one-line restatement of the mission>",
  "rationale": "<2-4 sentences: how you will achieve it, why these legs>",
  "steps": [
    {"kind": "run_program", "name": "<program name from the catalog>"},
    {"kind": "goto", "x": 1.5, "y": -2.0, "mode": "move"},
    {"kind": "dock"} | {"kind": "undock"} | {"kind": "wait", "secs": 5}
  ]
}"""

MAX_STEPS = 12
MAX_COORD = 30.0
MAX_WAIT = 600.0
_NAME_RE = re.compile(r"^[a-z0-9_]+$")


class PlanError(ValueError):
    pass


def extract_json(text):
    """Find the first JSON object in model output (tolerates code fences and
    leading/trailing prose). Raises PlanError if none parses."""
    if not text or not text.strip():
        raise PlanError("empty model output")
    t = text.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", t, re.DOTALL | re.IGNORECASE)
    if fence:
        t = fence.group(1).strip()
    start = t.find("{")
    end = t.rfind("}")
    if start == -1 or end <= start:
        raise PlanError("no JSON object found in model output")
    try:
        return json.loads(t[start:end + 1])
    except ValueError as e:
        raise PlanError(f"invalid JSON from model: {e}") from e


def known_programs(repo_root):
    """Offline allowlist: the programs actually shipped in programs/*.toy."""
    names = []
    for f in sorted(glob.glob(os.path.join(repo_root, "programs", "*.toy"))):
        names.append(os.path.basename(f)[:-4])
    return names


def _num(v, what):
    try:
        f = float(v)
    except (TypeError, ValueError):
        raise PlanError(f"{what}: not a number: {v!r}")
    if not math.isfinite(f):
        raise PlanError(f"{what}: not finite: {v!r}")
    return f


class Step(object):
    def __init__(self, kind, args):
        self.kind = kind
        self.args = args

    def __repr__(self):
        return "Step(%s %s)" % (self.kind, self.args)

    def describe(self):
        if self.kind == "run_program":
            return "run %s" % self.args["name"]
        if self.kind == "goto":
            return "goto (%.2f, %.2f) [%s]" % (self.args["x"], self.args["y"], self.args["mode"])
        if self.kind == "wait":
            return "wait %.0fs" % self.args["secs"]
        return self.kind


class Plan(object):
    def __init__(self, goal, rationale, steps):
        self.goal = goal
        self.rationale = rationale
        self.steps = steps

    def describe(self):
        lines = ["Goal: %s" % self.goal]
        if self.rationale:
            lines.append("Why:  %s" % self.rationale)
        lines.append("Plan:")
        for i, s in enumerate(self.steps, 1):
            lines.append("  %d. %s" % (i, s.describe()))
        return "\n".join(lines)


def validate_plan(obj, allowed_programs):
    """Validate the raw model JSON. Returns a Plan or raises PlanError.
    `allowed_programs` is the set of executable program names."""
    if not isinstance(obj, dict):
        raise PlanError("plan must be a JSON object")
    goal = str(obj.get("goal", "")).strip()[:300]
    rationale = str(obj.get("rationale", "")).strip()[:600]
    raw_steps = obj.get("steps")
    if not isinstance(raw_steps, list) or not raw_steps:
        raise PlanError("plan needs a non-empty 'steps' list")
    if len(raw_steps) > MAX_STEPS:
        raise PlanError("too many steps (%d > %d)" % (len(raw_steps), MAX_STEPS))
    allowed = set(allowed_programs)
    steps = []
    for i, raw in enumerate(raw_steps):
        if not isinstance(raw, dict):
            raise PlanError("step %d is not an object" % (i + 1))
        kind = str(raw.get("kind", "")).strip().lower()
        if kind == "run_program":
            name = str(raw.get("name", "")).strip().lower()
            if not _NAME_RE.match(name):
                raise PlanError("step %d: bad program name %r" % (i + 1, name))
            if name not in allowed:
                raise PlanError("step %d: program %r not in catalog %s" %
                                (i + 1, name, sorted(allowed)))
            steps.append(Step("run_program", {"name": name}))
        elif kind == "goto":
            x = _num(raw.get("x"), "step %d goto.x" % (i + 1))
            y = _num(raw.get("y"), "step %d goto.y" % (i + 1))
            if abs(x) > MAX_COORD or abs(y) > MAX_COORD:
                raise PlanError("step %d: goto out of range (±%.0f m)" % (i + 1, MAX_COORD))
            mode = str(raw.get("mode", "move")).strip().lower()
            if mode not in ("move", "push"):
                raise PlanError("step %d: goto mode must be move|push" % (i + 1))
            steps.append(Step("goto", {"x": x, "y": y, "mode": mode}))
        elif kind == "dock":
            steps.append(Step("dock", {}))
        elif kind == "undock":
            steps.append(Step("undock", {}))
        elif kind == "wait":
            secs = _num(raw.get("secs", 1), "step %d wait.secs" % (i + 1))
            if not (0 < secs <= MAX_WAIT):
                raise PlanError("step %d: wait.secs must be in (0, %.0f]" % (i + 1, MAX_WAIT))
            steps.append(Step("wait", {"secs": secs}))
        else:
            raise PlanError("step %d: unknown kind %r" % (i + 1, kind))
    if not goal:
        goal = "(mission)"
    return Plan(goal, rationale, steps)
