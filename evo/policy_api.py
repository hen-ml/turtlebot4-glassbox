"""The contract between an evolved policy and whatever runs it (simulator or robot).

A policy is one Python file that defines either ``make_policy()`` or a class
``Policy``; the object has

    reset(goal)  goal is (x, y) in odom, the point the ball should end up at
    act(obs)     returns (v, w) or (v, w, done): m/s forward, rad/s left-positive

``obs`` (Obs below) carries only what the robot has: odometry (x, y, yaw), the
latest ball detection (or None if never seen): the ball's centre in the robot
frame, its range from the camera and its age in seconds; the goal; a front-bumper
flag (pressed during the last step); (v, w) measured from odometry over the
last step; and the latest 2-D lidar scan (walls and obstacles; the ball only if it
reaches the scan plane). The simulator produces exactly this, and the robot's runner
(turtlebot4-glassbox policy_runner.py) converts its sensors into it. Commands are
clamped by the runner; a policy cannot exceed the speed limits.

Optionally, the object may expose ``belief`` (a dict, or a method returning one)
to say what it currently believes; the robot's runner reads it after every
``act()``, logs it and draws it on the operator's map. It never affects control,
and a runner that does not know it (the simulator) simply ignores it:

    ball    (x, y) in odom, where the policy thinks the ball is (None = no idea)
    ball_t  obs.t at which it last updated that estimate (shown as its age)
    mode    a short label for what it is doing ("search", "push", ...)

Policy files are LLM-written, so ``load_policy`` checks them before running:
imports must come from ``ALLOWED_IMPORTS`` and a list of builtins/dunders that
would reach the file system or the interpreter is refused. This is a guard
against accidents, not a security boundary; run untrusted code in the sandboxed
evaluator (evaluate.py), and only run vetted files on the robot.
"""
from __future__ import annotations

import ast
import builtins
import math
import types
from dataclasses import dataclass

ALLOWED_IMPORTS = {"math", "random", "collections", "dataclasses", "functools",
                   "itertools", "evo.policy_lib", "evo"}
BANNED_NAMES = {"open", "exec", "eval", "compile", "__import__", "globals", "locals",
                "vars", "input", "breakpoint", "getattr", "setattr", "delattr",
                "memoryview", "exit", "quit", "help", "__builtins__", "__loader__",
                "__spec__"}
ALLOWED_DUNDER_ATTRS = {"__init__", "__name__"}
_SAFE_BUILTINS = {n: getattr(builtins, n) for n in (
    "abs", "all", "any", "bool", "dict", "enumerate", "filter", "float", "int", "isinstance",
    "len", "list", "map", "max", "min", "pow", "range", "reversed", "round", "set", "sorted",
    "str", "sum", "tuple", "zip", "Exception", "ValueError", "ZeroDivisionError", "object",
    "super", "property", "staticmethod", "classmethod", "print", "hasattr", "type",
    "frozenset", "divmod", "iter", "next", "slice", "NotImplementedError", "RuntimeError",
    "KeyError", "IndexError", "TypeError", "AttributeError", "ArithmeticError", "True",
    "False", "None") if hasattr(builtins, n)}
_SAFE_BUILTINS["__build_class__"] = builtins.__build_class__


@dataclass
class Obs:
    t: float                 # s since the episode started
    x: float                 # odom pose
    y: float
    yaw: float
    ball: dict | None        # {x_loc, y_loc, dist, age} robot frame at capture; None = never seen
    goal: tuple              # (x, y) odom
    bump: bool               # front bumper pressed during the last step (wall OR ball)
    v: float = 0.0           # forward speed measured from odometry over the last step
    w: float = 0.0           # yaw rate measured from odometry over the last step
    scan: tuple | None = None  # lidar ranges (m), beam i at bearing -pi + 2*pi*i/len(scan)
                               # in the robot frame (beam 0 straight back, len/2 straight ahead,
                               # counter-clockwise); lidar max = nothing seen. None until the first scan
    scan_age: float = 0.0    # s since that scan


class PolicyRejected(Exception):
    pass


def check_source(src: str) -> None:
    """Raise PolicyRejected if `src` imports or touches something it should not."""
    try:
        tree = ast.parse(src)
    except SyntaxError as e:
        raise PolicyRejected(f"syntax error: {e}") from e
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.name not in ALLOWED_IMPORTS:
                    raise PolicyRejected(f"import {a.name} not allowed")
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            if mod not in ALLOWED_IMPORTS or node.level:
                raise PolicyRejected(f"from {mod} import ... not allowed")
            if mod == "evo" and any(a.name != "policy_lib" for a in node.names):
                raise PolicyRejected("only `from evo import policy_lib` is allowed")
        elif isinstance(node, ast.Name) and node.id in BANNED_NAMES:
            raise PolicyRejected(f"name {node.id} not allowed")
        elif isinstance(node, ast.Attribute) and node.attr.startswith("__") \
                and node.attr not in ALLOWED_DUNDER_ATTRS:
            raise PolicyRejected(f"attribute {node.attr} not allowed")
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            pass
    names = {n.name for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
    if "make_policy" not in names and "Policy" not in names:
        raise PolicyRejected("define make_policy() or class Policy")


def _guarded_import(name, globals=None, locals=None, fromlist=(), level=0):  # noqa: A002
    if level or name not in ALLOWED_IMPORTS:
        raise ImportError(f"import {name} not allowed in a policy")
    return builtins.__import__(name, globals, locals, fromlist, level)


def load_policy_source(src: str, name: str = "policy"):
    """Check, exec and instantiate a policy from source text."""
    check_source(src)
    mod = types.ModuleType(name)
    safe = dict(_SAFE_BUILTINS)
    safe["__import__"] = _guarded_import
    mod.__dict__["__builtins__"] = safe
    mod.__dict__["__name__"] = name
    exec(compile(src, f"<{name}>", "exec"), mod.__dict__)  # noqa: S102 (checked above)
    if "make_policy" in mod.__dict__:
        pol = mod.__dict__["make_policy"]()
    else:
        pol = mod.__dict__["Policy"]()
    for m in ("reset", "act"):
        if not callable(getattr(pol, m, None)):
            raise PolicyRejected(f"policy object has no {m}()")
    return pol


def load_policy(path: str):
    with open(path) as f:
        return load_policy_source(f.read(), name=path.rsplit("/", 1)[-1].removesuffix(".py"))


def clamp_cmd(out, v_max: float, w_max: float):
    """Normalise a policy's return value to (v, w, done) inside the limits."""
    done = False
    if isinstance(out, dict):
        v, w, done = out.get("v", 0.0), out.get("w", 0.0), bool(out.get("done", False))
    elif isinstance(out, (tuple, list)) and len(out) >= 2:
        v, w = out[0], out[1]
        done = bool(out[2]) if len(out) > 2 else False
    else:
        raise TypeError(f"act() must return (v, w) or (v, w, done), got {type(out).__name__}")
    v, w = float(v), float(w)
    if not math.isfinite(v):
        v = 0.0
    if not math.isfinite(w):
        w = 0.0
    return max(-v_max, min(v_max, v)), max(-w_max, min(w_max, w)), done
