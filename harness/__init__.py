"""Laptop harness for turtlebot4-glassbox.

Turns a free-text mission into a validated plan of tool calls, confirms with
the human once, executes it against the robot's existing HTTP API, verifies
each leg, re-plans at most once on failure, and stops. Robot-agnostic core:
see robot_api.py for the TurtleBot 4 adapter (MicroDuck would be a sibling).

Stdlib only — no new dependencies on the laptop or the robot.
"""

__version__ = "0.1.0"
