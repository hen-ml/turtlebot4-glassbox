"""Seed policy: the obvious first attempt, kept deliberately simple.

find the ball -> drive to a stance behind it on the ball->goal line -> face the
goal -> push forward while steering at the ball -> stop when the ball estimate
is inside the goal radius.
"""
import math

from evo import policy_lib as L

STANDOFF = 0.55      # m behind the ball to line up
STANCE_TOL = 0.15    # m
ALIGN_TOL = 0.15     # rad
PUSH_V = 0.14        # m/s
TURN_K = 1.5         # rad/s per rad of heading error
DONE_R = 0.25        # m, ball estimate this close to the goal -> done
SEARCH_W = 0.4       # rad/s spin while searching


class Policy:
    def reset(self, goal):
        self.goal = goal
        self.ball = None          # last ball estimate (odom)
        self.ball_t = None        # obs.t of that estimate
        self.mode = "search"
        self.belief = dict(ball=None, ball_t=None, mode=self.mode)

    def act(self, obs):
        b = L.ball_odom(obs)
        if b is not None and obs.ball["age"] < 1.0:
            self.ball, self.ball_t = b, obs.t
        out = self._decide(obs)
        # Optional self-report (evo.policy_api `belief`): the robot logs it and
        # draws it on the map; it does not change what this controller does.
        self.belief = dict(ball=self.ball, ball_t=self.ball_t, mode=self.mode)
        return out

    def _decide(self, obs):
        if self.ball is None:
            return 0.0, SEARCH_W
        if L.dist(self.ball, self.goal) < DONE_R:
            return 0.0, 0.0, True

        robot = (obs.x, obs.y)
        stance = L.push_stance(self.ball, self.goal, STANDOFF)
        along, lateral = L.stance_error(robot, self.ball, self.goal)

        if self.mode == "push":
            if along > -0.15 or lateral > 0.25:      # lost the line: re-line-up
                self.mode = "approach"
            else:
                err = L.heading_to(obs.x, obs.y, obs.yaw, self.ball)
                return PUSH_V, L.clamp(TURN_K * err, -0.8, 0.8)

        if L.dist(robot, stance) > STANCE_TOL:
            self.mode = "approach"
            err = L.heading_to(obs.x, obs.y, obs.yaw, stance)
            v = 0.25 * max(0.0, math.cos(err)) if abs(err) < 0.6 else 0.0
            return v, L.clamp(TURN_K * err, -0.8, 0.8)

        err = L.heading_to(obs.x, obs.y, obs.yaw, self.goal)
        if abs(err) > ALIGN_TOL:
            return 0.0, L.clamp(TURN_K * err, -0.8, 0.8)
        self.mode = "push"
        return PUSH_V, 0.0


def make_policy():
    return Policy()
