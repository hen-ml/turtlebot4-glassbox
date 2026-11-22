"""python -m harness "push the ball to the wall" [--pi URL] [--model NAME]
                                      [--dry-run] [--yes] [--deadline SECS]

Requires a robot reachable at --pi (or TB4_PI_URL), except for --dry-run,
which only compiles and validates the plan (needs just a local model server).
"""
import argparse
import sys

from .llm_client import LLMClient
from .mission import MissionHarness
from .robot_api import TB4Robot


def main(argv=None):
    ap = argparse.ArgumentParser(prog="harness", description=__doc__)
    ap.add_argument("mission", nargs="+", help="free-text mission, e.g. 'push the ball to the wall'")
    ap.add_argument("--pi", default=None, help="robot base URL (default: $TB4_PI_URL)")
    ap.add_argument("--llm-url", default=None, help="OpenAI-compatible /v1 base URL (default: $TB4_LLM_URL, http://localhost:1234/v1)")
    ap.add_argument("--model", default=None, help="model id (default: $TB4_LLM_MODEL)")
    ap.add_argument("--deadline", type=float, default=900.0)
    ap.add_argument("--dry-run", action="store_true", help="compile + validate only; robot untouched")
    ap.add_argument("--yes", action="store_true", help="skip the once-per-plan confirmation")
    args = ap.parse_args(argv)

    mission_text = " ".join(args.mission).strip()
    pi = args.pi or __import__("os").environ.get("TB4_PI_URL")

    llm = LLMClient.from_env()
    if args.llm_url:
        llm.base_url = args.llm_url.rstrip("/")
    if args.model:
        llm.model = args.model

    if args.dry_run:
        # Validate against the repo's program catalog; no robot needed.
        stub = TB4Robot(pi or "http://127.0.0.1:9", timeout=1.0,
                        fetcher=_Unreachable)
        harness = MissionHarness(stub, llm)
        plan = harness.compile(mission_text)
        print("\n" + plan.describe())
        print("\n(dry-run: nothing sent to the robot)")
        return 0

    if not pi:
        print("error: --pi or TB4_PI_URL required (e.g. http://<pi-ip>:5000)", file=sys.stderr)
        return 2

    robot = TB4Robot(pi)
    harness = MissionHarness(robot, llm)
    report = harness.run(mission_text, assume_yes=args.yes, deadline_s=args.deadline)
    return 0 if report["ok"] else 1


def _Unreachable(url, payload, timeout, method=None):
    return 0, "{}"


if __name__ == "__main__":
    sys.exit(main())
