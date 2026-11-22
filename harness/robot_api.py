"""TurtleBot 4 adapter — thin HTTP client over the navigator's existing
Flask API (tb4_claude_nav.py). The robot is unchanged; this speaks to it
exactly the way the web UI does.

Stdlib only. Inject `fetcher` for tests.
"""
import json
import urllib.request
import urllib.error


class RobotError(RuntimeError):
    pass


def _http(url, payload, timeout, method=None):
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    req = urllib.request.Request(url, data=data, method=method or ("POST" if data else "GET"),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8", errors="replace")
            return resp.status, body
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace") if e.fp else ""
        return e.code, body
    except Exception as e:
        raise RobotError(f"robot request failed: {e}") from e


class TB4Robot(object):
    def __init__(self, base_url, timeout=8.0, fetcher=None):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._fetch = fetcher or _http

    # ── endpoints ────────────────────────────────────────────────────────
    def get_state(self):
        status, body = self._fetch(self.base_url + "/state", None, self.timeout)
        if status != 200:
            raise RobotError(f"/state HTTP {status}: {body[:200]}")
        try:
            return json.loads(body)
        except ValueError as e:
            raise RobotError(f"/state returned non-JSON: {e}") from e

    def list_programs(self):
        status, body = self._fetch(self.base_url + "/programs", None, self.timeout)
        try:
            progs = json.loads(body)
            return [p.get("name", "") for p in progs if isinstance(p, dict)]
        except ValueError:
            return []

    def health(self):
        """Cheap pre-flight: are cam/lidar/base fresh? (3 green = go.)"""
        st = self.get_state()
        return bool(st.get("cam_ok") and st.get("lidar_ok") and st.get("base_ok")), st

    def run_program(self, name):
        status, body = self._fetch(self.base_url + "/run", {"program": name}, self.timeout)
        ok = status == 200
        try:
            data = json.loads(body)
        except ValueError:
            data = {"raw": body[:300]}
        if not ok:
            raise RobotError(f"/run {name} → HTTP {status}: {body[:200]}")
        return data

    def goto_xy(self, x, y, mode="move"):
        status, body = self._fetch(self.base_url + "/goto_xy",
                                   {"x": float(x), "y": float(y), "mode": mode}, self.timeout)
        if status != 200:
            raise RobotError(f"/goto_xy → HTTP {status}: {body[:200]}")

    def run_stop(self):
        try:
            self._fetch(self.base_url + "/run_stop", {}, self.timeout)
            self._fetch(self.base_url + "/stop", {}, self.timeout)
        except RobotError:
            pass  # stop must never throw on the way out

    def dock(self):
        status, body = self._fetch(self.base_url + "/dock", {}, self.timeout)
        if status != 200:
            raise RobotError(f"/dock → HTTP {status}: {body[:200]}")

    def undock(self):
        status, body = self._fetch(self.base_url + "/undock", {}, self.timeout)
        if status != 200:
            raise RobotError(f"/undock → HTTP {status}: {body[:200]}")
