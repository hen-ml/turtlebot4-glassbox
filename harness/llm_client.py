"""OpenAI-compatible chat client for a *local* model server (LM Studio,
Ollama, llama.cpp server — anything speaking /v1/chat/completions).

Stdlib only. Inject `fetcher` for tests. No API keys are used or accepted;
bind the server to localhost/LAN only.
"""
import json
import os
import time
import urllib.request
import urllib.error


class LLMError(RuntimeError):
    pass


def _http_post(url, payload, headers, timeout):
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=body, method="POST",
                                 headers={"Content-Type": "application/json", **headers})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8", errors="replace") if e.fp else ""
    except Exception as e:                                     # URLError, timeout, refused…
        raise LLMError(f"LLM request failed: {e}") from e


class LLMClient:
    def __init__(self, base_url, model, timeout=90.0, fetcher=None):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self._fetch = fetcher or _http_post
        self.last_raw = None          # kept for the glass-box log

    @classmethod
    def from_env(cls):
        base = os.environ.get("TB4_LLM_URL", "http://localhost:1234/v1")
        model = os.environ.get("TB4_LLM_MODEL", "qwen2.5-7b-instruct")
        return cls(base, model, timeout=float(os.environ.get("TB4_LLM_TIMEOUT", "90")))

    def chat(self, messages, *, json_mode=True, temperature=0.2, max_tokens=1500, retries=1):
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        url = self.base_url + "/chat/completions"
        err = None
        for attempt in range(retries + 1):
            status, body = self._fetch(url, payload, {}, self.timeout)
            if status >= 400:
                err = LLMError(f"LLM HTTP {status}: {body[:400]}")
                time.sleep(0.5 * (attempt + 1))
                continue
            try:
                data = json.loads(body)
                content = data["choices"][0]["message"]["content"]
            except (ValueError, KeyError, IndexError) as e:
                err = LLMError(f"malformed LLM response: {e}")
                time.sleep(0.5 * (attempt + 1))
                continue
            self.last_raw = content
            return content
        raise err or LLMError("LLM call failed")
