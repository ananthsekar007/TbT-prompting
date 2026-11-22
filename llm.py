import hashlib
import json
import os
import random
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DEFAULT_BASE_URL = "https://openai.rc.asu.edu/v1"


def load_env(path=ROOT / ".env"):
    if not Path(path).exists():
        return
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


class LLMClient:
    """Thin wrapper over the OpenAI-compatible gateway with retries, an on-disk cache
    and a per-case record of every call (so the runner can log prompts/completions)."""

    def __init__(self, model, max_tokens=4000, reasoning_effort="none", cache_dir=ROOT / "cache", max_concurrent=6,
                 request_timeout=900):
        from openai import OpenAI

        load_env()
        base_url = os.environ.get("OPENAI_BASE_URL") or os.environ.get("YODB_LLM_ENDPOINT") or DEFAULT_BASE_URL
        self.client = OpenAI(base_url=base_url, api_key=os.environ["OPENAI_API_KEY"], timeout=request_timeout, max_retries=0)
        self.base_url = base_url
        self.model = model
        self.max_tokens = max_tokens
        self.reasoning_effort = reasoning_effort
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self._slots = threading.Semaphore(max_concurrent)

    @property
    def records(self):
        if not hasattr(self._local, "records"):
            self._local.records = []
        return self._local.records

    @property
    def notes(self):
        if not hasattr(self._local, "notes"):
            self._local.notes = {}
        return self._local.notes

    def reset(self):
        self._local.records = []
        self._local.notes = {}

    def chat(self, messages, temperature=0.0, n=1, tag=""):
        params = {"model": self.model, "temperature": temperature, "n": n,
                  "max_tokens": self.max_tokens, "reasoning_effort": self.reasoning_effort}
        key = hashlib.sha256(json.dumps([messages, params, tag], sort_keys=True).encode()).hexdigest()
        cache_file = self.cache_dir / f"{key}.json"

        if cache_file.exists():
            result = json.loads(cache_file.read_text())
            result["cached"] = True
        else:
            with self._slots:
                result = self._call(messages, params)
            cache_file.write_text(json.dumps(result))
            result["cached"] = False

        self.records.append({"messages": messages, "params": params, "tag": tag, **result})
        return result

    def _call(self, messages, params, attempts=8):
        delay = 5
        for attempt in range(attempts):
            start = time.time()
            try:
                raw = self.client.chat.completions.with_raw_response.create(
                    model=params["model"],
                    messages=messages,
                    temperature=params["temperature"],
                    n=params["n"],
                    max_tokens=params["max_tokens"],
                    extra_body={"reasoning_effort": params["reasoning_effort"]},
                    extra_headers={"x-obleth-boons": "off"},
                )
                resp = raw.parse()
                choices = []
                for ch in resp.choices:
                    msg = ch.message
                    choices.append({
                        "content": msg.content or "",
                        "reasoning": getattr(msg, "reasoning", None) or (msg.model_extra or {}).get("reasoning"),
                        "finish_reason": ch.finish_reason,
                    })
                usage = resp.usage
                return {
                    "choices": choices,
                    "usage": {"prompt_tokens": usage.prompt_tokens if usage else 0,
                              "completion_tokens": usage.completion_tokens if usage else 0},
                    "latency_s": time.time() - start,
                    "response_model": resp.model,
                    "reasoning_effort_header": raw.headers.get("x-obleth-reasoning-effort"),
                }
            except Exception as e:
                status = getattr(e, "status_code", None)
                retryable = status in (408, 409, 429, 500, 502, 503, 504) or status is None
                if not retryable or attempt == attempts - 1:
                    raise
                wait = delay * (2 ** attempt) + random.uniform(0, 3)
                print(f"  [llm] {type(e).__name__} ({status}); retrying in {wait:.0f}s", flush=True)
                time.sleep(min(wait, 300))
