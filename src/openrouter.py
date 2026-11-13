"""Minimal OpenRouter client with the provider pinned on every call.

Standard library only. The original generation run went through litellm, which
sends the same request; calling the HTTP API directly keeps the provider
setting visible in one place.
"""
import json
import os
import time
import urllib.request
from pathlib import Path

URL = "https://openrouter.ai/api/v1/chat/completions"


def _load_dotenv(path=Path(__file__).resolve().parent.parent / ".env"):
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip("'\""))


def _api_key():
    _load_dotenv()
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise RuntimeError("OPENROUTER_API_KEY is not set (copy .env.example to .env).")
    return key


def chat(model, messages, providers, *, timeout, retries=3,
         require_parameters=False, **params):
    """One chat completion. Returns the decoded JSON response.

    `providers` is the preferred host order; the host that actually served the
    call is in the response's "provider" field and should be stored with it.
    `require_parameters` routes only to hosts that support every parameter sent
    (needed for logprobs).
    """
    provider = {"order": providers, "allow_fallbacks": True}
    if require_parameters:
        provider["require_parameters"] = True
    body = json.dumps({"model": model, "messages": messages,
                       "provider": provider, **params}).encode()
    headers = {"Authorization": f"Bearer {_api_key()}",
               "Content-Type": "application/json"}
    for attempt in range(retries):
        try:
            req = urllib.request.Request(URL, data=body, headers=headers)
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.load(r)
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(1.5 * (attempt + 1))
