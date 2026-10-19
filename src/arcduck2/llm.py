"""OpenAI-compatible chat client (SGLang / vLLM / OpenRouter).

Blocking by design: the agent calls it through ``asyncio.to_thread``. Retries
cover server start-up (connection refused, 5xx, 429); a 4xx that names the
context window raises ``ContextTooLong`` so the caller can evict history.
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import requests

from .config import Config

PYTHON_TOOL = {
    "type": "function",
    "function": {
        "name": "python",
        "description": (
            "Run Python against the live game. Variables grid, prev_grid, levels_completed, "
            "level_actions, available_actions are fresh each call; things you define persist. "
            "action(name, x=None, y=None) executes a real game action."
        ),
        "parameters": {
            "type": "object",
            "properties": {"code": {"type": "string", "description": "Python source to execute."}},
            "required": ["code"],
        },
    },
}

_FENCE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL)


class ContextTooLong(Exception):
    pass


class ModelUnavailable(Exception):
    pass


@dataclass
class ToolCall:
    id: str
    name: str
    code: str


@dataclass
class LLMReply:
    content: str = ""
    reasoning: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    finish_reason: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_tokens: int = 0
    raw_tool_calls: list[dict] = field(default_factory=list)

    @property
    def code(self) -> str:
        """Code to run: the first tool call, else a fenced block in the text."""
        if self.tool_calls:
            return self.tool_calls[0].code
        match = _FENCE.search(self.content or "")
        return match.group(1).strip("\n") if match else ""


def parse_tool_arguments(raw: Any) -> str:
    if isinstance(raw, dict):
        return str(raw.get("code", ""))
    if not isinstance(raw, str):
        return ""
    try:
        obj = json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        return raw
    if isinstance(obj, dict):
        return str(obj.get("code", ""))
    return str(obj) if isinstance(obj, str) else ""


def parse_response(data: dict) -> LLMReply:
    choice = (data.get("choices") or [{}])[0]
    msg = choice.get("message") or {}
    reply = LLMReply(
        content=msg.get("content") or "",
        reasoning=msg.get("reasoning_content") or msg.get("reasoning") or "",
        finish_reason=choice.get("finish_reason") or "",
    )
    for i, tc in enumerate(msg.get("tool_calls") or []):
        fn = tc.get("function") or {}
        reply.tool_calls.append(ToolCall(tc.get("id") or f"call_{i}", fn.get("name") or "python", parse_tool_arguments(fn.get("arguments"))))
    reply.raw_tool_calls = list(msg.get("tool_calls") or [])
    usage = data.get("usage") or {}
    reply.prompt_tokens = int(usage.get("prompt_tokens") or 0)
    reply.completion_tokens = int(usage.get("completion_tokens") or 0)
    details = usage.get("prompt_tokens_details") or {}
    reply.cached_tokens = int(details.get("cached_tokens") or 0)
    return reply


class LLMClient:
    def __init__(self, cfg: Config, post: Callable[..., Any] | None = None, sleep: Callable[[float], None] = time.sleep,
                 max_wait_s: float = 1800.0):
        self.cfg = cfg
        self._post = post or requests.post
        self._sleep = sleep
        self.max_wait_s = max_wait_s

    def chat(self, messages: list[dict], max_tokens: int | None = None) -> LLMReply:
        payload = {
            "model": self.cfg.model,
            "messages": messages,
            "tools": [PYTHON_TOOL],
            "tool_choice": "auto",
            "temperature": self.cfg.temperature,
            "top_p": self.cfg.top_p,
            "top_k": self.cfg.top_k,
            "max_tokens": max_tokens or self.cfg.max_output_tokens,
            "chat_template_kwargs": {"preserve_thinking": True},
        }
        headers = {"Authorization": f"Bearer {self.cfg.api_key}", "Content-Type": "application/json"}
        url = f"{self.cfg.base_url}/chat/completions"
        deadline = time.monotonic() + self.max_wait_s
        delay = 2.0
        last = ""
        while True:
            try:
                resp = self._post(url, headers=headers, data=json.dumps(payload), timeout=self.cfg.request_timeout_s)
            except (requests.ConnectionError, requests.Timeout) as exc:
                last = f"{type(exc).__name__}: {exc}"
            else:
                status = resp.status_code
                if status == 200:
                    try:
                        return parse_response(resp.json())
                    except ValueError as exc:
                        last = f"bad JSON from server: {exc}"
                elif status == 400 or status == 413 or status == 422:
                    text = (resp.text or "")[:500].lower()
                    if any(k in text for k in ("context", "too long", "maximum", "exceed", "length")):
                        raise ContextTooLong(text)
                    raise ModelUnavailable(f"HTTP {status}: {text}")
                elif status in (401, 403, 404):
                    raise ModelUnavailable(f"HTTP {status}: {(resp.text or '')[:200]}")
                else:
                    last = f"HTTP {status}: {(resp.text or '')[:200]}"
            if time.monotonic() >= deadline:
                raise ModelUnavailable(f"gave up after {self.max_wait_s:.0f}s: {last}")
            self._sleep(delay)
            delay = min(delay * 1.7, 30.0)
