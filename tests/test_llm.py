import json
import pytest
import requests
from arcduck2.config import Config
from arcduck2.llm import LLMClient, ContextTooLong, ModelUnavailable, parse_response, parse_tool_arguments


class Resp:
    def __init__(self, status=200, body=None, text=""):
        self.status_code, self._body, self.text = status, body, text or json.dumps(body or {})

    def json(self):
        if self._body is None:
            raise ValueError("no json")
        return self._body


OK = {"choices": [{"finish_reason": "tool_calls", "message": {
    "content": "", "reasoning_content": "hmm",
    "tool_calls": [{"id": "c1", "function": {"name": "python", "arguments": '{"code": "print(1)"}'}}]}}],
    "usage": {"prompt_tokens": 100, "completion_tokens": 7, "prompt_tokens_details": {"cached_tokens": 60}}}


def test_parse_response_tool_call_and_usage():
    r = parse_response(OK)
    assert r.code == "print(1)" and r.reasoning == "hmm"
    assert (r.prompt_tokens, r.completion_tokens, r.cached_tokens) == (100, 7, 60)
    assert r.raw_tool_calls[0]["id"] == "c1"


def test_fenced_block_fallback_and_argument_shapes():
    r = parse_response({"choices": [{"message": {"content": "do it\n```python\nx=1\nprint(x)\n```"}}]})
    assert r.code == "x=1\nprint(x)" and not r.tool_calls
    assert parse_tool_arguments('{"code": "a"}') == "a"
    assert parse_tool_arguments("print(2)") == "print(2)"   # not JSON -> raw code
    assert parse_tool_arguments({"code": "b"}) == "b"
    assert parse_tool_arguments(None) == ""


def test_retries_then_succeeds():
    calls = []

    def post(url, **kw):
        calls.append(url)
        if len(calls) == 1:
            raise requests.ConnectionError("refused")
        if len(calls) == 2:
            return Resp(503, text="starting")
        return Resp(200, OK)

    sleeps = []
    c = LLMClient(Config(), post=post, sleep=sleeps.append)
    assert c.chat([{"role": "user", "content": "hi"}]).code == "print(1)"
    assert len(calls) == 3 and len(sleeps) == 2


def test_payload_contents():
    seen = {}

    def post(url, headers, data, timeout):
        seen.update(json.loads(data), url=url)
        return Resp(200, OK)

    LLMClient(Config(), post=post).chat([{"role": "user", "content": "x"}])
    assert seen["url"].endswith("/chat/completions")
    assert seen["tools"][0]["function"]["name"] == "python"
    assert seen["chat_template_kwargs"] == {"preserve_thinking": True}
    assert seen["top_k"] == 20 and seen["max_tokens"] == Config().max_output_tokens


def test_context_too_long_and_auth_errors():
    c = LLMClient(Config(), post=lambda *a, **k: Resp(400, text="maximum context length exceeded"), sleep=lambda s: None)
    with pytest.raises(ContextTooLong):
        c.chat([])
    c = LLMClient(Config(), post=lambda *a, **k: Resp(401, text="no"), sleep=lambda s: None)
    with pytest.raises(ModelUnavailable):
        c.chat([])


def test_gives_up_after_max_wait():
    c = LLMClient(Config(), post=lambda *a, **k: (_ for _ in ()).throw(requests.ConnectionError("x")),
                  sleep=lambda s: None, max_wait_s=0.0)
    with pytest.raises(ModelUnavailable):
        c.chat([])
