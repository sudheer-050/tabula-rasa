"""The agent talking to a real (local) OpenAI-compatible HTTP server over sockets."""
import asyncio
import json
import threading
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, HTTPServer

from fakes import FakeGame
from arcduck2.agent import GameAgent
from arcduck2.config import Config
from arcduck2.llm import LLMClient

CODE = "action('ACTION1')\naction('ACTION1')\naction('ACTION1')"


def make_server(state):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            state["requests"].append({"path": self.path, "auth": self.headers.get("Authorization"), "body": body})
            if len(state["requests"]) == 1:           # simulate a server that is still starting
                self.send_response(503); self.end_headers(); self.wfile.write(b"warming up"); return
            resp = {
                "choices": [{"finish_reason": "tool_calls", "message": {
                    "role": "assistant", "content": "", "reasoning_content": "plan: go right",
                    "tool_calls": [{"id": f"call_{len(state['requests'])}", "type": "function",
                                    "function": {"name": "python", "arguments": json.dumps({"code": CODE})}}]}}],
                "usage": {"prompt_tokens": 1234, "completion_tokens": 56, "prompt_tokens_details": {"cached_tokens": 1000}},
            }
            data = json.dumps(resp).encode()
            self.send_response(200); self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data))); self.end_headers(); self.wfile.write(data)

    srv = HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


async def test_agent_over_real_http():
    state = {"requests": []}
    srv = make_server(state)
    try:
        cfg = replace(Config(), base_url=f"http://127.0.0.1:{srv.server_port}/v1", api_key="k", tool_timeout_s=3.0)
        llm = LLMClient(cfg, sleep=lambda s: None)
        g = FakeGame(levels=1)
        a = GameAgent(g, cfg, llm)
        await asyncio.wait_for(a.run(), 30)
    finally:
        srv.shutdown()
    assert g.game_run.state == "won"
    assert len(state["requests"]) == 2                              # 503 retried once, then success
    first = state["requests"][1]
    assert first["path"] == "/v1/chat/completions" and first["auth"] == "Bearer k"
    body = first["body"]
    assert body["model"] == cfg.model and body["tools"][0]["function"]["name"] == "python"
    msgs = body["messages"]
    assert msgs[0]["role"] == "system" and msgs[1]["role"] == "user"
    # the intro carries both text and an image data URL
    parts = msgs[1]["content"]
    assert parts[0]["type"] == "text" and parts[1]["image_url"]["url"].startswith("data:image/png;base64,")
    assert a.stats.cached_tokens == 1000 and a.stats.generated_tokens == 56
