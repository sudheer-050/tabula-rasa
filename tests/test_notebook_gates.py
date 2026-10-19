"""The two fail-fast cells added to the notebook, exercised with the failures seen on Kaggle."""
import json
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from build_notebook import GPU_CHECK_CELL, SERVER_GATE_CELL  # noqa: E402


def run_gpu_cell(monkeypatch, smi_output):
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: SimpleNamespace(stdout=smi_output, returncode=0))
    exec(compile(GPU_CHECK_CELL, "<gpu cell>", "exec"), {})


def test_t4_is_rejected_with_a_clear_message(monkeypatch):
    with pytest.raises(RuntimeError, match="Wrong accelerator.*Tesla T4.*RTX PRO 6000"):
        run_gpu_cell(monkeypatch, "Tesla T4, 15360\n")


def test_no_gpu_is_rejected(monkeypatch):
    with pytest.raises(RuntimeError, match="no GPU"):
        run_gpu_cell(monkeypatch, "")


def test_rtx_pro_6000_is_accepted(monkeypatch):
    run_gpu_cell(monkeypatch, "NVIDIA RTX PRO 6000 Blackwell Server Edition, 97887\n")


def test_any_96gb_card_is_accepted_even_if_the_name_differs(monkeypatch):
    run_gpu_cell(monkeypatch, "Some Future Card, 98304\n")


def serve(handler_cls):
    srv = HTTPServer(("127.0.0.1", 0), handler_cls)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


class Good(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_POST(self):
        self.rfile.read(int(self.headers["Content-Length"]))
        data = json.dumps({"choices": [{"message": {"role": "assistant", "content": "OK"}}]}).encode()
        self.send_response(200); self.send_header("Content-Length", str(len(data))); self.end_headers(); self.wfile.write(data)


def gate_namespace(base_url, tmp_path):
    return {"SERVER_BASE_URL": base_url, "SERVED_MODEL_NAME": "flashnext", "time": SimpleNamespace(sleep=lambda s: None),
            "WORKING_DIR": tmp_path}


def test_server_gate_passes_against_a_live_server(tmp_path):
    srv = serve(Good)
    try:
        exec(compile(SERVER_GATE_CELL, "<gate>", "exec"), gate_namespace(f"http://127.0.0.1:{srv.server_port}/v1", tmp_path))
    finally:
        srv.shutdown()


def test_server_gate_fails_loudly_and_shows_the_server_log(tmp_path):
    (tmp_path / "serve.log").write_text("RuntimeError: cudaHostRegister failed\n")
    ns = gate_namespace("http://127.0.0.1:9/v1", tmp_path)       # nothing listens on port 9
    ns["Path"] = Path
    with pytest.raises(RuntimeError, match="(?s)not answering.*cudaHostRegister failed"):
        exec(compile(SERVER_GATE_CELL, "<gate>", "exec"), ns)


def test_server_that_returns_http_500_is_rejected(tmp_path):
    class Bad(Good):
        def do_POST(self):
            self.rfile.read(int(self.headers["Content-Length"]))
            self.send_response(500); self.end_headers()

    srv = serve(Bad)
    try:
        with pytest.raises(RuntimeError, match="not answering"):
            exec(compile(SERVER_GATE_CELL, "<gate>", "exec"), gate_namespace(f"http://127.0.0.1:{srv.server_port}/v1", tmp_path))
    finally:
        srv.shutdown()
