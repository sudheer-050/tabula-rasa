"""Regression tests for the failures seen on Kaggle's pinned image: broken Pillow / scipy."""
import asyncio
import io
import subprocess
import sys
import textwrap
import zlib
from dataclasses import replace
from pathlib import Path

import numpy as np

from fakes import FakeGame, ScriptedLLM
from arcduck2 import perception as P
from arcduck2.agent import GameAgent
from arcduck2.config import Config

SRC = str(Path(__file__).resolve().parents[1] / "src")
WIN = "action('ACTION1')\naction('ACTION1')\naction('ACTION1')"


def decode_png(data: bytes) -> np.ndarray:
    """Independent minimal decoder (filter 0 only) so the encoder is checked without any library."""
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    w = h = 0
    pos, idat = 8, b""
    while pos < len(data):
        n = int.from_bytes(data[pos:pos + 4], "big")
        tag, body = data[pos + 4:pos + 8], data[pos + 8:pos + 8 + n]
        crc = int.from_bytes(data[pos + 8 + n:pos + 12 + n], "big")
        assert zlib.crc32(tag + body) & 0xFFFFFFFF == crc, f"bad CRC in {tag}"
        if tag == b"IHDR":
            w, h = int.from_bytes(body[:4], "big"), int.from_bytes(body[4:8], "big")
            assert body[8:] == bytes([8, 2, 0, 0, 0])
        elif tag == b"IDAT":
            idat += body
        pos += 12 + n
    raw = np.frombuffer(zlib.decompress(idat), dtype=np.uint8).reshape(h, w * 3 + 1)
    assert (raw[:, 0] == 0).all()
    return raw[:, 1:].reshape(h, w, 3)


def test_png_is_pixel_exact_without_any_image_library():
    g = np.zeros((64, 64), dtype=np.int8)
    g[10:13, 20:23] = 8
    g[40, 5] = 14
    img = decode_png(P.render_png(g, scale=10, rulers=False))
    assert img.shape == (640, 640, 3)
    assert tuple(img[105, 215]) == P.ARC_RGB[8]                 # inside the red block
    assert tuple(img[405, 55]) == P.ARC_RGB[14]
    assert tuple(img[0, 0]) == P.ARC_RGB[0]


def test_png_diff_outline_and_rulers():
    a = np.zeros((64, 64), dtype=np.int8)
    b = a.copy()
    b[5, 5] = 9
    img = decode_png(P.render_png(b, scale=10, diff_against=a, rulers=True))
    pad = max(14, 10 + 4)
    assert tuple(img[pad + 50, pad + 50]) == (255, 0, 255)      # outline corner of the changed cell
    assert tuple(img[pad + 55, pad + 55]) == P.ARC_RGB[9]       # its interior keeps the colour
    assert tuple(img[0, pad]) == (0, 0, 0)                      # x tick at 0


def test_pillow_not_imported_and_works_when_pillow_is_broken():
    code = textwrap.dedent(f"""
        import sys
        sys.path.insert(0, {SRC!r})
        class Broken:                      # emulate Kaggle's mixed-version PIL
            def find_spec(self, name, path=None, target=None):
                if name == "PIL" or name.startswith("PIL."):
                    raise ImportError("cannot import name '_Ink' from 'PIL._typing'")
        sys.meta_path.insert(0, Broken())
        import numpy as np
        from arcduck2 import perception as P
        png = P.render_png(np.zeros((64, 64), dtype=np.int8), scale=10, diff_against=np.ones((64, 64), dtype=np.int8))
        assert png[:8] == b"\\x89PNG\\r\\n\\x1a\\n" and "PIL" not in sys.modules
        print("ok", len(png))
    """)
    out = subprocess.run([sys.executable, "-I", "-c", code], capture_output=True, text=True, timeout=60)
    assert out.returncode == 0 and "ok" in out.stdout, out.stderr


def test_components_fallback_matches_scipy():
    rng = np.random.default_rng(3)
    g = rng.integers(0, 4, size=(64, 64)).astype(np.int8)
    with_scipy = P.segments(g, background=0, limit=10000)
    code = textwrap.dedent(f"""
        import sys, json
        sys.path.insert(0, {SRC!r})
        class Broken:
            def find_spec(self, name, path=None, target=None):
                if name == "scipy" or name.startswith("scipy."):
                    raise ImportError("scipy is broken")
        sys.meta_path.insert(0, Broken())
        import numpy as np
        from arcduck2 import perception as P
        rng = np.random.default_rng(3)
        g = rng.integers(0, 4, size=(64, 64)).astype(np.int8)
        print(json.dumps(P.segments(g, background=0, limit=10000)))
    """)
    out = subprocess.run([sys.executable, "-I", "-c", code], capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr
    import json
    without = json.loads(out.stdout)
    key = lambda s: (s["color"], s["y0"], s["x0"], s["size"])
    assert sorted(map(key, with_scipy)) == sorted(map(key, without))
    assert len(with_scipy) > 50                                   # a non-trivial comparison


async def test_agent_survives_total_image_failure(monkeypatch):
    def boom(*a, **k):
        raise ImportError("cannot import name '_Ink' from 'PIL._typing'")

    monkeypatch.setattr(P, "render_png", boom)
    g = FakeGame(levels=1)
    llm = ScriptedLLM([WIN])
    a = GameAgent(g, replace(Config(), tool_timeout_s=3.0), llm)
    await asyncio.wait_for(a.run(), 30)
    assert g.game_run.state == "won"                              # the game is still played, text only
    assert all(isinstance(m["content"], str) for m in llm.seen[0] if m["role"] == "user")
