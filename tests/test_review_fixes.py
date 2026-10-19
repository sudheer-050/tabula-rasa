"""Regression tests for problems found while reviewing the finished agent loop."""
import asyncio
import time
from dataclasses import replace

from fakes import FakeGame, ScriptedLLM, code_reply
from arcduck2.agent import GameAgent
from arcduck2.config import Config
from arcduck2.llm import ModelUnavailable
from arcduck2.solver import DuckV2Solver

STEP = "action('ACTION1')"
WIN = "action('ACTION1')\naction('ACTION1')\naction('ACTION1')"


def cfg(**kw):
    return replace(Config(), tool_timeout_s=3.0, **kw)


def count_images(msgs):
    return sum(1 for m in msgs if isinstance(m.get("content"), list)
               for p in m["content"] if p.get("type") == "image_url")


async def test_only_the_newest_board_image_is_sent():
    g = FakeGame(levels=2)
    # ACTION2 moves left, so we can take many board-changing steps without finishing the level
    llm = ScriptedLLM(["action('ACTION1')\naction('ACTION2')"] * 5 + [WIN, WIN])
    a = GameAgent(g, cfg(), llm)
    await asyncio.wait_for(a.run(), 30)
    for i in range(2, len(llm.seen)):
        assert count_images(llm.seen[i]) <= 1, f"call {i} carried {count_images(llm.seen[i])} images"
    last = [m for m in llm.seen[-1] if m["role"] == "user"]
    assert any("[board image omitted]" in str(m["content"]) for m in last)


async def test_stripping_old_images_leaves_the_prefix_stable():
    """The cached prefix must not change from one call to the next, except at its tail."""
    g = FakeGame(levels=2)
    llm = ScriptedLLM(["action('ACTION1')\naction('ACTION2')"] * 6 + [WIN, WIN])
    # Deliberately tiny batches to build history for the cache-stability check below;
    # not what forced-commitment is meant to catch, so disable it for this test.
    a = GameAgent(g, cfg(probe_streak_limit=100), llm)
    await asyncio.wait_for(a.run(), 30)
    import json
    for i in range(3, len(llm.seen) - 1):
        prev, cur = llm.seen[i], llm.seen[i + 1]
        stable = 0
        for x, y in zip(prev, cur):
            if json.dumps(x, sort_keys=True) != json.dumps(y, sort_keys=True):
                break
            stable += 1
        # everything except the last two messages of the previous prompt is byte-identical
        assert stable >= len(prev) - 3, f"call {i}->{i + 1}: only {stable}/{len(prev)} leading messages unchanged"


async def test_http_400_recovery_strips_images_and_continues():
    g = FakeGame(levels=1)
    llm = ScriptedLLM(lambda msgs, i: ModelUnavailable("HTTP 400: image rejected") if i == 0 else WIN)
    a = GameAgent(g, cfg(), llm)
    await asyncio.wait_for(a.run(), 30)
    assert g.game_run.state == "won"
    assert count_images(llm.seen[1]) == 0            # the retry went out without images


async def test_waiting_games_do_not_start_turns_after_the_deadline():
    class SlowLLM:
        def __init__(self):
            self.calls = 0

        def chat(self, messages, max_tokens=None):
            self.calls += 1
            time.sleep(0.4)
            return code_reply("print('thinking')")

    games = [FakeGame() for _ in range(12)]
    llm = SlowLLM()
    s = DuckV2Solver(config=replace(Config(), slots=1, tool_timeout_s=3.0), hard_deadline_epoch=time.time() + 1.5)
    s.setup()
    s._llm = llm
    t0 = time.time()
    await asyncio.wait_for(s.run_games(games), 60)
    elapsed = time.time() - t0
    s.teardown()
    # one slot, 0.4s per call: without the fix all 12 games would each run a turn after the
    # deadline (>= 12 * 0.4 = 4.8s). With it the run stops within about one extra call.
    assert elapsed < 3.5, f"run overshot its deadline: {elapsed:.1f}s"
    assert llm.calls <= 6
    assert all(g.game_run.final_score is not None for g in games)


async def test_auto_reset_does_not_block_the_event_loop():
    """The engine call for the auto-RESET must run off the loop thread."""
    import threading
    seen = {}

    class ThreadSpyGame(FakeGame):
        def execute_action(self, action, generated_tokens=0, uncached_input_tokens=0):
            if action.id.name == "RESET":
                seen["thread"] = threading.current_thread().name
            return super().execute_action(action, generated_tokens, uncached_input_tokens)

    g = ThreadSpyGame(levels=1)
    llm = ScriptedLLM(["action('ACTION3')", "print('a')", "print('b')", WIN])
    a = GameAgent(g, cfg(), llm)
    await asyncio.wait_for(a.run(), 30)
    assert seen["thread"] != threading.main_thread().name
