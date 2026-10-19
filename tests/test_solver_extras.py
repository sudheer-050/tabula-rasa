import asyncio
from dataclasses import replace

from fakes import FakeGame, ScriptedLLM
from arcduck2.config import Config
from arcduck2.llm import LLMReply
from arcduck2.solver import DuckV2Solver


def make_solver(llm, **cfg_kw):
    cfg = replace(Config(), slots=2, tool_timeout_s=3.0, **cfg_kw)
    s = DuckV2Solver(config=cfg)
    s.setup()
    s._llm = llm
    return s


class SlowLLM:
    """Blocks (like a server still loading) until released, then plays a winning move."""

    def __init__(self, delay):
        self.delay = delay

    def chat(self, messages, max_tokens=None):
        import time
        time.sleep(self.delay)
        return ScriptedLLM(["action('ACTION1')\naction('ACTION1')\naction('ACTION1')"]).chat(messages)


async def test_warmup_reset_fires_only_when_nothing_acted():
    g = FakeGame(levels=1)
    s = make_solver(SlowLLM(2.5), warmup_after_s=1.0)
    await asyncio.wait_for(s.run_games([g]), 30)
    names = [n for n, _ in g.game_run.history]
    assert names[0] == "RESET"                     # liveness action issued during the slow start
    assert g.game_run.state == "won"
    s.teardown()


async def test_no_warmup_when_a_game_acts_in_time():
    g = FakeGame(levels=1)
    s = make_solver(ScriptedLLM(["action('ACTION1')\naction('ACTION1')\naction('ACTION1')"]), warmup_after_s=1.5)
    await asyncio.wait_for(s.run_games([g]), 30)
    assert [n for n, _ in g.game_run.history].count("RESET") == 0
    s.teardown()


async def test_status_line_and_loop_do_not_disturb_the_run(capsys):
    games = [FakeGame(levels=1) for _ in range(2)]
    s = make_solver(ScriptedLLM(["action('ACTION1')\naction('ACTION1')\naction('ACTION1')"]), status_every_s=0.2)
    await asyncio.wait_for(s.run_games(games), 30)
    assert all(g.game_run.state == "won" for g in games)
    from arcduck2.scheduler import PriorityScheduler
    line = s.status_line(PriorityScheduler(2, 10.0), 0.0)
    assert "games=2 won=2" in line and "levels_done=2" in line
    s.teardown()
