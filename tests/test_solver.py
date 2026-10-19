import asyncio
import pickle
import time
from dataclasses import replace
import pytest
from fakes import FakeGame, ScriptedLLM
from arcduck2.config import Config
from arcduck2.solver import DuckV2Solver

WIN = "action('ACTION1')\naction('ACTION1')\naction('ACTION1')"


def make_solver(llm, **kw):
    cfg = replace(Config(), slots=2, tool_timeout_s=3.0, **kw.pop("cfg", {}))
    s = DuckV2Solver(config=cfg, **kw)
    s.setup()
    s._llm = llm
    return s


async def test_many_games_through_few_slots():
    games = [FakeGame() for _ in range(6)]
    s = make_solver(ScriptedLLM([WIN]))
    await asyncio.wait_for(s.run_games(games), 60)
    assert all(g.game_run.state == "won" for g in games)
    assert all(g.game_run.final_score == 2.0 for g in games)
    s.teardown()


async def test_hard_deadline_winds_down_all_games():
    games = [FakeGame() for _ in range(4)]
    s = make_solver(ScriptedLLM(["print('thinking forever')"]), hard_deadline_epoch=time.time() + 2.0)
    t = time.time()
    await asyncio.wait_for(s.run_games(games), 60)
    assert time.time() - t < 30
    assert all(g.game_run.state == "gave_up" and g.game_run.final_score is not None for g in games)
    assert all("deadline" in (g.game_run.solver_note or "") for g in games)
    s.teardown()


async def test_cancellation_finishes_every_game():
    games = [FakeGame() for _ in range(3)]
    s = make_solver(ScriptedLLM(["print('thinking forever')"]))
    task = asyncio.create_task(s.run_games(games))
    await asyncio.sleep(1.0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert all(g.game_run.final_score is not None for g in games)
    s.teardown()


async def test_one_crashing_game_does_not_sink_the_rest():
    class BoomGame(FakeGame):
        @property
        def current_state(self):
            raise RuntimeError("engine exploded")

    good, bad = FakeGame(), BoomGame()
    s = make_solver(ScriptedLLM([WIN]))
    await asyncio.wait_for(s.run_games([bad, good]), 30)
    assert good.game_run.state == "won"
    assert "agent init failed" in bad.game_run.solver_note and bad.game_run.final_score is not None
    s.teardown()


async def test_agent_crash_mid_game_is_contained():
    class FlakyLLM(ScriptedLLM):
        def chat(self, messages, max_tokens=None):
            if len(self.seen) == 1:
                self.seen.append(None)
                raise ValueError("unexpected server payload")
            return super().chat(messages, max_tokens)

    crash, good = FakeGame(), FakeGame()
    llms = {id(crash): FlakyLLM([WIN]), id(good): ScriptedLLM([WIN])}
    s = make_solver(ScriptedLLM([WIN]))

    class Router:
        def chat(self, messages, max_tokens=None):
            # route by which game the first user message belongs to is not possible; alternate instead
            Router.n += 1
            if Router.n == 2:
                raise ValueError("unexpected server payload")
            return ScriptedLLM([WIN]).chat(messages)

    Router.n = 0
    s._llm = Router()
    await asyncio.wait_for(s.run_games([crash, good]), 30)
    states = sorted([crash.game_run.state, good.game_run.state])
    assert "won" in states                       # the other game was unaffected
    assert all(g.game_run.final_score is not None for g in (crash, good))
    s.teardown()


def test_solver_pickles_without_runtime_objects():
    s = DuckV2Solver(config=replace(Config(), slots=3))
    s.setup()
    clone = pickle.loads(pickle.dumps(s))
    assert clone._llm is None and clone.agents == []
    s.teardown()
