import asyncio
from dataclasses import replace

from fakes import FakeGame, ScriptedLLM
from arcduck2.agent import GameAgent
from arcduck2.config import Config


def cfg(**kw):
    return replace(Config(), tool_timeout_s=3.0, **kw)


WIN = "action('ACTION1')\naction('ACTION1')\naction('ACTION1')"


async def test_auto_reset_after_idle_game_over():
    g = FakeGame(levels=1)
    llm = ScriptedLLM([
        "action('ACTION3')",                         # dies
        "print('thinking about the death')",         # 2 turns of analysis, never resets
        "print('still thinking')",
        WIN,
    ])
    a = GameAgent(g, cfg(), llm)
    await asyncio.wait_for(a.run(), 30)
    names = [n for n, _ in g.game_run.history]
    assert names[:2] == ["ACTION3", "RESET"]          # reset issued on the model's behalf
    assert g.game_run.state == "won"
    assert any("issued for you" in str(m["content"]) for m in llm.seen[-1] if m["role"] == "user")


async def test_model_that_resets_itself_is_not_double_reset():
    g = FakeGame(levels=1)
    llm = ScriptedLLM(["action('ACTION3')", "action('RESET')", WIN])
    a = GameAgent(g, cfg(), llm)
    await asyncio.wait_for(a.run(), 30)
    names = [n for n, _ in g.game_run.history]
    assert names.count("RESET") == 1 and g.game_run.state == "won"


async def test_history_turn_cap_drains_oldest_turns():
    g = FakeGame(levels=1)
    steps = ["notes = 'ACTION1 moves right'\nprint('t')"] * 14 + [WIN]
    llm = ScriptedLLM(steps)
    a = GameAgent(g, cfg(max_history_turns=6, history_turn_drain=3), llm)
    await asyncio.wait_for(a.run(), 30)
    assert a.stats.trims >= 1 and g.game_run.state == "won"
    assert max(len(m) for m in llm.seen) <= 2 + 3 * 7           # never far above the cap
    assert "ACTION1 moves right" in str(llm.seen[-1][1]["content"])   # string notes survive too


def test_config_clamps_bad_environment(monkeypatch):
    monkeypatch.setenv("ARCD_SLOTS", "banana")
    monkeypatch.setenv("ARCD_CONTEXT_BUDGET", "5")
    monkeypatch.setenv("ARCD_CONTEXT_DRAIN_TARGET", "999999999")
    monkeypatch.setenv("ARCD_TEMPERATURE", "-3")
    monkeypatch.setenv("ARCD_EXPOSE_UNDO", "off")
    monkeypatch.setenv("ARCD_BASE_URL", "http://x:1/v1/")
    c = Config.from_env()
    assert c.slots == 10                                  # unparsable -> default
    assert c.context_budget_tokens == 8 * 1024            # clamped up
    assert c.context_drain_target_tokens < c.context_budget_tokens
    assert c.temperature == 0.0
    assert c.expose_undo is False
    assert c.base_url == "http://x:1/v1"
