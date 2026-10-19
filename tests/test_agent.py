import asyncio
from dataclasses import replace
import pytest
from fakes import FakeGame, ScriptedLLM, code_reply
from arcduck2.agent import GameAgent
from arcduck2.config import Config
from arcduck2.llm import ContextTooLong, ModelUnavailable, LLMReply


def cfg(**kw):
    return replace(Config(), tool_timeout_s=3.0, **kw)


def check_structure(msgs):
    """Every tool message must answer a tool_call of the nearest earlier assistant."""
    pending = set()
    for m in msgs:
        if m["role"] == "assistant":
            assert not pending, "unanswered tool calls before next assistant turn"
            pending = {tc["id"] for tc in m.get("tool_calls") or []}
        elif m["role"] == "tool":
            assert m["tool_call_id"] in pending
            pending.discard(m["tool_call_id"])
    assert not pending or msgs[-1]["role"] == "assistant"


async def play(agent, **kw):
    await asyncio.wait_for(agent.run(**kw), 30)


async def test_plays_to_win_and_keeps_reasoning():
    g = FakeGame()
    llm = ScriptedLLM([
        "print(grid.shape)",
        "action('ACTION1')\naction('ACTION1')\naction('ACTION1')",   # level 1
        "action('ACTION1')\naction('ACTION1')\naction('ACTION1')",   # level 2 -> win
    ])
    a = GameAgent(g, cfg(), llm)
    await play(a)
    assert g.game_run.state == "won" and g.game_run.levels_completed == 2
    assert a.stats.model_calls == 3 and a.stats.tool_errors == 0
    last = llm.seen[-1]
    check_structure(last)
    assert last[0]["role"] == "system"
    assistants = [m for m in last if m["role"] == "assistant"]
    assert all(m.get("reasoning_content") == "thinking..." for m in assistants)   # thinking retained
    assert any("LEVEL COMPLETE" in str(m["content"]) for m in last)
    assert "calls=3" in g.game_run.solver_note


async def test_game_over_prompt_and_fatal_memory():
    g = FakeGame()
    llm = ScriptedLLM([
        "action('ACTION1')\naction('ACTION3')",   # dies
        "action('RESET')",
        # The one safe ACTION1 is replayed for free after RESET (frontier memory); this
        # call picks up from there and retries the fatal move, which is refused.
        "r = action('ACTION3')\nprint('second try:', r.line())",  # refused, costs nothing
        "action('ACTION1')\naction('ACTION1')",
    ] + ["action('ACTION1')\naction('ACTION1')\naction('ACTION1')"] * 3)
    a = GameAgent(g, cfg(), llm)
    await play(a)
    msgs = llm.seen[2]
    assert any("GAME_OVER." in str(m["content"]) and "deterministic" in str(m["content"]) for m in msgs if m["role"] == "user")
    flat = str(llm.seen[3])
    assert "REFUSED" in flat and "no action spent" in flat.lower()
    names = [n for n, _ in g.game_run.history]
    assert names.count("ACTION3") == 1                     # never replayed the fatal move


async def test_frontier_replay_skips_known_safe_prefix():
    g = FakeGame()
    llm = ScriptedLLM([
        "action('ACTION1')\naction('ACTION1')\naction('ACTION3')",  # dies 2 actions deep
        "action('RESET')",
        "action('ACTION1')",                                          # finishes level 1 from the replayed frontier
        "action('ACTION1')\naction('ACTION1')\naction('ACTION1')",    # level 2 -> win
    ])
    a = GameAgent(g, cfg(), llm)
    await play(a)
    assert g.game_run.state == "won"
    assert a.stats.model_calls == 4                          # not 5: the 2 safe actions cost no model call
    assert a.stats.replay_actions == 2
    assert "replay=2" in g.game_run.solver_note
    flat = str(llm.seen[-1])
    assert "[cascade] Replayed 2/2" in flat and "no model cost" in flat


async def test_best_replay_completes_level_with_zero_model_calls():
    g = FakeGame()
    a = GameAgent(g, cfg(), ScriptedLLM([
        "action('ACTION1')\naction('ACTION1')\naction('ACTION1')",   # level 2 -> win
    ]))
    # Seed a proven-winning sequence for level 1, as if an earlier pass had found it.
    a.session.best[0] = (("ACTION1", None, None),) * 3
    await play(a)
    assert g.game_run.state == "won"
    assert a.stats.model_calls == 1                           # level 1 cost zero model calls
    assert a.stats.replay_actions == 3
    assert g.game_run.levels_completed == 2


async def test_forced_commitment_blocks_drip_then_recovers_via_commit_plan():
    g = FakeGame()
    llm = ScriptedLLM([
        "action('ACTION1')",                                          # small call #1 (x: 2->3)
        "action('ACTION1')",                                          # small call #2 (x: 3->4); streak now at limit
        "r = action('ACTION1')\nprint(r.line())",                     # blocked: streak already at limit
        "commit_plan(['ACTION1', 'ACTION1', 'ACTION1'])",             # long enough; wins level 1 on the first action
        "action('ACTION1')\naction('ACTION1')\naction('ACTION1')",    # level 2 -> win
    ])
    a = GameAgent(g, cfg(min_commit_batch=3, probe_streak_limit=2), llm)
    await play(a)
    assert g.game_run.state == "won"
    assert a.stats.model_calls == 5
    names = [n for n, _ in g.game_run.history]
    assert names.count("ACTION1") == 6                        # the blocked attempt never reached the engine
    blocked_turn = str(llm.seen[3])
    assert "blocked" in blocked_turn and "commit_plan" in blocked_turn


async def test_no_tool_call_is_nudged_not_fatal():
    g = FakeGame(levels=1)
    llm = ScriptedLLM([
        LLMReply(content="I think we should move", reasoning="r", finish_reason="stop", completion_tokens=10, prompt_tokens=100),
        "action('ACTION1')\naction('ACTION1')\naction('ACTION1')",
    ])
    a = GameAgent(g, cfg(), llm)
    await play(a)
    assert g.game_run.state == "won" and a.stats.no_tool_turns == 1
    assert "did not call the python tool" in str(llm.seen[1])


async def test_text_fenced_code_fallback_runs():
    g = FakeGame(levels=1)
    llm = ScriptedLLM([code_reply("action('ACTION1')\naction('ACTION1')\naction('ACTION1')", tool=False)])
    a = GameAgent(g, cfg(), llm)
    await play(a)
    assert g.game_run.state == "won"
    check_structure(llm.seen[-1])


async def test_multiple_tool_calls_all_answered():
    g = FakeGame(levels=1)
    llm = ScriptedLLM([code_reply("action('ACTION1')", n_calls=3),
                       "action('ACTION1')\naction('ACTION1')"])
    a = GameAgent(g, cfg(), llm)
    await play(a)
    assert g.game_run.state == "won"
    check_structure(llm.seen[1])
    tool_msgs = [m for m in llm.seen[1] if m["role"] == "tool"]
    assert len(tool_msgs) == 3 and "Not executed" in tool_msgs[1]["content"]


async def test_context_trim_keeps_brief_and_notes():
    g = FakeGame(levels=1)
    steps = ["notes = ['ACTION1 moves right']\nprint('x' * 3000)"] * 12 + ["action('ACTION1')\naction('ACTION1')\naction('ACTION1')"]
    llm = ScriptedLLM(steps)
    a = GameAgent(g, cfg(context_budget_tokens=8 * 1024, context_drain_target_tokens=4 * 1024), llm)
    await play(a)
    assert a.stats.trims >= 1 and g.game_run.state == "won"
    final = llm.seen[-1]
    check_structure(final)
    assert "Older messages were trimmed" in str(final[1]["content"])
    assert "ACTION1 moves right" in str(final[1]["content"])      # notes survive trimming
    assert len(final) < 2 + 3 * 13


async def test_llm_summary_captured_before_eviction_when_enabled():
    g = FakeGame(levels=1)
    llm = ScriptedLLM([
        "notes = ['learned X']\nprint('y')",
        "print('z')",
        "print('w')",
        LLMReply(content="Summary: board has a moving block; ACTION1 is safe; try ACTION2 next.",
                 reasoning="", finish_reason="stop", completion_tokens=20, prompt_tokens=50),
    ])
    a = GameAgent(g, cfg(use_llm_summary=True, context_drain_target_tokens=10), llm)
    for _ in range(3):
        await a._turn()
    assert len(a.turns) == 3 and a.stats.model_calls == 3
    changed = await a.trim_context(force=True)
    assert changed and len(a.turns) < 3
    assert a.stats.model_calls == 4                          # the summary request counted too
    assert "Summary: board has a moving block" in a._summary
    text = a.brief["content"][0]["text"] if isinstance(a.brief["content"], list) else a.brief["content"]
    assert "Summary of what was learned before older messages were trimmed" in text
    assert "Summary: board has a moving block" in text


async def test_llm_summary_off_by_default():
    g = FakeGame(levels=1)
    llm = ScriptedLLM(["print('y')", "print('z')", "print('w')"])
    a = GameAgent(g, cfg(context_drain_target_tokens=10), llm)   # use_llm_summary defaults False
    for _ in range(3):
        await a._turn()
    await a.trim_context(force=True)
    assert a._summary == "" and a.stats.model_calls == 3        # no extra call spent


async def test_context_too_long_recovery():
    g = FakeGame(levels=1)
    llm = ScriptedLLM(lambda msgs, i: ContextTooLong("ctx") if i == 0 else "action('ACTION1')\naction('ACTION1')\naction('ACTION1')")
    a = GameAgent(g, cfg(), llm)
    await play(a)
    assert g.game_run.state == "won"


async def test_model_unavailable_stops_cleanly():
    g = FakeGame()
    llm = ScriptedLLM([ModelUnavailable("down")])
    a = GameAgent(g, cfg(max_consecutive_model_failures=3), llm)
    await play(a)
    assert a.stop_reason == "model unavailable" and g.game_run.state == "playing"


async def test_deadline_stops():
    g = FakeGame()
    llm = ScriptedLLM(["print('thinking')"])
    t = {"now": 0.0}

    def clock():
        t["now"] += 1.0
        return t["now"]

    a = GameAgent(g, cfg(), llm, clock=clock)
    await play(a, deadline=20.0)
    assert a.stop_reason == "deadline"


async def test_tool_error_is_reported_to_model():
    g = FakeGame(levels=1)
    llm = ScriptedLLM(["1/0", "action('ACTION1')\naction('ACTION1')\naction('ACTION1')"])
    a = GameAgent(g, cfg(), llm)
    await play(a)
    assert a.stats.tool_errors == 1 and "ZeroDivisionError" in str(llm.seen[1])


async def test_idle_model_releases_slot():
    from arcduck2.scheduler import PriorityScheduler
    g = FakeGame()
    sched = PriorityScheduler(1, 100.0)
    llm = ScriptedLLM(["print('only thinking')"])
    a = GameAgent(g, cfg(), llm, scheduler=sched)
    task = asyncio.create_task(a.run(deadline=a.clock() + 1.0))
    await asyncio.wait_for(task, 20)
    assert not sched.active    # slot returned on exit


async def test_unexpected_model_exception_is_retried():
    g = FakeGame(levels=1)
    llm = ScriptedLLM(lambda msgs, i: ValueError("weird payload") if i == 0 else WIN_CODE)
    a = GameAgent(g, cfg(), llm)
    await play(a)
    assert g.game_run.state == "won" and a.stats.model_failures == 0   # reset after the recovery


async def test_repeated_internal_errors_stop_the_game_not_the_run():
    g = FakeGame()
    a = GameAgent(g, cfg(), ScriptedLLM([WIN_CODE]))

    async def broken_turn():
        raise KeyError("bug")
    a._turn = broken_turn
    await play(a)
    assert "internal errors" in a.stop_reason


WIN_CODE = "action('ACTION1')\naction('ACTION1')\naction('ACTION1')"
