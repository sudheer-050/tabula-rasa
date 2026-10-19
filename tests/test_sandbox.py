import time
from dataclasses import replace
from fakes import FakeGame
from arcduck2.config import Config
from arcduck2.sandbox import Sandbox
from arcduck2.session import GameSession


def make(**kw):
    kw.setdefault('guards_from_level', 0)
    cfg = replace(Config(), tool_timeout_s=2.0, **kw)
    g = FakeGame(undo=True)
    return g, GameSession(g), cfg


def test_print_and_state_persistence():
    g, s, cfg = make()
    sb = Sandbox(s, cfg)
    out = sb.run("def f(a):\n    return a*2\nx = 21\nprint(f(x), grid.shape, levels_completed)")
    assert "42 (64, 64) 0" in out.text and not out.error
    out2 = sb.run("print(f(5), x)")
    assert "10 21" in out2.text


def test_actions_stop_after_level_up():
    g, s, cfg = make()
    sb = Sandbox(s, cfg)
    out = sb.run("for _ in range(10):\n    action('ACTION1')\nprint('done')")
    assert out.level_up and out.executed_actions == 3
    assert "re-read the new board" in out.text
    assert len(g.game_run.history) == 3


def test_actions_stop_after_noop_and_gameover():
    g, s, cfg = make()
    sb = Sandbox(s, cfg)
    out = sb.run("action('ACTION4')\naction('ACTION1')")
    assert out.executed_actions == 1 and "changed nothing" in out.text
    g, s, cfg = make()
    sb = Sandbox(s, cfg)
    out = sb.run("action('ACTION3')\naction('RESET')")
    assert out.game_over and out.executed_actions == 1 and "GAME_OVER" in out.text


def test_per_call_action_limit():
    g, s, cfg = make(max_actions_per_call=2)
    sb = Sandbox(s, cfg)
    out = sb.run("action('ACTION1')\naction('ACTION2')\naction('ACTION1')")
    assert out.executed_actions == 2 and "per-call action limit" in out.text


def test_infinite_loop_is_killed():
    g, s, cfg = make()
    sb = Sandbox(s, cfg)
    t = time.time()
    out = sb.run("while True:\n    pass")
    assert out.timed_out and out.error and time.time() - t < 10
    assert "exceeded" in out.text
    # the sandbox is still usable afterwards
    assert "ok" in sb.run("print('ok')").text


def test_blocked_import_and_open():
    g, s, cfg = make()
    sb = Sandbox(s, cfg)
    assert "not allowed" in sb.run("import os").text
    assert "NameError" in sb.run("open('x')").text
    assert "ok" in sb.run("import numpy, itertools\nprint('ok')").text


def test_errors_and_truncation():
    g, s, cfg = make(tool_output_tokens=128)
    sb = Sandbox(s, cfg)
    out = sb.run("1/0")
    assert out.error and "ZeroDivisionError" in out.text
    big = sb.run("print('x'*100000)")
    assert len(big.text) < 1000 and "omitted" in big.text


def test_token_charged_once_per_call():
    g, s, cfg = make()
    sb = Sandbox(s, cfg)
    sb.run("action('ACTION1')\naction('ACTION2')", generated_tokens=777)
    assert [t for _, t in g.game_run.history] == [777, 0]


def test_frame_diff_helper_and_syntax_error():
    g, s, cfg = make()
    sb = Sandbox(s, cfg)
    out = sb.run("action('ACTION1')\nd = frame_diff()\nprint(d['changed_cells'], d['objects'])")
    assert "2" in out.text and "moved" in out.text
    assert "SyntaxError" in sb.run("def (").text


def test_commit_plan_runs_a_list_and_stops_on_level_up():
    g, s, cfg = make()
    sb = Sandbox(s, cfg)
    out = sb.run("commit_plan(['ACTION1', ('ACTION1', None, None), 'ACTION1', {'name': 'ACTION1'}])")
    assert out.level_up and out.executed_actions == 3           # FakeGame levels at x>=5, started x=2
    assert len(g.game_run.history) == 3


def test_bare_action_blocked_once_probe_streak_hits_limit():
    g, s, cfg = make(probe_streak_limit=2, min_commit_batch=3)
    sb = Sandbox(s, cfg)
    s.probe_streak = 2                          # simulate two prior small calls, as agent.py would track
    out = sb.run("r = action('ACTION1')\nprint(r.line())")
    assert out.executed_actions == 0 and "blocked" in out.text and "commit_plan" in out.text
    assert len(g.game_run.history) == 0          # truly refused, no action spent


def test_commit_plan_bypasses_the_gate_once_it_is_long_enough():
    g, s, cfg = make(probe_streak_limit=2, min_commit_batch=3)
    sb = Sandbox(s, cfg)
    s.probe_streak = 2
    out = sb.run("commit_plan(['ACTION1', 'ACTION1', 'ACTION1'])")
    assert out.executed_actions == 3 and len(g.game_run.history) == 3


def test_commit_plan_itself_refuses_a_too_short_plan_while_gated():
    g, s, cfg = make(probe_streak_limit=2, min_commit_batch=3)
    sb = Sandbox(s, cfg)
    s.probe_streak = 2
    out = sb.run("commit_plan(['ACTION1'])")
    assert out.executed_actions == 0 and "too short" in out.text
    assert len(g.game_run.history) == 0
