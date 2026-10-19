from fakes import FakeGame
from arcduck2.session import GameSession


def test_basic_move_and_level_up():
    g = FakeGame()
    s = GameSession(g)
    assert s.available_names() == ["RESET", "ACTION1", "ACTION2", "ACTION3", "ACTION4"]
    r = s.execute("action1")
    assert r.ok and r.board_changed and not r.level_up
    for _ in range(2):
        r = s.execute("ACTION1")
    assert r.level_up and r.levels_completed == 1
    assert s.level_actions == 0 and s.total_actions == 3


def test_fatal_path_refused_after_reset():
    g = FakeGame()
    s = GameSession(g)
    assert s.execute("ACTION1").ok
    r = s.execute("ACTION3")
    assert r.ok and r.game_over
    assert not s.execute("ACTION1").ok  # only RESET accepted
    assert s.execute("RESET").ok
    assert s.execute("ACTION1").ok
    bad = s.execute("ACTION3")
    assert not bad.ok and bad.refused and "deterministic" in bad.error
    assert g.game_run.history[-1][0] == "ACTION1"  # nothing spent on the refused action
    assert "known-fatal" in s.fatal_summary()


def test_noop_refused_twice_then_allowed():
    g = FakeGame()
    s = GameSession(g)
    assert s.execute("ACTION4").ok  # no change
    n = len(g.game_run.history)
    assert s.execute("ACTION4").refused
    assert s.execute("ACTION4").refused
    assert s.execute("ACTION4").ok  # lifted on the third try
    assert len(g.game_run.history) == n + 1


def test_undo_gated_and_pops_path():
    s = GameSession(FakeGame(undo=False))
    assert not s.execute("UNDO").ok
    g = FakeGame(undo=True)
    s = GameSession(g)
    assert "UNDO" in s.available_names()
    s.execute("ACTION1")
    assert len(s.path) == 1
    assert s.execute("UNDO").ok
    assert len(s.path) == 0
    assert "UNDO" not in s.available_names(expose_undo=False)
    assert not s.execute("UNDO", expose_undo=False).ok


def test_validation_errors_cost_nothing():
    g = FakeGame()
    s = GameSession(g)
    for args in [("CLICK", 1, 1), ("ACTION1", 3, 4), ("BOGUS",), ("ACTION5",)]:
        r = s.execute(*args)
        assert not r.ok
    assert len(g.game_run.history) == 0


def test_finished_game_refuses():
    g = FakeGame(levels=1)
    s = GameSession(g)
    for _ in range(3):
        r = s.execute("ACTION1")
    assert r.won and s.is_won
    assert not s.execute("ACTION1").ok
