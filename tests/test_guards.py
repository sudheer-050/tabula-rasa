from dataclasses import replace

from fakes import FakeGame
from arcduck2.config import Config
from arcduck2.sandbox import Sandbox
from arcduck2.session import GameSession


def make(**kw):
    cfg = replace(Config(), tool_timeout_s=2.0, **kw)
    g = FakeGame(undo=True)
    return g, GameSession(g), cfg


def test_noop_stop_guard_is_off_on_first_level_by_default():
    g, s, cfg = make(guards_from_level=1)
    sb = Sandbox(s, cfg)
    out = sb.run("action('ACTION4')\naction('ACTION1')")
    assert out.executed_actions == 2 and "changed nothing" not in out.text   # free experimentation on level 1
    sb.run("for _ in range(3):\n    action('ACTION1')")
    assert s.levels_completed == 1
    out = sb.run("action('ACTION4')\naction('ACTION1')")
    assert out.executed_actions == 1 and "changed nothing" in out.text      # guard active from level 2


def test_noop_refusal_follows_guard_level():
    g, s, cfg = make(guards_from_level=1)
    sb = Sandbox(s, cfg)                      # sets session.guards_from_level = 1
    assert s.execute("ACTION4").ok
    assert s.execute("ACTION4").ok            # level 1: repeat of a no-op is allowed
    sb.run("for _ in range(3):\n    action('ACTION1')")
    assert s.execute("ACTION4").ok            # level 2: first no-op is executed...
    assert s.execute("ACTION4").refused       # ...the repeat is refused


def test_fatal_refusal_applies_on_every_level():
    g, s, cfg = make(guards_from_level=5)
    Sandbox(s, cfg)
    s.execute("ACTION1")
    assert s.execute("ACTION3").game_over
    s.execute("RESET")
    s.execute("ACTION1")
    assert s.execute("ACTION3").refused


def test_animation_hint_appears():
    g, s, cfg = make(guards_from_level=0)
    sb = Sandbox(s, cfg)
    out = sb.run("for _ in range(3):\n    action('ACTION1')")
    assert out.level_up and "animation_frames" in out.text
