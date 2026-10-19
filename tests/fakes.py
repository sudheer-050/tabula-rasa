"""A tiny deterministic fake of taaf.game.Game for offline tests.

Board: 64x64, a player pixel (color 8) on row 10 starting at x=2.
ACTION1 -> move right, ACTION2 -> move left, ACTION3 -> step into a pit (GAME_OVER),
ACTION4 -> does nothing. Reaching x=5 completes the level; 2 levels total.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import arcengine
import numpy as np


@dataclass
class FakeFrame:
    data: np.ndarray


class FakeRaw:
    def __init__(self, frames, state, levels, avail):
        self.frame = frames
        self.state = state
        self.levels_completed = levels
        self.available_actions = avail


class FakeState:
    def __init__(self, raw, prev=None, just_won=False):
        self.raw = raw
        self.previous_action = prev
        self.just_won_level = just_won

    @property
    def frame(self):
        return FakeFrame(self.raw.frame[-1])

    @property
    def animation_frames(self):
        return [FakeFrame(f) for f in self.raw.frame[:-1]]

    @property
    def available_actions(self):
        raw = list(self.raw.available_actions)
        return raw if 0 in raw else [0, *raw]

    @property
    def levels_completed(self):
        return self.raw.levels_completed


@dataclass
class FakeRun:
    number_of_levels: int
    state: str = "playing"
    history: list = field(default_factory=list)
    actions_per_level: list = field(default_factory=list)
    levels_completed: int = 0
    final_score: float | None = None
    solver_note: str | None = None


class FakeGame:
    def __init__(self, undo=False, levels=2):
        self.number_of_levels = levels
        self.grid_size = (64, 64)
        self.base_actions_per_level = [4] * levels
        self.game_id = "fake-0001"
        self.undo = undo
        self.x = 2
        self.level = 0
        self.dead = False
        self.game_run = FakeRun(levels, actions_per_level=[0] * levels)
        self._current_state = self._mk(None, False)

    @property
    def current_state(self):
        return self._current_state

    def _frame(self):
        g = np.zeros((64, 64), dtype=np.int8)
        g[10, self.x] = 8
        g[20, 30] = 3 + self.level
        return g

    def _avail(self):
        a = [0, 1, 2, 3, 4]
        if self.undo:
            a.append(7)
        return a

    def _mk(self, prev, just_won, frames=None):
        st = arcengine.GameState.GAME_OVER if self.dead else (
            arcengine.GameState.WIN if self.level >= self.number_of_levels else arcengine.GameState.NOT_FINISHED)
        raw = FakeRaw(frames or [self._frame()], st, self.level, self._avail())
        return FakeState(raw, prev, just_won)

    def execute_action(self, action, generated_tokens=0, uncached_input_tokens=0):
        assert self.game_run.state == "playing"
        if action.id.value not in self._current_state.available_actions:
            raise ValueError("not available")
        before = self.level
        a = action.id.value
        if a == 0:
            self.x, self.dead = 2, False
        elif self.dead:
            raise RuntimeError("empty frame after game over")
        elif a == 1:
            self.x += 1
        elif a == 2:
            self.x = max(0, self.x - 1)
        elif a == 3:
            self.dead = True
        elif a == 7:
            self.x = max(0, self.x - 1)
        if not self.dead and self.x >= 5 and a != 0:
            self.level += 1
            self.x = 2
        just = self.level > before
        if just and self.level < self.number_of_levels:
            frames = [np.full((64, 64), 14, dtype=np.int8), self._frame()]
        else:
            frames = None
        self.game_run.history.append((action.id.name, generated_tokens))
        self.game_run.actions_per_level[min(before, self.number_of_levels - 1)] += 1
        if self.level >= self.number_of_levels:
            self.game_run.state = "won"
        self._current_state = self._mk(action, just, frames)
        self.game_run.levels_completed = self.level
        return self._current_state

    def finish_game(self, generated_tokens=0, uncached_input_tokens=0):
        if self.game_run.state == "playing":
            self.game_run.state = "gave_up"
        self.game_run.final_score = float(self.game_run.levels_completed)


# ---------------------------------------------------------------------------
from arcduck2.llm import LLMReply, ToolCall, ContextTooLong  # noqa: E402


def code_reply(code, reasoning="thinking...", tokens=50, tool=True, n_calls=1):
    calls = [ToolCall(f"call_{i}", "python", code) for i in range(n_calls)] if tool else []
    raw = [{"id": c.id, "type": "function", "function": {"name": "python", "arguments": '{"code": "x"}'}} for c in calls]
    return LLMReply(content="" if tool else f"```python\n{code}\n```", reasoning=reasoning, tool_calls=calls,
                    raw_tool_calls=raw, finish_reason="tool_calls", prompt_tokens=1000, completion_tokens=tokens)


class ScriptedLLM:
    """Plays back replies in order (or calls a function(messages, idx))."""

    def __init__(self, steps):
        self.steps, self.i, self.seen = steps, 0, []

    def chat(self, messages, max_tokens=None):
        self.seen.append([dict(m) for m in messages])
        step = self.steps(messages, self.i) if callable(self.steps) else self.steps[min(self.i, len(self.steps) - 1)]
        self.i += 1
        if isinstance(step, Exception):
            raise step
        if isinstance(step, str):
            step = code_reply(step)
        return step
