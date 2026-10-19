"""Bridge between one TAAF ``Game`` and the agent.

Owns everything that touches the framework: action validation, per-level
bookkeeping, fatal-path memory and no-op memory. The scoring rule makes every
engine action (including RESET and failed attempts) cost efficiency on the
current level, so this class refuses actions that are provably wasted:

* an action sequence that already ended in GAME_OVER on this level
  (the games are deterministic, so repeating it cannot succeed), and
* an action repeated in a bit-identical frame after it already changed nothing.

Both refusals cost no engine action. The no-op refusal is lifted after two
refusals in a row, because hidden (undrawn) state can make an identical frame
behave differently.

It also keeps positive memory, the mirror image of ``fatal``/``noop_seen``:

* ``best``, the shortest action sequence known to CLEAR a level outright, and
* ``frontier``, the longest action sequence known NOT to end in GAME_OVER,
  banked from a failed attempt's path up to (not including) the fatal action.

RESET only restarts the level currently being attempted (completed levels are
never revisited), so a level is typically retried several times before it is
first cleared -- exactly where ``best`` (set only on completion) cannot help
yet but ``frontier`` can. The agent replays whichever is available with no
model call after a RESET, so repeated attempts don't re-pay LLM cost (or risk
a worse outcome) for ground already proven safe, and spend the turn budget
pushing past the known point instead.

Finally, ``probe_streak`` counts consecutive small tool calls (below
``cfg.min_commit_batch`` actions) on the current level. The sandbox reads it
to force a real plan via ``commit_plan`` once it crosses
``cfg.probe_streak_limit``, since small-step drip play -- not GAME_OVER
retries -- is what trace data showed actually burning the action budget. It
resets on a level change (a new level may need fresh exploration) but not on
RESET (a retry hasn't earned a new grace period).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import arcengine
import numpy as np

from . import perception as P

SIMPLE_NAMES = {"ACTION1": 1, "ACTION2": 2, "ACTION3": 3, "ACTION4": 4, "ACTION5": 5}
ALIASES = {"CLICK": "ACTION6", "UNDO": "ACTION7"}
NOOP_REFUSALS_BEFORE_ALLOW = 2


def canonical_action(name: Any) -> str:
    raw = str(name or "").strip().upper().replace(" ", "")
    return ALIASES.get(raw, raw)


def model_name(engine_name: str) -> str:
    return {"ACTION6": "CLICK", "ACTION7": "UNDO"}.get(engine_name, engine_name)


@dataclass
class ActionResult:
    ok: bool
    action: str = ""
    error: str = ""
    refused: bool = False
    board_changed: bool = False
    level_up: bool = False
    game_over: bool = False
    won: bool = False
    diff: str = ""
    levels_completed: int = 0
    level_actions: int = 0

    def line(self) -> str:
        if not self.ok:
            tag = "REFUSED (no action spent)" if self.refused else "ERROR (no action spent)"
            return f"{self.action or '?'}: {tag}: {self.error}"
        bits = [f"{self.action}: {'changed' if self.board_changed else 'NO CHANGE'}"]
        if self.level_up:
            bits.append(f"LEVEL UP -> {self.levels_completed} complete")
        if self.won:
            bits.append("GAME WON")
        if self.game_over:
            bits.append("GAME_OVER (only RESET is accepted now)")
        if self.diff and self.board_changed:
            bits.append(self.diff)
        return " | ".join(bits)


@dataclass
class GameSession:
    game: Any
    grid: np.ndarray = field(init=False)
    prev_grid: np.ndarray = field(init=False)
    animation: list[np.ndarray] = field(default_factory=list)
    path: list[tuple] = field(default_factory=list)
    fatal: dict[int, set] = field(default_factory=dict)
    best: dict[int, tuple] = field(default_factory=dict)
    frontier: dict[int, tuple] = field(default_factory=dict)
    noop_seen: dict[tuple, int] = field(default_factory=dict)
    action_log: list[tuple[int, str, bool]] = field(default_factory=list)
    probe_streak: int = 0
    refusal_streak: int = 0
    error_count: int = 0
    guards_from_level: int = 0

    def __post_init__(self) -> None:
        frame = self._frame()
        self.grid = frame
        self.prev_grid = frame.copy()

    # --- read-only views ---------------------------------------------------
    def _state(self):
        return self.game.current_state

    def _frame(self) -> np.ndarray:
        return np.asarray(self._state().frame.data).astype(np.int16)

    @property
    def raw_state(self) -> arcengine.GameState:
        return self._state().raw.state

    @property
    def is_game_over(self) -> bool:
        return self.raw_state == arcengine.GameState.GAME_OVER

    @property
    def is_won(self) -> bool:
        return self.game.game_run is not None and self.game.game_run.state == "won"

    @property
    def is_playing(self) -> bool:
        run = self.game.game_run
        return run is not None and run.state == "playing"

    @property
    def levels_completed(self) -> int:
        return int(self._state().levels_completed)

    @property
    def number_of_levels(self) -> int:
        return int(self.game.number_of_levels)

    @property
    def level_actions(self) -> int:
        run = self.game.game_run
        idx = min(self.levels_completed, self.number_of_levels - 1)
        if run is None or idx < 0 or idx >= len(run.actions_per_level):
            return 0
        return int(run.actions_per_level[idx])

    @property
    def total_actions(self) -> int:
        run = self.game.game_run
        return 0 if run is None else len(run.history)

    def available_names(self, expose_undo: bool = True) -> list[str]:
        names = []
        for aid in self._state().available_actions:
            engine = arcengine.GameAction.from_id(aid).name
            if engine == "ACTION7" and not expose_undo:
                continue
            names.append(model_name(engine))
        return names

    # --- action execution --------------------------------------------------
    def execute(self, name: Any, x: Any = None, y: Any = None, *, expose_undo: bool = True,
                generated_tokens: int = 0, uncached_input_tokens: int = 0) -> ActionResult:
        engine = canonical_action(name)
        shown = model_name(engine) if engine else str(name)
        if not self.is_playing:
            return ActionResult(False, shown, "the game is already finished", refused=True)
        if engine == "ACTION7" and not expose_undo:
            return ActionResult(False, shown, "UNDO is not enabled", refused=True)
        if engine not in {"RESET", "ACTION6", "ACTION7", *SIMPLE_NAMES}:
            return ActionResult(False, shown, f"unknown action {name!r}; use {self.available_names(expose_undo)}", refused=True)

        valid_ids = self._state().available_actions
        aid = arcengine.GameAction[engine].value
        if aid not in valid_ids:
            return ActionResult(False, shown, f"{shown} is not available now; available: {self.available_names(expose_undo)}", refused=True)

        data: dict[str, int] = {}
        if engine == "ACTION6":
            try:
                xi, yi = int(x), int(y)
            except (TypeError, ValueError):
                return ActionResult(False, shown, "CLICK needs integer x and y", refused=True)
            w, h = self.game.grid_size
            if not (0 <= xi < w and 0 <= yi < h):
                return ActionResult(False, shown, f"CLICK ({xi},{yi}) is outside the {w}x{h} grid", refused=True)
            data = {"x": xi, "y": yi}
        elif x is not None or y is not None:
            return ActionResult(False, shown, f"{shown} takes no coordinates", refused=True)

        if self.is_game_over and engine != "RESET":
            return ActionResult(False, shown, "GAME_OVER: only RESET is accepted", refused=True)

        level = self.levels_completed
        sig = (engine, data.get("x"), data.get("y"))

        if engine != "RESET":
            candidate = tuple(self.path) + (sig,)
            if candidate in self.fatal.get(level, ()):
                return ActionResult(False, shown, "this exact action sequence already ended in GAME_OVER on this level; the game is deterministic, choose a different action", refused=True)
            key = (level, P.grid_hash(self.grid), sig)
            seen = self.noop_seen.get(key, 0) if level >= self.guards_from_level else 0
            if seen:
                self.refusal_streak += 1
                if self.refusal_streak <= NOOP_REFUSALS_BEFORE_ALLOW:
                    return ActionResult(False, shown, "this action already changed nothing in this exact frame; try something different", refused=True)
            self.refusal_streak = 0

        before = self.grid
        try:
            new_state = self.game.execute_action(
                arcengine.ActionInput(id=arcengine.GameAction[engine], data=data),
                generated_tokens=generated_tokens,
                uncached_input_tokens=uncached_input_tokens,
            )
        except (ValueError, RuntimeError) as exc:
            self.error_count += 1
            return ActionResult(False, shown, f"{type(exc).__name__}: {exc}")
        self.error_count = 0

        after = np.asarray(new_state.frame.data).astype(np.int16)
        self.animation = [np.asarray(f.data).astype(np.int16) for f in new_state.animation_frames]
        self.prev_grid, self.grid = before, after
        changed = not np.array_equal(before, after)
        level_up = bool(new_state.just_won_level)
        over = new_state.raw.state == arcengine.GameState.GAME_OVER
        won = self.is_won

        if engine == "RESET":
            self.path = []
        elif level_up:
            winning_path = tuple(self.path) + (sig,)
            current_best = self.best.get(level)
            if current_best is None or len(winning_path) < len(current_best):
                self.best[level] = winning_path
            self.path = []
            self.noop_seen = {}
            self.probe_streak = 0
        elif engine == "ACTION7":
            if changed and self.path:
                self.path.pop()
            elif not changed:
                key = (level, P.grid_hash(before), sig)
                self.noop_seen[key] = self.noop_seen.get(key, 0) + 1
        else:
            if not changed:
                key = (level, P.grid_hash(before), sig)
                self.noop_seen[key] = self.noop_seen.get(key, 0) + 1
            self.path.append(sig)
            if over:
                self.fatal.setdefault(level, set()).add(tuple(self.path))
                # Everything before the fatal action was proven not to be immediately
                # fatal; bank it so the next attempt can replay straight back to this
                # frontier for free instead of re-deriving the same safe opening moves.
                safe_prefix = tuple(self.path[:-1])
                current_frontier = self.frontier.get(level)
                if safe_prefix and (current_frontier is None or len(safe_prefix) > len(current_frontier)):
                    self.frontier[level] = safe_prefix
        self.action_log.append((level, shown, changed))

        return ActionResult(
            True, shown, board_changed=changed, level_up=level_up, game_over=over, won=won,
            diff=P.diff_summary(before, after) if changed else "",
            levels_completed=int(new_state.levels_completed), level_actions=self.level_actions,
        )

    def fatal_summary(self) -> str:
        level = self.levels_completed
        paths = self.fatal.get(level, set())
        if not paths:
            return ""
        shortest = sorted(paths, key=len)[:3]
        rendered = ["[" + ", ".join(f"{model_name(a)}" + (f"({x},{y})" if x is not None else "") for a, x, y in p) + "]" for p in shortest]
        return f"{len(paths)} known-fatal action sequence(s) on this level, e.g. " + "; ".join(rendered)
