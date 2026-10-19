"""The ``python`` tool: a persistent per-game namespace the model programs against.

Design points
* One namespace per game; everything the model defines persists across calls,
  observation variables are re-injected fresh each call.
* ``action()`` is the only way to touch the game. Within one call it stops
  executing after a level change, GAME_OVER, a win, or an action that changed
  nothing, so a stale plan can never burn actions on the wrong board.
* Imports are whitelisted and file access is removed.
* Timeouts use an async exception, so even an infinite pure-Python loop dies.
"""
from __future__ import annotations

import builtins
import ctypes
import io
import threading
import traceback
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from . import perception as P
from .config import Config
from .session import ActionResult, GameSession

ALLOWED_MODULES = {
    "numpy", "scipy", "math", "itertools", "collections", "heapq", "functools", "re", "copy",
    "json", "random", "typing", "dataclasses", "bisect", "operator", "statistics", "fractions",
    "string", "time", "array", "enum", "queue", "numbers", "decimal",
}
_REMOVED_BUILTINS = {"open", "input", "breakpoint", "exit", "quit", "help", "memoryview", "compile"}


class ToolTimeout(BaseException):
    """Raised inside the worker thread when the call exceeds its time limit."""


def _limited_import(name, globals=None, locals=None, fromlist=(), level=0):
    root = name.split(".")[0]
    if level != 0 or root not in ALLOWED_MODULES:
        raise ImportError(f"import of {name!r} is not allowed; allowed: {sorted(ALLOWED_MODULES)}")
    return builtins.__import__(name, globals, locals, fromlist, level)


def _safe_builtins() -> dict:
    safe = {k: v for k, v in vars(builtins).items() if k not in _REMOVED_BUILTINS}
    safe["__import__"] = _limited_import
    return safe


@dataclass
class ToolOutcome:
    text: str
    results: list[ActionResult] = field(default_factory=list)
    error: bool = False
    timed_out: bool = False
    executed_actions: int = 0
    level_up: bool = False
    game_over: bool = False
    won: bool = False


@dataclass
class _Call:
    generated_tokens: int
    limit: int
    results: list[ActionResult] = field(default_factory=list)
    stop_reason: str = ""
    charged_tokens: bool = False
    cancelled: threading.Event = field(default_factory=threading.Event)


def truncate_middle(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    keep = max(16, (max_chars - 40) // 2)
    return text[:keep] + f"\n...[{len(text) - 2 * keep} chars omitted]...\n" + text[-keep:]


def _parse_action_spec(spec: Any) -> tuple[Any, Any, Any]:
    """A commit_plan() entry: a bare name, a (name, x, y) tuple, or a dict."""
    if isinstance(spec, str):
        return spec, None, None
    if isinstance(spec, dict):
        return spec.get("name") or spec.get("action"), spec.get("x"), spec.get("y")
    if isinstance(spec, (list, tuple)):
        name = spec[0] if spec else None
        x = spec[1] if len(spec) > 1 else None
        y = spec[2] if len(spec) > 2 else None
        return name, x, y
    raise TypeError(f"commit_plan: unrecognised action spec {spec!r}; use a string, (name, x, y), or a dict")


class Sandbox:
    def __init__(self, session: GameSession, cfg: Config):
        self.session = session
        self.cfg = cfg
        session.guards_from_level = cfg.guards_from_level
        self.ns: dict[str, Any] = {"__builtins__": _safe_builtins(), "__name__": "__game__", "np": np}
        self._call: _Call | None = None
        self._user_keys: set[str] = set()
        self._injected: set[str] = set()
        self._in_commit_plan = False

    # --- helpers exposed to the model --------------------------------------
    def _frame_diff(self, before=None, after=None) -> dict:
        a = self.session.prev_grid if before is None else P.as_array(before)
        b = self.session.grid if after is None else P.as_array(after)
        mask = P.diff_mask(a, b)
        ys, xs = np.nonzero(mask)
        return {
            "changed_cells": int(mask.sum()),
            "bbox": None if not len(xs) else (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())),
            "summary": P.diff_summary(a, b),
            "objects": P.object_changes(a, b),
        }

    def _action(self, name, x=None, y=None) -> ActionResult:
        call = self._call
        if call is None or call.cancelled.is_set():
            raise ToolTimeout()
        if not self._in_commit_plan and self.session.probe_streak >= self.cfg.probe_streak_limit:
            res = ActionResult(
                False, str(name),
                f"blocked: {self.session.probe_streak} consecutive small calls on this level without a "
                f"committed plan. Call commit_plan([...]) with your next {self.cfg.min_commit_batch}+ actions "
                f"(or up to where you expect a level change or genuine uncertainty) instead of single "
                f"action() calls.", refused=True,
            )
            call.results.append(res)
            return res
        if call.stop_reason:
            res = ActionResult(False, str(name), f"not executed: {call.stop_reason}", refused=True)
            call.results.append(res)
            return res
        if len([r for r in call.results if r.ok]) >= call.limit:
            res = ActionResult(False, str(name), f"per-call action limit ({call.limit}) reached; stop and look at the board", refused=True)
            call.results.append(res)
            call.stop_reason = res.error
            return res
        gen = 0 if call.charged_tokens else call.generated_tokens
        res = self.session.execute(name, x, y, expose_undo=self.cfg.expose_undo, generated_tokens=gen)
        if res.ok:
            call.charged_tokens = True
            if res.level_up:
                call.stop_reason = "the level changed during this call; re-read the new board before acting"
            elif res.game_over:
                call.stop_reason = "GAME_OVER happened during this call; analyse it, then RESET"
            elif res.won:
                call.stop_reason = "the game is won"
            elif not res.board_changed and self.session.levels_completed >= self.cfg.guards_from_level:
                call.stop_reason = "the previous action changed nothing; inspect before continuing"
        call.results.append(res)
        return res

    def _commit_plan(self, actions: Any) -> list[ActionResult]:
        """Run a list of actions back-to-back, bypassing the forced-commitment gate that
        blocks bare action() once the model has racked up too many small calls in a row.
        Stops early on the same conditions action() already stops a call for (level change,
        GAME_OVER, win, no-op); those don't need repeating here since they flow through
        action() itself."""
        try:
            actions = list(actions)
        except TypeError:
            actions = []
        call = self._call
        if call is None or call.cancelled.is_set():
            raise ToolTimeout()
        if self.session.probe_streak >= self.cfg.probe_streak_limit and len(actions) < self.cfg.min_commit_batch:
            res = ActionResult(
                False, "commit_plan",
                f"plan too short ({len(actions)} action(s); need >= {self.cfg.min_commit_batch} while in "
                f"forced-commit mode). Extend the plan, or if you are genuinely still uncertain include your "
                f"best guesses for the remaining steps anyway -- execution stops safely at the first "
                f"unexpected result.", refused=True,
            )
            call.results.append(res)
            return [res]
        results = []
        self._in_commit_plan = True
        try:
            for spec in actions:
                name, x, y = _parse_action_spec(spec)
                res = self._action(name, x, y)
                results.append(res)
                if call.stop_reason:
                    break
        finally:
            self._in_commit_plan = False
        return results

    def _inject(self) -> None:
        s = self.session
        values = {
            "grid": s.grid.copy(),
            "prev_grid": s.prev_grid.copy(),
            "levels_completed": s.levels_completed,
            "number_of_levels": s.number_of_levels,
            "level_actions": s.level_actions,
            "available_actions": s.available_names(self.cfg.expose_undo),
            "animation_frames": [f.copy() for f in s.animation],
            "action": self._action,
            "commit_plan": self._commit_plan,
            "frame_diff": self._frame_diff,
            "segments": lambda g=None, **kw: P.segments(s.grid if g is None else g, **kw),
            "ascii_grid": lambda g=None: P.ascii_grid(s.grid if g is None else g),
            "diff_summary": lambda a=None, b=None: P.diff_summary(s.prev_grid if a is None else a, s.grid if b is None else b),
        }
        self._injected = set(values)
        self.ns.update(values)

    # --- execution ----------------------------------------------------------
    def run(self, code: str, generated_tokens: int = 0) -> ToolOutcome:
        """Execute ``code``. Blocking; call from a worker thread."""
        call = _Call(generated_tokens=generated_tokens, limit=self.cfg.max_actions_per_call)
        self._call = call
        self._inject()
        out = io.StringIO()

        def _print(*args, sep=" ", end="\n", file=None, flush=False):
            out.write(sep.join(str(a) for a in args) + end)

        self.ns["__builtins__"]["print"] = _print
        status: dict[str, Any] = {"error": None}

        def target() -> None:
            try:
                exec(compile(code, "<tool>", "exec"), self.ns)
            except ToolTimeout:
                status["error"] = "timeout"
            except SystemExit:
                status["error"] = "SystemExit is not allowed"
            except BaseException:  # noqa: BLE001 - the model's code may raise anything
                status["error"] = traceback.format_exc(limit=4)

        thread = threading.Thread(target=target, daemon=True, name="arcd-tool")
        thread.start()
        thread.join(self.cfg.tool_timeout_s)
        timed_out = False
        if thread.is_alive():
            timed_out = True
            call.cancelled.set()
            if thread.ident is not None:
                ctypes.pythonapi.PyThreadState_SetAsyncExc(ctypes.c_ulong(thread.ident), ctypes.py_object(ToolTimeout))
            thread.join(5.0)
        self._call = None

        for key in list(self.ns):
            if key not in self._injected and key not in ("__builtins__", "__name__", "np"):
                self._user_keys.add(key)

        results = call.results
        parts: list[str] = []
        text_out = out.getvalue()
        if text_out:
            parts.append(text_out.rstrip("\n"))
        if results:
            parts.append("ACTIONS:\n" + "\n".join(r.line() for r in results))
            frames = len(self.session.animation)
            if frames > 0 and any(r.ok for r in results):
                parts.append(f"(the last action produced {frames} intermediate animation frame(s) in `animation_frames`)")
        if timed_out or status["error"] == "timeout":
            parts.append(f"ERROR: the call exceeded {self.cfg.tool_timeout_s:.0f}s and was stopped. Actions already executed stay executed.")
        elif status["error"]:
            parts.append("ERROR:\n" + str(status["error"]).rstrip())
        if not parts:
            parts.append("(no output)")
        body = truncate_middle("\n".join(parts), int(self.cfg.tool_output_tokens * self.cfg.chars_per_token))
        ok_results = [r for r in results if r.ok]
        return ToolOutcome(
            text=body,
            results=results,
            error=bool(status["error"]) or timed_out,
            timed_out=timed_out,
            executed_actions=len(ok_results),
            level_up=any(r.level_up for r in ok_results),
            game_over=any(r.game_over for r in ok_results),
            won=any(r.won for r in ok_results),
        )
