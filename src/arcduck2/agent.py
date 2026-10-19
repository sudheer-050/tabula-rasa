"""The per-game agent loop."""
from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from . import perception as P
from . import prompts as T
from .config import Config
from .llm import ContextTooLong, LLMClient, LLMReply, ModelUnavailable
from .sandbox import Sandbox, ToolOutcome
from .session import GameSession

MAX_IDLE_TURNS_IN_BURST = 30
MAX_CONTEXT_RETRIES = 3
MAX_INTERNAL_ERRORS = 8
AUTO_RESET_AFTER_TURNS = 3


@dataclass
class Stats:
    model_calls: int = 0
    generated_tokens: int = 0
    prompt_tokens: int = 0
    cached_tokens: int = 0
    tool_calls: int = 0
    tool_errors: int = 0
    trims: int = 0
    no_tool_turns: int = 0
    model_failures: int = 0
    replay_actions: int = 0
    blocked_calls: int = 0
    commit_plan_calls: int = 0


class GameAgent:
    def __init__(self, game: Any, cfg: Config, llm: LLMClient, scheduler: Any = None,
                 clock: Callable[[], float] = time.monotonic, game_key: str = ""):
        self.game = game
        self.cfg = cfg
        self.llm = llm
        self.scheduler = scheduler
        self.clock = clock
        self.game_key = game_key or str(getattr(game, "game_id", "game"))
        self.session = GameSession(game)
        self.sandbox = Sandbox(self.session, cfg)
        self.stats = Stats()
        self.turns: list[list[dict]] = []
        self.brief: dict = {"role": "user", "content": ""}
        self.calib = 1.0
        self.started = clock()
        self.level_started = clock()
        self.stop_reason = ""
        self._held = False
        self._idle_turns = 0
        self.internal_errors = 0
        self._over_turns = 0
        self._image_failed = False
        self._last_est = 0
        self._summary = ""
        self._trace_path: Path | None = None
        if cfg.debug_dir:
            d = Path(cfg.debug_dir)
            d.mkdir(parents=True, exist_ok=True)
            self._trace_path = d / f"{self.game_key}.jsonl"
        self.brief = self._make_intro()

    # --- scheduler-facing state ------------------------------------------
    @property
    def level_index(self) -> int:
        return self.session.levels_completed

    @property
    def level_actions(self) -> int:
        return self.session.level_actions

    @property
    def levels_left(self) -> int:
        return max(0, self.session.number_of_levels - self.session.levels_completed)

    @property
    def seconds_on_level(self) -> float:
        return self.clock() - self.level_started

    # --- text builders -----------------------------------------------------
    def _notes_text(self) -> str:
        notes = self.sandbox.ns.get("notes")
        if isinstance(notes, str) and notes.strip():
            return "Your notes so far:\n" + notes.strip()[:4000] + "\n"
        if isinstance(notes, (list, tuple)) and notes:
            lines = [f"- {str(n)[:400]}" for n in list(notes)[-40:]]
            return "Your notes so far:\n" + "\n".join(lines) + "\n"
        return ""

    def _actions_text(self) -> str:
        return ", ".join(self.session.available_names(self.cfg.expose_undo))

    def _efficiency_note(self, spent: int) -> str:
        if spent >= self.cfg.efficiency_critical_actions:
            return T.EFFICIENCY_CRITICAL.format(spent=spent)
        if spent >= self.cfg.efficiency_warn_actions:
            return T.EFFICIENCY_WARN.format(spent=spent)
        return ""

    def _image_part(self, diff_against=None) -> dict | None:
        """The board as an image part, or None if rendering fails. A game must never be lost to
        a picture: without the image the model still gets the full text board."""
        try:
            png = P.render_png(self.session.grid, scale=self.cfg.image_scale, diff_against=diff_against)
            return {"type": "image_url", "image_url": {"url": P.png_data_url(png)}}
        except Exception as exc:  # noqa: BLE001
            if not self._image_failed:
                self._image_failed = True
                self._trace("image_render_failed", err=f"{type(exc).__name__}: {exc}"[:200])
            return None

    def _user(self, text: str, image: bool = True, diff_against=None) -> dict:
        part = self._image_part(diff_against) if image else None
        if part is None:
            return {"role": "user", "content": text}
        return {"role": "user", "content": [{"type": "text", "text": text}, part]}

    def _make_intro(self) -> dict:
        s = self.session
        text = T.INTRO_TEMPLATE.format(
            levels=s.number_of_levels, level=s.levels_completed + 1, done=s.levels_completed,
            actions=self._actions_text(), notes=self._notes_text(), extra="", board=P.ascii_grid(s.grid),
        )
        return self._user(text)

    def _make_memory(self) -> dict:
        s = self.session
        notes = self._notes_text() or "(no notes were written)\n"
        if self._summary:
            notes = f"Summary of what was learned before older messages were trimmed:\n{self._summary}\n\n" + notes
        fatal = s.fatal_summary()
        text = T.MEMORY_TEMPLATE.format(
            notes=notes.rstrip("\n"), fatal=fatal, level=s.levels_completed + 1, levels=s.number_of_levels,
            spent=s.level_actions, actions=self._actions_text(), board=P.ascii_grid(s.grid),
        ) + self._efficiency_note(s.level_actions)
        return self._user(text)

    # --- context management ------------------------------------------------
    @staticmethod
    def _message_chars(msg: dict) -> tuple[int, int]:
        chars, images = 0, 0
        content = msg.get("content")
        if isinstance(content, str):
            chars += len(content)
        elif isinstance(content, list):
            for part in content:
                if part.get("type") == "text":
                    chars += len(part.get("text", ""))
                elif part.get("type") == "image_url":
                    images += 1
        for key in ("reasoning_content", "reasoning"):
            if isinstance(msg.get(key), str):
                chars += len(msg[key])
        for tc in msg.get("tool_calls") or []:
            chars += len(json.dumps(tc))
        return chars, images

    def messages(self) -> list[dict]:
        out = [{"role": "system", "content": T.SYSTEM_PROMPT}, self.brief]
        for turn in self.turns:
            out.extend(turn)
        return out

    def estimate_tokens(self, msgs: list[dict] | None = None) -> int:
        total = 0
        for m in (msgs if msgs is not None else self.messages()):
            chars, images = self._message_chars(m)
            total += chars / self.cfg.chars_per_token + images * self.cfg.image_tokens + 6
        return int(total * self.calib)

    @staticmethod
    def _has_image(m: dict) -> bool:
        return isinstance(m.get("content"), list) and any(p.get("type") == "image_url" for p in m["content"])

    def _strip_old_images(self) -> int:
        """Keep only the newest ``keep_images`` board images; older ones become a text stub.

        Done every turn: the stripped message is always near the END of the prompt, so the
        server's cached prefix stays valid. Doing it only on overflow would rewrite most of
        the history in one go and throw the whole cache away."""
        carriers = [self.brief] + [m for turn in self.turns for m in turn]
        with_image = [m for m in carriers if self._has_image(m)]
        drop = with_image[: max(0, len(with_image) - self.cfg.keep_images)]
        for m in drop:
            text = " ".join(p.get("text", "") for p in m["content"] if p.get("type") == "text")
            m["content"] = text + "\n[board image omitted]"
        return len(drop)

    def _strip_all_images(self) -> None:
        for m in [self.brief] + [m for turn in self.turns for m in turn]:
            if self._has_image(m):
                text = " ".join(p.get("text", "") for p in m["content"] if p.get("type") == "text")
                m["content"] = text + "\n[board image omitted]"

    async def _maybe_capture_summary(self) -> None:
        """Ask the model to summarise what it has learned, right before eviction drops it.
        Opt-in (``cfg.use_llm_summary``): costs one extra, bounded model call per trim, in
        exchange for carrying real understanding into the post-trim memory brief instead of
        only the static template plus whatever `notes` happened to catch."""
        request = self._user(T.SUMMARY_REQUEST, image=False)
        try:
            reply = await asyncio.to_thread(self.llm.chat, self.messages() + [request], self.cfg.summary_max_tokens)
        except (ContextTooLong, ModelUnavailable) as exc:
            self._trace("summary_failed", err=f"{type(exc).__name__}: {exc}"[:200])
            return
        except Exception as exc:  # noqa: BLE001 - a failed summary must not abort the trim
            self._trace("summary_failed", err=f"{type(exc).__name__}: {exc}"[:200])
            return
        text = (reply.content or "").strip()
        if not text:
            return
        self._summary = text
        self.stats.model_calls += 1
        self.stats.generated_tokens += reply.completion_tokens
        self.stats.prompt_tokens += reply.prompt_tokens
        self.stats.cached_tokens += reply.cached_tokens
        self._trace("summary", chars=len(text))

    async def trim_context(self, force: bool = False) -> bool:
        """Drop old images, then whole old turns, down to the drain target. True if anything changed."""
        too_many_turns = len(self.turns) > self.cfg.max_history_turns
        if not force and not too_many_turns and self.estimate_tokens() <= self.cfg.context_budget_tokens:
            return False
        if self.cfg.use_llm_summary:
            await self._maybe_capture_summary()
        changed = self._strip_old_images() > 0
        evicted = False
        if too_many_turns:
            drop = min(len(self.turns) - 2, self.cfg.history_turn_drain)
            if drop > 0:
                del self.turns[:drop]
                evicted = True
        target = self.cfg.context_drain_target_tokens if not force else int(self.cfg.context_drain_target_tokens * 0.6)
        while len(self.turns) > 2 and self.estimate_tokens() > target:
            self.turns.pop(0)
            evicted = True
        if evicted:
            self.brief = self._make_memory()
            self.stats.trims += 1
        return changed or evicted

    # --- tracing -----------------------------------------------------------
    def _trace(self, event: str, **kw: Any) -> None:
        if self._trace_path is None:
            return
        rec = {"t": round(self.clock() - self.started, 1), "ev": event, "lvl": self.session.levels_completed,
               "la": self.session.level_actions, "ta": self.session.total_actions, **kw}
        try:
            with self._trace_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec) + "\n")
        except OSError:
            pass

    # --- scheduler slot ----------------------------------------------------
    async def _acquire(self) -> None:
        if self.scheduler is not None and not self._held:
            await self.scheduler.acquire(self)
            self._held = True

    def _release(self) -> None:
        if self.scheduler is not None and self._held:
            self.scheduler.release(self)
            self._held = False

    # --- one model turn ----------------------------------------------------
    def _assistant_message(self, reply: LLMReply) -> dict:
        msg: dict[str, Any] = {"role": "assistant", "content": reply.content or ""}
        if reply.reasoning:
            msg[self.cfg.reasoning_field] = reply.reasoning
        if reply.raw_tool_calls:
            msg["tool_calls"] = reply.raw_tool_calls
        return msg

    @staticmethod
    def _tool_replies(reply: LLMReply, text: str) -> list[dict]:
        """One tool message per tool call; only the first call is executed."""
        msgs = [{"role": "tool", "tool_call_id": reply.tool_calls[0].id, "content": text}]
        for extra in reply.tool_calls[1:]:
            msgs.append({"role": "tool", "tool_call_id": extra.id,
                         "content": "Not executed: only one python call per turn is run. Call it again if still needed."})
        return msgs

    async def _call_model(self) -> LLMReply | None:
        for _ in range(MAX_CONTEXT_RETRIES + 1):
            self._last_est = self.estimate_tokens()
            try:
                return await asyncio.to_thread(self.llm.chat, self.messages())
            except ContextTooLong:
                if not await self.trim_context(force=True):
                    self.turns = self.turns[-1:]
                    self.brief = self._make_memory()
            except ModelUnavailable as exc:
                self.stats.model_failures += 1
                self._trace("model_failure", err=str(exc)[:200])
                if str(exc).startswith("HTTP 400"):
                    # The server rejected the request itself. Shrink what we send before the next try.
                    self._strip_all_images()
                    await self.trim_context(force=True)
                return None
            except Exception as exc:  # noqa: BLE001 - e.g. a malformed payload; retry rather than abandon the game
                self.stats.model_failures += 1
                self._trace("model_error", err=f"{type(exc).__name__}: {exc}"[:200])
                await asyncio.sleep(1.0)
                return None
        return None

    async def _turn(self) -> dict:
        """Run one model call plus its tool call. Returns flags about what happened."""
        flags = {"trimmed": False, "level_up": False, "game_over": False, "acted": False}
        self._strip_old_images()
        flags["trimmed"] = await self.trim_context()
        reply = await self._call_model()
        if reply is None:
            return {**flags, "failed": True}
        self.stats.model_calls += 1
        self.stats.generated_tokens += reply.completion_tokens
        self.stats.prompt_tokens += reply.prompt_tokens
        self.stats.cached_tokens += reply.cached_tokens
        if reply.prompt_tokens and self._last_est:
            ratio = max(0.5, min(2.0, reply.prompt_tokens / max(1, self._last_est / max(self.calib, 1e-6))))
            self.calib = 0.7 * self.calib + 0.3 * ratio

        turn = [self._assistant_message(reply)]
        code = reply.code
        if not code.strip():
            self.stats.no_tool_turns += 1
            if reply.tool_calls:
                turn.extend(self._tool_replies(reply, "Empty code. " + T.NO_TOOL_NUDGE))
            else:
                turn.append({"role": "user", "content": T.NO_TOOL_NUDGE})
            self.turns.append(turn)
            self._trace("no_tool", fin=reply.finish_reason, gen=reply.completion_tokens)
            return flags

        self.stats.tool_calls += 1
        before = self.session.grid
        outcome: ToolOutcome = await asyncio.to_thread(self.sandbox.run, code, reply.completion_tokens)
        if outcome.error:
            self.stats.tool_errors += 1
        self.stats.blocked_calls += outcome.blocked
        if outcome.used_commit_plan:
            self.stats.commit_plan_calls += 1
        flags.update(level_up=outcome.level_up, game_over=outcome.game_over, acted=outcome.executed_actions > 0)

        s = self.session
        # A level-up/game-over/win ending a call isn't evidence of indecisive drip play --
        # it's a natural stop, already handled elsewhere (session.py resets the streak on
        # level_up; fatal-path memory handles game_over). Only a call that ran small and
        # then just... stopped, with nothing special happening, is the drip signal.
        if outcome.executed_actions > 0 and not (outcome.level_up or outcome.game_over or outcome.won):
            if outcome.executed_actions < self.cfg.min_commit_batch:
                s.probe_streak += 1
            else:
                s.probe_streak = 0
        text = outcome.text + "\n" + T.STATUS_TEMPLATE.format(
            level=s.levels_completed + 1, levels=s.number_of_levels, spent=s.level_actions, actions=self._actions_text()
        ) + self._efficiency_note(s.level_actions)
        if outcome.error and not outcome.timed_out:
            text += "\n" + T.CODE_ERROR_HINT
        if reply.tool_calls:
            turn.extend(self._tool_replies(reply, text))
            follow_in_user = ""
        else:
            follow_in_user = text

        observation = None
        if outcome.won:
            observation = None
        elif outcome.level_up:
            self.level_started = self.clock()
            observation = self._user(T.LEVEL_UP_TEMPLATE.format(
                level=s.levels_completed + 1, levels=s.number_of_levels, done=s.levels_completed,
                actions=self._actions_text(), board=P.ascii_grid(s.grid)))
        elif outcome.game_over and s.is_game_over:
            cause = f"It happened on level {s.levels_completed + 1} after {s.level_actions} actions on this level."
            observation = self._user(T.GAME_OVER_TEMPLATE.format(
                cause=cause, fatal=s.fatal_summary(), level=s.levels_completed + 1, levels=s.number_of_levels,
                board=P.ascii_grid(s.grid)), diff_against=before)
        elif outcome.executed_actions > 0:
            observation = self._user("Board after your actions (changed cells outlined in magenta):", diff_against=before)

        if follow_in_user:
            turn.append({"role": "user", "content": follow_in_user})
        if observation is not None:
            turn.append(observation)
        self.turns.append(turn)
        self._trace("tool", acts=outcome.executed_actions, err=outcome.error, up=outcome.level_up,
                    over=outcome.game_over, gen=reply.completion_tokens, ptok=reply.prompt_tokens, cached=reply.cached_tokens,
                    blocked=outcome.blocked, via_commit=outcome.used_commit_plan, streak=self.session.probe_streak)
        return flags

    async def _auto_reset_if_stalled(self, flags: dict) -> None:
        """After GAME_OVER only RESET is accepted. Free analysis is welcome, but a model that
        keeps thinking without ever resetting would stall the game, so reset for it. The RESET
        costs the same one action the model would have spent."""
        if not self.session.is_game_over or not self.session.is_playing:
            self._over_turns = 0
            return
        self._over_turns += 1
        if self._over_turns < AUTO_RESET_AFTER_TURNS:
            return
        self._over_turns = 0
        res = await asyncio.to_thread(self.session.execute, "RESET", None, None, expose_undo=self.cfg.expose_undo)
        self._trace("auto_reset", ok=res.ok)
        if res.ok:
            self.turns.append([self._user(
                f"You did not call RESET after GAME_OVER, so it was issued for you (it costs one action). "
                f"You are back at the start of level {self.session.levels_completed + 1}. "
                f"Do not repeat the sequence that failed.")])

    async def _replay_known_best(self) -> bool:
        """At the start of a fresh attempt (right after any RESET), replay a known-good
        sequence for this level with no model call, stopping the moment an action fails to
        execute or the attempt ends. Prefers ``best`` (a sequence proven to clear the level
        outright) over ``frontier`` (a longer sequence only proven not to be fatal); either
        way the model only takes back over once play reaches ground it hasn't already
        covered. The game is deterministic, so a clean replay reaches the same point it did
        before at zero LLM cost. Returns True if a replay happened (regardless of outcome),
        so the caller skips a model turn this pass."""
        s = self.session
        if s.path or not s.is_playing:
            return False
        level = s.levels_completed
        source = "best" if s.best.get(level) else "frontier" if s.frontier.get(level) else None
        if source is None:
            return False
        sequence = s.best[level] if source == "best" else s.frontier[level]
        before = s.grid
        replayed = []
        for engine, x, y in sequence:
            res = await asyncio.to_thread(s.execute, engine, x, y, expose_undo=self.cfg.expose_undo)
            if not res.ok:
                break
            replayed.append(res)
            self.stats.replay_actions += 1
            if res.level_up or res.game_over or res.won:
                break
        if not replayed:
            return False
        self._trace("replay", n=len(replayed), of=len(sequence), level=level, source=source,
                    up=replayed[-1].level_up, over=replayed[-1].game_over)
        last = replayed[-1]
        header = (f"[cascade] Replayed {len(replayed)}/{len(sequence)} known-good action(s) for level "
                  f"{level + 1} with no model cost ({'a proven-winning sequence' if source == 'best' else 'the deepest point previously reached without GAME_OVER'}).")
        if last.won:
            return True
        if last.level_up:
            self.level_started = self.clock()
            text = header + "\n" + T.LEVEL_UP_TEMPLATE.format(
                level=s.levels_completed + 1, levels=s.number_of_levels, done=s.levels_completed,
                actions=self._actions_text(), board=P.ascii_grid(s.grid))
            self.turns.append([self._user(text)])
        elif last.game_over:
            # Contradicts determinism (this exact sequence was fine before); treat it as
            # stale rather than replaying the same failure forever. session.execute() has
            # already recorded the new fatal path on its own.
            del (s.best if source == "best" else s.frontier)[level]
            cause = (f"It happened on level {s.levels_completed + 1} after {len(replayed)} replayed "
                     f"action(s) that were fine before — unexpected; do not retry that sequence.")
            text = header + "\n" + T.GAME_OVER_TEMPLATE.format(
                cause=cause, fatal=s.fatal_summary(), level=s.levels_completed + 1, levels=s.number_of_levels,
                board=P.ascii_grid(s.grid))
            self.turns.append([self._user(text, diff_against=before)])
        else:
            text = header + " Board now (changed cells outlined in magenta):"
            self.turns.append([self._user(text, diff_against=before)])
        return True

    # --- main loop ---------------------------------------------------------
    async def run(self, deadline: float | None = None) -> None:
        try:
            while self.session.is_playing:
                if deadline is not None and self.clock() >= deadline:
                    self.stop_reason = "deadline"
                    break
                if self.stats.model_failures >= self.cfg.max_consecutive_model_failures:
                    self.stop_reason = "model unavailable"
                    break
                if await self._replay_known_best():
                    continue                      # no model call spent on known ground
                await self._acquire()
                if (deadline is not None and self.clock() >= deadline) or not self.session.is_playing:
                    continue                      # woke up from the queue after the deadline: do not start a turn
                try:
                    flags = await self._turn()
                except asyncio.CancelledError:
                    raise
                except Exception as exc:  # noqa: BLE001 - an internal bug must not end the game outright
                    self.internal_errors += 1
                    self._trace("internal_error", err=f"{type(exc).__name__}: {exc}"[:300])
                    if self.internal_errors >= MAX_INTERNAL_ERRORS:
                        self.stop_reason = f"internal errors: {type(exc).__name__}: {exc}"[:200]
                        break
                    await asyncio.sleep(0.5)
                    continue
                if flags.get("failed"):
                    continue
                self.internal_errors = 0
                self.stats.model_failures = 0
                await self._auto_reset_if_stalled(flags)
                self._idle_turns = 0 if flags["acted"] else self._idle_turns + 1
                if flags["trimmed"] or flags["level_up"] or self._idle_turns >= MAX_IDLE_TURNS_IN_BURST:
                    if self._idle_turns >= MAX_IDLE_TURNS_IN_BURST:
                        self._idle_turns = 0
                    self._release()
                await asyncio.sleep(0)
        finally:
            self._release()
            note = self.stop_reason or ("won" if self.session.is_won else "ended")
            run = getattr(self.game, "game_run", None)
            if run is not None:
                run.solver_note = (f"{note}; calls={self.stats.model_calls} gen={self.stats.generated_tokens} "
                                   f"trims={self.stats.trims} tool_err={self.stats.tool_errors} notool={self.stats.no_tool_turns} "
                                   f"replay={self.stats.replay_actions} blocked={self.stats.blocked_calls} "
                                   f"commit_plan={self.stats.commit_plan_calls}")
            self._trace("end", note=note)
