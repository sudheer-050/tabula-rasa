"""TAAF ``Solver`` that plays every game with ``GameAgent`` behind one scheduler."""
from __future__ import annotations

import asyncio
import contextlib
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from typing import Any

from taaf.solver import Solver

from .agent import GameAgent
from .config import Config
from .llm import LLMClient
from .scheduler import PriorityScheduler


def _safe_finish(game: Any, generated_tokens: int = 0) -> None:
    run = getattr(game, "game_run", None)
    if run is None or run.final_score is not None:
        return
    with contextlib.suppress(Exception):
        game.finish_game(generated_tokens=generated_tokens)


@dataclass
class DuckV2Solver(Solver):
    label: str = "arcduck2"
    config: Config | None = None
    # Wall-clock epoch (time.time()) after which every game is wound down. Set this
    # to leave a safety margin before the notebook's hard limit.
    hard_deadline_epoch: float | None = None
    max_runtime_s_per_game: float | None = None
    stop_on_error: bool = False
    _cfg: Config | None = field(default=None, init=False, repr=False)
    _llm: Any = field(default=None, init=False, repr=False)
    _executor: Any = field(default=None, init=False, repr=False)
    agents: list = field(default_factory=list, init=False, repr=False)

    def _setup(self) -> None:
        cfg = self.config or Config.from_env()
        if self.max_runtime_s_per_game is not None:
            cfg = replace(cfg, max_runtime_s_per_game=float(self.max_runtime_s_per_game))
        self._cfg = cfg
        self._llm = LLMClient(cfg)

    def _teardown(self) -> None:
        if self._executor is not None:
            self._executor.shutdown(wait=False, cancel_futures=True)

    def __getstate__(self) -> dict[str, Any]:
        state = dict(self.__dict__)
        for key in ("_llm", "_executor", "agents"):
            state[key] = None if key != "agents" else []
        return state

    def _deadline_monotonic(self) -> float | None:
        if self.hard_deadline_epoch is None:
            return None
        return time.monotonic() + (self.hard_deadline_epoch - time.time())

    async def _run_games(self, games: list[Any]) -> None:
        assert self._cfg is not None and self._llm is not None, "setup() must run first"
        cfg = self._cfg
        loop = asyncio.get_running_loop()
        self._executor = ThreadPoolExecutor(max_workers=cfg.slots * 2 + 8, thread_name_prefix="arcd")
        loop.set_default_executor(self._executor)

        scheduler = PriorityScheduler(cfg.slots, cfg.max_runtime_s_per_game, cfg.tail_fade_fraction)
        hard = self._deadline_monotonic()
        started = time.monotonic()
        per_game_deadline = started + cfg.max_runtime_s_per_game
        deadline = per_game_deadline if hard is None else min(per_game_deadline, hard)

        self.agents = []
        for i, game in enumerate(games):
            try:
                key = f"{getattr(game, 'game_id', 'game')}-{i}"
                self.agents.append(GameAgent(game, cfg, self._llm, scheduler, game_key=key))
            except Exception as exc:  # noqa: BLE001 - one bad game must not sink the run
                self._note(game, f"agent init failed: {type(exc).__name__}: {exc}")
                _safe_finish(game)

        tasks = [asyncio.create_task(self._play(agent, deadline)) for agent in self.agents]
        helpers = [asyncio.create_task(self._warmup_watch()),
                   asyncio.create_task(self._status_loop(scheduler, started))]
        try:
            await asyncio.gather(*tasks)
        except asyncio.CancelledError:
            for t in tasks:
                t.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            for game in games:
                _safe_finish(game)
            raise
        finally:
            for h in helpers:
                h.cancel()
            await asyncio.gather(*helpers, return_exceptions=True)
            for game in games:
                _safe_finish(game)

    async def _warmup_watch(self) -> None:
        """Kaggle may cancel a run in which no game action happens for a long time. If nothing has
        acted after ``warmup_after_s`` (slow model start-up), spend one RESET on the first game."""
        assert self._cfg is not None
        await asyncio.sleep(self._cfg.warmup_after_s)
        if sum(a.session.total_actions for a in self.agents) > 0:
            return
        for agent in self.agents:
            if agent.session.is_playing:
                res = await asyncio.to_thread(agent.session.execute, "RESET", None, None)
                print(f"[arcduck2] warmup RESET on {agent.game_key}: ok={res.ok}", flush=True)
                return

    async def _status_loop(self, scheduler: PriorityScheduler, started: float) -> None:
        assert self._cfg is not None
        while True:
            await asyncio.sleep(self._cfg.status_every_s)
            try:
                print(self.status_line(scheduler, started), flush=True)
            except Exception:  # noqa: BLE001 - status output must never affect the run
                pass

    def status_line(self, scheduler: PriorityScheduler, started: float) -> str:
        done = sum(1 for a in self.agents if a.session.is_won)
        levels = sum(a.session.levels_completed for a in self.agents)
        actions = sum(a.session.total_actions for a in self.agents)
        calls = sum(a.stats.model_calls for a in self.agents)
        gen = sum(a.stats.generated_tokens for a in self.agents)
        cached = sum(a.stats.cached_tokens for a in self.agents)
        prompt = sum(a.stats.prompt_tokens for a in self.agents)
        elapsed = time.monotonic() - started
        hit = f"{100.0 * cached / prompt:.0f}%" if prompt else "n/a"
        return (f"[arcduck2 {elapsed / 60:.0f}min] games={len(self.agents)} won={done} levels_done={levels} "
                f"actions={actions} model_calls={calls} gen_tok={gen} cache_hit={hit} "
                f"active={len(scheduler.active)} waiting={len(scheduler.waiting)}")

    async def _play(self, agent: GameAgent, deadline: float) -> None:
        game = agent.game
        try:
            await agent.run(deadline)
        except asyncio.CancelledError:
            _safe_finish(game, generated_tokens=0)
            raise
        except Exception as exc:  # noqa: BLE001
            self._note(game, f"agent crashed: {type(exc).__name__}: {exc}\n{traceback.format_exc(limit=3)}")
            if self.stop_on_error:
                raise
        finally:
            _safe_finish(game)

    @staticmethod
    def _note(game: Any, text: str) -> None:
        run = getattr(game, "game_run", None)
        if run is not None:
            run.solver_note = text[:500]
