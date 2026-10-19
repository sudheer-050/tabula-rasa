"""Priority-based slot scheduler.

There are many more games than concurrent model streams. A game holds a slot
for a stretch of play and re-queues at natural boundaries (a level change, a
context trim, a stall). When a slot frees, the waiting game with the highest
expected score gain per unit of GPU time runs next.

priority = (A + B) * C
  A  gain from finishing the current level now: level weight * efficiency decay
  B  value of the levels that would still be open afterwards (fades at the end)
  C  likelihood the current level is still solvable given effort already spent
The constants are heuristics carried over from published fits of attempt data;
they only order games, so moderate error in them is harmless.
"""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Any, Callable

AGING_PER_SECOND = 0.5 / 1800.0


def priority(level_index: int, level_actions: int, seconds_on_level: float, levels_total: int,
             elapsed_fraction: float = 0.0, tail_fade: float = 0.2) -> float:
    n = max(1, int(levels_total))
    level_no = max(1, level_index + 1)
    a = max(0, level_actions)
    gain = level_no * (25.0 / (25.0 + a)) ** 2 * (55.0 / (n * (n + 1) / 2.0))
    remaining_after = max(0, n - level_no)
    cont = 8.0 if remaining_after >= 3 else {2: 7.0, 1: 5.0}.get(remaining_after, 0.0)
    if tail_fade > 0 and elapsed_fraction > 1.0 - tail_fade:
        cont *= max(0.0, (1.0 - elapsed_fraction) / tail_fade)
    t = max(0.0, seconds_on_level)
    likelihood = 0.25 * 2.0 ** (-((a / 115.0) ** 2)) + 0.75 * 2.0 ** (-((t / 62000.0) ** 2))
    return (gain + cont) * likelihood


@dataclass
class _Waiter:
    agent: Any
    future: asyncio.Future
    since: float


class PriorityScheduler:
    def __init__(self, slots: int, total_runtime_s: float, tail_fade: float = 0.2,
                 clock: Callable[[], float] = time.monotonic):
        if slots < 1:
            raise ValueError("slots must be >= 1")
        self.slots = slots
        self.total_runtime_s = max(1.0, total_runtime_s)
        self.tail_fade = tail_fade
        self.clock = clock
        self.started = clock()
        self.active: set[int] = set()
        self.waiting: list[_Waiter] = []

    def score(self, agent: Any, waited: float = 0.0) -> float:
        elapsed = min(1.0, (self.clock() - self.started) / self.total_runtime_s)
        base = priority(agent.level_index, agent.level_actions, agent.seconds_on_level,
                        agent.session.number_of_levels, elapsed, self.tail_fade)
        return base + waited * AGING_PER_SECOND

    async def acquire(self, agent: Any) -> None:
        key = id(agent)
        if key in self.active:
            return
        if len(self.active) < self.slots and not self.waiting:
            self.active.add(key)
            return
        fut: asyncio.Future = asyncio.get_running_loop().create_future()
        waiter = _Waiter(agent, fut, self.clock())
        self.waiting.append(waiter)
        self._dispatch()
        try:
            await fut
        except asyncio.CancelledError:
            if fut.done() and not fut.cancelled():
                self.active.discard(key)  # slot was granted just before the cancel landed
                self._dispatch()
            elif waiter in self.waiting:
                self.waiting.remove(waiter)
            raise

    def release(self, agent: Any) -> None:
        self.active.discard(id(agent))
        self._dispatch()

    def _dispatch(self) -> None:
        now = self.clock()
        while len(self.active) < self.slots and self.waiting:
            best = max(self.waiting, key=lambda w: self.score(w.agent, now - w.since))
            self.waiting.remove(best)
            if best.future.done():
                continue
            self.active.add(id(best.agent))
            best.future.set_result(None)
