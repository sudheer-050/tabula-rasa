import asyncio
import pytest
from arcduck2.scheduler import PriorityScheduler, priority


def test_priority_properties():
    fresh = priority(0, 0, 0, 10)
    assert fresh > 0
    assert priority(0, 200, 0, 10) < fresh                    # effort spent lowers odds
    assert priority(3, 0, 0, 10) > priority(0, 0, 0, 10) - 8   # later level is worth more (gain term)
    # tail fade removes continuation value near the end
    assert priority(0, 0, 0, 10, elapsed_fraction=0.99) < priority(0, 0, 0, 10, elapsed_fraction=0.0)
    # last level: weight 10, no continuation value -> exactly the gain term
    assert abs(priority(9, 0, 0, 10) - 10.0) < 1e-9
    assert abs(priority(0, 0, 0, 10) - 9.0) < 1e-9             # 1 + continuation 8
    assert priority(0, 0, 0, 1) > 0                            # single-level game is fine
    assert priority(0, 0, 0, 0) > 0                            # degenerate input does not crash


class FakeSession:
    def __init__(self, n): self.number_of_levels = n


class FakeAgent:
    def __init__(self, level=0, actions=0, n=10, t=0.0):
        self.level_index, self.level_actions, self.seconds_on_level = level, actions, t
        self.session = FakeSession(n)


async def test_slot_limit_and_priority_order():
    s = PriorityScheduler(2, 1000.0)
    a, b = FakeAgent(), FakeAgent()
    await s.acquire(a); await s.acquire(b)
    assert len(s.active) == 2
    low = FakeAgent(actions=500)           # low priority
    high = FakeAgent(level=2, actions=0)   # high priority
    order = []

    async def run(name, ag):
        await s.acquire(ag); order.append(name)

    t1 = asyncio.create_task(run("low", low))
    t2 = asyncio.create_task(run("high", high))
    await asyncio.sleep(0.01)
    assert order == []
    s.release(a)
    await asyncio.sleep(0.01)
    assert order == ["high"]
    s.release(b)
    await asyncio.sleep(0.01)
    assert order == ["high", "low"]
    await asyncio.gather(t1, t2)


async def test_cancel_while_waiting_frees_nothing_wrongly():
    s = PriorityScheduler(1, 1000.0)
    a = FakeAgent(); await s.acquire(a)
    w = FakeAgent()
    task = asyncio.create_task(s.acquire(w))
    await asyncio.sleep(0.01)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert not s.waiting and len(s.active) == 1
    s.release(a)
    assert not s.active
    await s.acquire(FakeAgent())  # still usable


async def test_reacquire_is_idempotent_and_invalid_slots():
    s = PriorityScheduler(1, 10.0)
    a = FakeAgent()
    await s.acquire(a); await s.acquire(a)
    assert len(s.active) == 1
    with pytest.raises(ValueError):
        PriorityScheduler(0, 10.0)
