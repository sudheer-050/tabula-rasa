"""Drive DuckV2Solver through TAAF's real Benchmark on public games (online API), fake model."""
import asyncio
import logging
import os
import sys
import time
from dataclasses import replace
from datetime import datetime, timedelta
from pathlib import Path

import arc_agi
import taaf.benchmark
import taaf.game_api as ga

sys.path.insert(0, os.path.dirname(__file__))
from smoke_online import FakeModel  # noqa: E402  (module guards its own main)

MODE = sys.argv[1] if len(sys.argv) > 1 else "deadline"   # deadline | cancel
IDS = ["ls20-9607627b", "ft09-0d8bbf25", "vc33-5430563c"]


async def main():
    from arcduck2.config import Config
    from arcduck2.solver import DuckV2Solver

    logging.getLogger("arc_agi").setLevel(logging.WARNING)
    spec = ga.ArcadeSpec(operation_mode=arc_agi.OperationMode.ONLINE, environments_dir="")
    games = [ga.GameAPI(env_name=i, arcade_spec=spec) for i in IDS]
    cfg = replace(Config(), slots=2, tool_timeout_s=20.0)
    solver = DuckV2Solver(
        config=cfg,
        hard_deadline_epoch=time.time() + (40 if MODE == "deadline" else 3600),
    )
    job = Path("smoke_job"); job.mkdir(exist_ok=True)
    bm = taaf.benchmark.Benchmark(games=games, solver=solver, job_dir=job)
    # the real notebook pins these before the run; keep the same shape
    soft_end = datetime.now() + timedelta(seconds=25) if MODE == "cancel" else None
    orig = DuckV2Solver._setup

    def patched_setup(self):
        orig(self)
        self._llm = FakeModel()

    DuckV2Solver._setup = patched_setup
    await bm.run(soft_end_time=soft_end, minimal_diagnostics=True)
    for g in bm.games_played if hasattr(bm, "games_played") else []:
        pass
    print("BENCHMARK DONE; attrs:", [a for a in dir(bm) if "game" in a.lower() or "run" in a.lower()][:20])


asyncio.run(main())
