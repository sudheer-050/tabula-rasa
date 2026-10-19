"""Run DuckV2Solver on real public games through ARC's public API with a scripted fake model."""
import asyncio, os, sys, time, logging
from dataclasses import replace

import arc_agi
import taaf.game
import taaf.game_api as ga
from arcduck2.config import Config
from arcduck2.llm import LLMReply, ToolCall
from arcduck2.solver import DuckV2Solver

IDS = sys.argv[1].split(",") if len(sys.argv) > 1 and "," in sys.argv[1] else ["ls20-9607627b", "ft09-0d8bbf25", "vc33-5430563c"]
SECONDS = float(sys.argv[2]) if len(sys.argv) > 2 else 90

EXPLORE = '''
names = [a for a in available_actions if a not in ("RESET", "UNDO", "CLICK")]
print("avail", available_actions, "lvl", levels_completed, "/", number_of_levels, "sizes", sorted({int(s["size"]) for s in segments(grid)})[:8])
for n in names[:3]:
    r = action(n)
    print(n, r.line()[:160])
    if not r.ok or not r.board_changed:
        continue
if "CLICK" in available_actions:
    segs = segments(grid)
    if segs:
        s = segs[-1]
        r = action("CLICK", int(s["cx"]), int(s["cy"]))
        print("click", r.line()[:160])
notes = (globals().get("notes") or []) + ["explored"]
'''
THINK = "print('thinking', levels_completed, level_actions)"


class FakeModel:
    def __init__(self): self.n = {}
    def chat(self, messages, max_tokens=None):
        key = id(messages[1]) if False else messages[1]["content"][0]["text"][:60] if isinstance(messages[1]["content"], list) else messages[1]["content"][:60]
        self.n[key] = self.n.get(key, 0) + 1
        code = EXPLORE if self.n[key] <= 4 else THINK
        calls = [ToolCall("c", "python", code)]
        raw = [{"id": "c", "type": "function", "function": {"name": "python", "arguments": "{\"code\": \"x\"}"}}]
        time.sleep(0.2)
        return LLMReply(content="", reasoning="explore", tool_calls=calls, raw_tool_calls=raw, finish_reason="tool_calls",
                        prompt_tokens=2000, completion_tokens=80)


async def main():
    logging.getLogger("arc_agi").setLevel(logging.WARNING)
    spec = ga.ArcadeSpec(operation_mode=arc_agi.OperationMode.ONLINE, environments_dir="")
    session = taaf.game.RunSession()
    games = [ga.GameAPI(env_name=i, arcade_spec=spec) for i in IDS]
    for g in games:
        g.start_game(session)
        print("started", g.game_id, "levels", g.number_of_levels, "baseline", g.base_actions_per_level)
    cfg = replace(Config(), slots=2, tool_timeout_s=20.0, debug_dir=os.path.abspath("smoke_trace"))
    solver = DuckV2Solver(config=cfg, hard_deadline_epoch=time.time() + SECONDS)
    solver.setup(); solver._llm = FakeModel()
    await solver.run_games(games)
    solver.teardown()
    for g in games:
        r = g.game_run
        print("RESULT", g.game_id, r.state, "levels", r.levels_completed, "actions", sum(r.actions_per_level), "score", r.final_score, "|", r.solver_note)
    session.close()

if __name__ == "__main__":
    asyncio.run(main())
