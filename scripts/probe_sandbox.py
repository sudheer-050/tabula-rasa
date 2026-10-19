import sys, logging
from dataclasses import replace
import arc_agi, taaf.game, taaf.game_api as ga
from arcduck2.config import Config
from arcduck2.session import GameSession
from arcduck2.sandbox import Sandbox
sys.path.insert(0, "scripts")
logging.getLogger("arc_agi").setLevel(logging.WARNING)
src = open("scripts/smoke_online.py", encoding="utf-8").read()
code = src.split("EXPLORE = '''")[1].split("'''")[0]
spec = ga.ArcadeSpec(operation_mode=arc_agi.OperationMode.ONLINE, environments_dir="")
sess = taaf.game.RunSession()
g = ga.GameAPI(env_name=sys.argv[1], arcade_spec=spec); g.start_game(sess)
sb = Sandbox(GameSession(g), replace(Config(), tool_timeout_s=20.0))
out = sb.run(code, 80)
print(out.text); print("executed", out.executed_actions, "error", out.error)
g.finish_game(); sess.close()
