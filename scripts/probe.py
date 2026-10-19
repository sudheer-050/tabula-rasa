import sys, logging
import arc_agi, taaf.game, taaf.game_api as ga
from arcduck2.session import GameSession
from arcduck2 import perception as P
logging.getLogger("arc_agi").setLevel(logging.WARNING)
gid = sys.argv[1]
spec = ga.ArcadeSpec(operation_mode=arc_agi.OperationMode.ONLINE, environments_dir="")
sess = taaf.game.RunSession()
g = ga.GameAPI(env_name=gid, arcade_spec=spec); g.start_game(sess)
s = GameSession(g)
print("avail", s.available_names(), "levels", s.number_of_levels, "grid", s.grid.shape, s.grid.dtype, "colors", sorted(set(s.grid.ravel().tolist())))
segs = P.segments(s.grid)
print("segments", len(segs), segs[:6])
for n in [x for x in s.available_names() if x not in ("RESET","CLICK","UNDO")][:4]:
    r = s.execute(n); print(n, r.line()[:200])
if "CLICK" in s.available_names():
    for sg in segs[:4]:
        r = s.execute("CLICK", int(sg["cx"]), int(sg["cy"])); print("click", (sg["cx"], sg["cy"]), r.line()[:200])
print("actions", s.total_actions, "avail now", s.available_names())
if "UNDO" in s.available_names():
    r = s.execute("UNDO"); print("undo", r.line()[:160])
g.finish_game(); sess.close()
