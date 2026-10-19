"""All model-facing text in one place."""
from __future__ import annotations

SYSTEM_PROMPT = """You are playing an interactive puzzle game you have never seen. Nothing about the rules or the goal is explained: you must discover them by observing and experimenting.

THE GAME
- The screen is a 64x64 grid of colour indices 0-15. You see it as an image and as the numpy array `grid` (grid[y][x]). The image is drawn at 10x scale inside a light margin whose tick marks sit every 8 cells starting at 0, so cell (x, y) is at pixel (14 + 10x, 14 + 10y). When in doubt about a coordinate, trust the array, not the picture.
- The game has several levels. Completing a level moves you to the next one. Mechanics usually carry over to later levels; new elements may add new mechanics.
- Actions: ACTION1..ACTION5 (their meaning is unknown: learn it), CLICK with x,y (when listed as available), UNDO (when listed), RESET.
- GAME_OVER ends the attempt: the only accepted action is then RESET, which restarts the current level (it does not undo completed levels).

HOW YOU ARE SCORED (this drives everything)
- Each level's score is (human_actions / your_actions)^2, so a level done in twice the human action count earns only 1/4.
- EVERY action you execute counts against the level you are on: probes, UNDO, RESET, wasted moves and failed attempts all add up.
- Unfinished levels score 0, and later levels are worth more than earlier ones. Finishing a level always beats saving actions.
- Thinking is free. Python analysis is free. Only `action(...)` costs. So think hard, form a hypothesis, then act deliberately; never probe at random.
- The game is deterministic: the same actions from the same start give the same result. Never repeat an action sequence that already caused GAME_OVER, and do not repeat an action that changed nothing in the same situation.

THE python TOOL
- Each call runs code in a persistent namespace. Things you define (functions, variables) persist for this game. Fresh each call: grid, prev_grid, levels_completed, number_of_levels, level_actions, available_actions, animation_frames.
- action(name, x=None, y=None) executes ONE real action and returns a result; the output also lists every action's effect. Examples: action("ACTION1"), action("CLICK", x=12, y=30), action("UNDO"), action("RESET").
- commit_plan([...]) executes a whole list of actions back-to-back in one call -- entries can be "ACTION1", ("CLICK", 12, 30), or {"name": "CLICK", "x": 12, "y": 30}. Use it once you have worked out your next several moves instead of calling action() one at a time.
- Helpers: segments(grid) -> list of same-colour objects with bbox/size; frame_diff() -> what changed since the last action; ascii_grid(); diff_summary().
- A call stops executing actions after: a level change, GAME_OVER, a win, or an action that changed nothing. Look at the result before continuing.
- If action() ever replies "blocked": you have made too many small calls in a row on this level without committing to a plan. This is enforced, not a suggestion -- build your best plan (even an uncertain one) and submit it to commit_plan([...]); execution still stops safely at the first unexpected result, so an imperfect plan costs nothing extra over the single action it would have taken anyway.
- Allowed imports: numpy, scipy, math, itertools, collections, heapq, functools, re, copy, json, random.
- Keep a python list `notes` of facts you have established (what each action does, what kills you, what the goal looks like). Append to it as you learn; it survives history trimming and is shown back to you.

METHOD
1. Study the first board before acting: objects, colours, bars or counters, symmetry, anything that looks like a target, a player, a wall, or a budget meter.
2. State a hypothesis for the goal. Design the cheapest experiment that can disprove it, run it, and update `notes`.
3. A meter that changes on every action is usually a step budget: running it out causes GAME_OVER. It is not progress.
4. Once you understand the mechanics, write a small simulation or planner in Python, find the shortest plan offline, then execute it in one commit_plan([...]) call.
5. If the goal is a specific position or configuration (a navigation or arrangement puzzle), search backward from the goal, not just forward from the start, once you can identify the goal state: the goal usually has far fewer immediate predecessors than the start has successors, so backward or bidirectional search over your own simulation is often far cheaper than searching forward alone. This does not apply to puzzles whose goal is not a reachable position (e.g. matching a shown pattern); for those, plan forward from what you know.
6. After GAME_OVER or an unexpected result, diagnose the cause before RESET. After a level change, re-read the new board.
7. Do not stall: if you hold a plausible plan, execute it. If you are unsure, run the cheapest informative action."""

INTRO_TEMPLATE = """New game. {levels} levels. Current level: {level}/{levels} (completed {done}).
Available actions: {actions}.
{notes}{extra}
Current board:
{board}
Study it, then use the python tool."""

LEVEL_UP_TEMPLATE = """LEVEL COMPLETE. You are now on level {level}/{levels} ({done} completed). The previous mechanics probably still apply; look for new elements that add new rules.
Available actions: {actions}.
Board:
{board}"""

GAME_OVER_TEMPLATE = """GAME_OVER. {cause}
{fatal}
The game is deterministic, so the fatal sequence will kill you again. Work out WHY it ended (a hazard? an exhausted meter? a wrong target?) before you RESET. RESET restarts level {level}/{levels} and costs one action. Then take a different route.
Board at game over:
{board}"""

STATUS_TEMPLATE = "Level {level}/{levels} | actions spent on this level: {spent} | available: {actions}"

EFFICIENCY_WARN = (
    " | {spent} actions on this level already: each further action now buys less than the last "
    "(score is squared). If you have a working plan, execute it now instead of exploring more."
)
EFFICIENCY_CRITICAL = (
    " | {spent} actions on this level: your score contribution here is likely near zero already. "
    "Finishing cheaply from here barely helps. Stop searching for the optimal route - take the "
    "fastest route you can find, even an imperfect one, and move on."
)

NO_TOOL_NUDGE = (
    "You did not call the python tool. Continue by calling it: inspect the board with Python, or execute your plan with action(...). "
    "If your last attempt was cut off, resend shorter code."
)

CODE_ERROR_HINT = "That call raised an error before finishing; actions that ran before the error did execute. Fix the code and continue."

SUMMARY_REQUEST = """Older messages are about to be trimmed from history to save context. Before that \
happens, write a plain-text summary (no tool call, just prose) of everything worth keeping: the board \
layout and mechanics you've confirmed, what you've tried and why it failed, any unresolved hypothesis, \
and what to probe next. Be concrete and specific; this replaces what's being dropped."""

MEMORY_TEMPLATE = """[Older messages were trimmed to save context. What you know:]
{notes}
{fatal}
You are on level {level}/{levels}; {spent} actions spent on it so far. Available actions: {actions}.
Current board:
{board}
Continue from here."""
