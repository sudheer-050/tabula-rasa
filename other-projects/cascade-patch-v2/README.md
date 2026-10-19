# Cascade patch v2

An earlier, much smaller intervention on the same competition (ARC Prize 2026 — ARC-AGI-3), before the `arcduck2` rewrite in the main repo. Instead of a new agent, this is a single monkey-patch applied on top of the community "Franzen stack" harness's own agent loop (`inference.agent.tool_agent.ToolAgent.analyze`).

**What it does:** after a level `RESET`, replay the best known-successful action sequence for that level — one action per turn, zero LLM calls — until the replay runs out or an action fails, then hand control back to the normal LLM-driven turn. The idea: stop re-paying model cost for moves already proven to work, and spend the saved budget pushing further into the level.

**Fixed three bugs** found in an earlier version of the same idea (used for prior submissions):
1. The replay index only re-armed on a level change, not on every `RESET` — so the cheap replay fired once per level *ever*, not once per attempt.
2. The cached "best" sequence could be overwritten by a longer *failing* attempt, not just a successful one.
3. The action parser silently dropped `UNDO` from the vocabulary it tracked, corrupting any cached sequence that used it.

**Result:** scored **29.97** on the public leaderboard — at the time, this was the best score achieved on this competition, and the thing `arcduck2`'s first real submission (8.01) initially regressed against. See the [main repo's README](../../README.md) for how that regression got diagnosed and fixed.

Everything else in the notebook this patch lived in (the harness itself, model serving, the base agent loop) is the community TAAF/Franzen stack, not reproduced here — only the patch, which was the original contribution.
