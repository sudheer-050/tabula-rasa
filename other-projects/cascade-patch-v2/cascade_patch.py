# --- Cascade patch v2 (own addition): prefix-replay cheap tier before LLM escalation ---
#
# What this does: after a level RESET, replay the best known-successful action sequence
# recorded for that level, one action per turn, with no LLM call - until either the
# replay runs out or an action fails to execute, then hand control back to the normal
# LLM-driven analyze(). The intent is to stop re-paying LLM cost for moves we already
# know work, and spend the saved budget pushing further into each level.
#
# Three bugs fixed vs. the original patch:
#   1. The cheap-replay path now re-arms on every RESET (every attempt), not just the
#      first RESET ever seen for a level. Previously `replay_idx` only reset on a level
#      change, so after the first successful replay it stayed pinned at the end and never
#      replayed again for that level, even on later retries.
#   2. `best` is now only updated from an attempt that actually finished the level
#      (`level_completed` / `run_complete`). The original treated `game_over` (a failed
#      attempt) the same as a win when deciding whether to cache it as the sequence to
#      replay next time -- a long failing run could overwrite a short winning one.
#   3. The recognized action vocabulary is now read from the harness's own
#      `action_names` module instead of a hardcoded {"UP","DOWN","LEFT","RIGHT","SPACE"}
#      set. This run's own config enables `EXPOSE_UNDO=on` (see cell 4); the hardcoded
#      list silently dropped every UNDO action from the tracked sequence, corrupting
#      whatever got cached as "best" for any level where UNDO was used.


def _install_cascade():
    try:
        import inference.agent.tool_agent as tam
        import inference.agent.action_names as action_names
    except Exception as exc:
        print("taaf.custom: cascade skipped (import): " + repr(exc))
        return
    if not (
        hasattr(tam, "ToolAgent")
        and hasattr(tam, "AnalyzerTurnResult")
        and hasattr(tam, "load_runtime_state")
    ):
        print("taaf.custom: cascade skipped (tool_agent module shape unexpected)")
        return
    ToolAgent = tam.ToolAgent
    if not hasattr(ToolAgent, "analyze"):
        print("taaf.custom: cascade skipped (no analyze method)")
        return
    original_analyze = ToolAgent.analyze

    import re as cre

    mouse_re = cre.compile(r"MOUSE\(row=(\d+),\s*col=(\d+)\)")

    def _known_simple_actions():
        # Base vocabulary from the harness's own engine<->model action map, plus UNDO
        # only when this run actually exposes it -- stays correct if that flag changes.
        names = {"UP", "DOWN", "LEFT", "RIGHT", "SPACE"}
        try:
            names.update(str(v).upper() for v in action_names.ENGINE_TO_MODEL_ACTION.values())
        except Exception:
            pass
        try:
            if action_names.undo_exposure_mode() == "on":
                names.add("UNDO")
        except Exception:
            pass
        return names

    def display_to_action(display):
        s = str(display).strip()
        m = mouse_re.match(s)
        if m:
            return {"action": "ACTION6", "row": int(m.group(1)), "col": int(m.group(2))}
        if s.upper() in _known_simple_actions():
            return {"action": s.upper()}
        return None

    def cascaded_analyze(self, *args, **kwargs):
        try:
            if len(args) >= 1 and "step_env" in kwargs:
                state_path = args[0]
                step_env = kwargs["step_env"]
                if state_path is not None and step_env is not None and state_path.exists():
                    cascade = getattr(self, "_cascade_state", None)
                    if cascade is None:
                        cascade = {
                            "level": None,
                            "attempt": [],
                            "best": {},
                            "replay_idx": 0,
                            "replaying": False,
                            "seen_resets": 0,
                        }
                        self._cascade_state = cascade

                    current_frame, history_entries = tam.load_runtime_state(state_path)
                    level = getattr(current_frame, "level", None)
                    if level != cascade.get("level"):
                        cascade.update(
                            level=level, attempt=[], replay_idx=0, replaying=False, seen_resets=0
                        )

                    # Count RESETs rather than just checking the last action, so a fresh
                    # attempt is detected exactly once even if analyze() is called more
                    # than once while the most recent action is still RESET.
                    reset_count = sum(
                        1 for h in history_entries if getattr(h, "action", None) == "RESET"
                    )
                    if reset_count != cascade.get("seen_resets"):
                        cascade["seen_resets"] = reset_count
                        cascade["attempt"] = []
                        cascade["replay_idx"] = 0
                        cascade["replaying"] = bool(cascade["best"].get(level))

                    best = cascade["best"].get(level) or []
                    if cascade.get("replaying") and cascade["replay_idx"] < len(best):
                        action_dict = best[cascade["replay_idx"]]
                        result = step_env(dict(action_dict))
                        if isinstance(result, dict) and result.get("executed"):
                            cascade["replay_idx"] += 1
                            cascade["attempt"] = list(best[: cascade["replay_idx"]])
                            if cascade["replay_idx"] >= len(best):
                                cascade["replaying"] = False
                            return tam.AnalyzerTurnResult(step_executed=True)
                        cascade["replaying"] = False
                    elif cascade.get("replaying"):
                        cascade["replaying"] = False

                    def tracking_step_env(arguments, _orig=step_env, _lvl=level, _cascade=cascade):
                        result = _orig(arguments)
                        try:
                            if isinstance(result, dict):
                                for disp in result.get("executed_actions") or []:
                                    parsed = display_to_action(disp)
                                    if parsed is not None:
                                        _cascade["attempt"].append(parsed)
                                succeeded = bool(
                                    result.get("level_completed") or result.get("run_complete")
                                )
                                if succeeded and len(_cascade["attempt"]) > len(
                                    _cascade["best"].get(_lvl) or []
                                ):
                                    _cascade["best"][_lvl] = list(_cascade["attempt"])
                        except Exception:
                            pass
                        return result

                    kwargs["step_env"] = tracking_step_env
        except Exception as exc:
            print("taaf.custom: cascade pre-check failed, using plain LLM turn: " + repr(exc))
        return original_analyze(self, *args, **kwargs)

    ToolAgent.analyze = cascaded_analyze
    print(
        "taaf.custom: cascade v2 installed "
        "(prefix-replay per attempt, success-only best, dynamic action vocabulary)"
    )


try:
    _install_cascade()
except Exception as _cascade_top_exc:
    print("taaf.custom: cascade install crashed: " + repr(_cascade_top_exc))
