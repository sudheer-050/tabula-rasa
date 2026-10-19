"""Reproduce the archived study: Python 3.10+ and NumPy; no model or network.

Run from any working directory: python reproduce.py
Writes paper_results.json and paper_verification.txt beside this script.
"""
from __future__ import annotations

import collections
import contextlib
import hashlib
import io
import json
from pathlib import Path
import runpy
import sys
import tempfile
import types
import unittest

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import analyze
from parse_log import parse, score_from_levels


def audit_cascade():
    """Execute the unmodified archived patch against an explicit fake interface."""
    state = types.SimpleNamespace(level=1, history=[types.SimpleNamespace(action="RESET")])

    class Environment:
        outcome = {}

        def step(self, arguments):
            result = {"executed": True, "executed_actions": [arguments["action"]], **self.outcome}
            if result.get("level_completed"):
                state.level += 1
            if result.get("game_over"):
                state.history.append(types.SimpleNamespace(action="RESET"))
            return result

    class Agent:
        model_calls = 0
        next_action = "UP"
        wrapped_owner = None

        def analyze(self, path, *, step_env):
            self.model_calls += 1
            self.wrapped_owner = getattr(step_env, "__self__", None)
            step_env({"action": self.next_action})
            return types.SimpleNamespace(step_executed=False)

    modules = {name: types.ModuleType(name) for name in
               ["inference", "inference.agent", "inference.agent.tool_agent", "inference.agent.action_names"]}
    modules["inference"].__path__ = []
    modules["inference.agent"].__path__ = []
    modules["inference"].agent = modules["inference.agent"]
    tam = modules["inference.agent.tool_agent"]
    tam.ToolAgent = Agent
    tam.AnalyzerTurnResult = lambda **kw: types.SimpleNamespace(**kw)
    tam.load_runtime_state = lambda path: (types.SimpleNamespace(level=state.level), list(state.history))
    actions = modules["inference.agent.action_names"]
    actions.ENGINE_TO_MODEL_ACTION = {"ACTION1": "UP", "ACTION7": "UNDO"}
    actions.undo_exposure_mode = lambda: "on"
    modules["inference.agent"].tool_agent = tam
    modules["inference.agent"].action_names = actions
    previous = {name: sys.modules.get(name) for name in modules}
    sys.modules.update(modules)
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            runpy.run_path(str(ROOT / "archived_cascade_v2.py"))
        agent, env = Agent(), Environment()
        results = []
        # Failure/restart, then solve level 1 with UNDO, fail/restart level 2, solve it.
        with tempfile.TemporaryDirectory(prefix="arc-paper-") as directory:
            path = Path(directory) / "state.marker"
            path.touch()
            for action, outcome in [("UP", {"game_over": True}), ("UNDO", {}),
                                    ("UP", {"level_completed": True}),
                                    ("UP", {"game_over": True}),
                                    ("UP", {"level_completed": True})]:
                agent.next_action, env.outcome = action, outcome
                results.append(agent.analyze(path, step_env=env.step))
        return {"model_calls": agent.model_calls,
                "replay_returns": sum(bool(r.step_executed) for r in results),
                "cached_levels": sorted(agent._cascade_state["best"]),
                "level_one_actions": agent._cascade_state["best"][1],
                "original_callback_has_owner": env.step.__self__ is env,
                "wrapped_callback_has_owner": agent.wrapped_owner is not None}
    finally:
        for name, module in previous.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


class EvidenceChecks(unittest.TestCase):
    def test_successful_replay_is_unused_in_level_only_lifecycle(self):
        result = audit_cascade()
        self.assertEqual(result["model_calls"], 5)
        self.assertEqual(result["replay_returns"], 0)
        self.assertEqual(result["cached_levels"], [1, 2])

    def test_actual_archived_wrapper_loses_bound_owner(self):
        result = audit_cascade()
        self.assertTrue(result["original_callback_has_owner"])
        self.assertFalse(result["wrapped_callback_has_owner"])

    def test_success_cache_retains_undo_and_excludes_failed_attempt(self):
        self.assertEqual(audit_cascade()["level_one_actions"], [{"action": "UNDO"}, {"action": "UP"}])

    def test_score_cap_cannot_credit_unsolved_level(self):
        # Level 1 is 115%, level 2 unsolved. Environment capped at 1/(1+2).
        self.assertAlmostEqual(score_from_levels([1, 0], [10, 10], 1), 100 / 3)


def main():
    manifest = json.loads((ROOT / "paper_evidence.json").read_text(encoding="utf8"))
    reports, groups = [], []
    for expected in manifest["runs"]:
        path = ROOT / expected["artifact"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected["sha256"], path.name
        runs, _, summary, _ = parse(str(path))
        assert len(runs) == 100
        by_game = collections.defaultdict(list)
        for run in runs:
            by_game[run.game].append(run)
        assert len(by_game) == 25 and all(len(rs) == 4 for rs in by_game.values())
        errors = [abs(r.score - score_from_levels(r.level_actions, r.level_baseline, r.levels_done)) for r in runs]
        assert max(errors) < .005
        mean = sum(r.score for r in runs) / len(runs)
        ceiling = sum(100 * sum(range(1, r.levels_done + 1)) / sum(range(1, r.n_levels + 1)) for r in runs) / len(runs)
        actions, tokens = sum(r.actions for r in runs), sum(r.tokens for r in runs)
        for key, value in [("mean_score", mean), ("actions", actions), ("generated_tokens", tokens)]:
            assert abs(value - expected[key]) < 1e-9, (path.name, key)
        reports.append({"artifact": path.name, "runs": len(runs), "games": len(by_game),
                        "mean_score": mean, "completion_ceiling": ceiling,
                        "efficiency_headroom": ceiling - mean, "actions": actions,
                        "generated_tokens": tokens, "tokens_per_action": tokens / actions,
                        "completed_levels": sum(r.levels_done for r in runs),
                        "outcomes": dict(collections.Counter(r.state for r in runs)),
                        "maximum_score_error": max(errors), "duration": summary.get("duration")})
        groups.append(by_game)
    # Reuse the archived analysis and its documented seeded NumPy RNG.
    comparison = analyze.paired(*groups)
    buffer = io.StringIO()
    tests = unittest.TextTestRunner(stream=buffer, verbosity=2).run(
        unittest.defaultTestLoader.loadTestsFromTestCase(EvidenceChecks))
    assert tests.wasSuccessful()
    result = {"runs": reports, "paired_comparison": comparison,
              "actual_cascade_simulation": audit_cascade(), "tests_passed": tests.testsRun,
              "limitations": "Historical comparison; not an isolated randomized ablation. Fake runtime tests; not a GPU benchmark."}
    (ROOT / "paper_results.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf8")
    text = buffer.getvalue() + "\nAll 200 scores reconstructed within 0.005; raw-log hashes match.\n"
    text += f"Game-paired mean delta: {comparison['mean_delta']:.4f}\n"
    text += f"Game bootstrap 95% interval: {comparison['ci95_boot_games']}\n"
    text += "No new GPU run, model training, or competition submission performed.\n"
    (ROOT / "paper_verification.txt").write_text(text, encoding="utf8")
    print(text)


if __name__ == "__main__":
    main()
