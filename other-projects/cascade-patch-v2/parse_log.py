"""Parse Duck-harness / Franzen-stack Kaggle run logs into per-run and per-level tables.

Usage:
    python -I parse_log.py RUN.log [--label NAME] [--out-prefix PATH] [--check]

Reads the `[finished] <game> state=... level=a/b score=... actions=... tokens=...
per-level=act/base,act/base,...` lines that the benchmark prints once per (game, pass),
plus the final `benchmark:` summary block and the periodic `mean score:` snapshots.

Writes (when --out-prefix is given):
    <prefix>_runs.csv     one row per (game, pass)
    <prefix>_levels.csv   one row per (game, pass, level)
    <prefix>_curve.csv    elapsed_s, mean_score snapshots (progress over the 8h budget)

`--check` recomputes every run's score from its per-level actions with the official
arc_agi 0.9.9 scorecard formula (see score_from_levels) and reports mismatches.

Pass index: the log does not print which pass a run belongs to, so passes are numbered
in finish order per game (0..n-1). Passes are exchangeable replicates, so this only
matters if you want to pair "pass k" across two logs, which you should NOT do: pair
by game, not by pass.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field, asdict

FINISHED_RE = re.compile(
    r"\[finished\]\s+(?P<game>\S+)\s+state=(?P<state>\S+)\s+level=(?P<done>\d+)/(?P<nlev>\d+)"
    r"\s+score=(?P<score>[\d.]+)\s+actions=(?P<actions>\d+)\s+tokens=(?P<tokens>\d+)"
    r"\s+per-level=(?P<per>[\d/,]+)"
)
TIME_RE = re.compile(r"^\s*(?P<t>[\d.]+)s\s+\d+\s")
MEAN_RE = re.compile(r"^\s*(?P<t>[\d.]+)s\s+\d+\s+mean score:\s+(?P<m>[\d.]+)")
SUMMARY_KEYS = ("games", "passes", "runs", "started", "ended", "duration", "mean score",
                "median score", "total actions", "total tokens")
CASCADE_RE = re.compile(r"taaf\.custom: (cascade [^\n]*)")


@dataclass
class Run:
    game: str
    pass_idx: int
    state: str
    levels_done: int
    n_levels: int
    score: float          # as printed by the harness
    actions: int
    tokens: int
    finished_at_s: float
    level_actions: list = field(default_factory=list)
    level_baseline: list = field(default_factory=list)

    @property
    def game_family(self) -> str:
        return self.game.split("-")[0]


def level_score(actions: int, baseline: int, completed: bool) -> float:
    """Official arc_agi 0.9.9 EnvironmentScoreCalculator.add_level: 100*(b/a)^2, capped at 115."""
    if not completed or actions <= 0:
        return 0.0
    return min(((baseline / actions) ** 2) * 100.0, 115.0)


def score_from_levels(level_actions, level_baseline, levels_done) -> float:
    """Official to_score(): level-index-weighted mean (level k has weight k, 1-based),
    then capped at 100 * (weight of levels with score>0) / (total weight)."""
    tot = wsum = wpos = 0.0
    for i, (a, b) in enumerate(zip(level_actions, level_baseline)):
        w = i + 1
        s = level_score(a, b, i < levels_done)
        tot += s * w
        wsum += w
        if s > 0:
            wpos += w
    if wsum == 0:
        return 0.0
    return min(tot / wsum, wpos / wsum * 100.0)


def parse(path: str):
    runs: list[Run] = []
    curve: list[tuple[float, float]] = []
    summary: dict[str, str] = {}
    cascade = None
    counter: dict[str, int] = defaultdict(int)
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            tm = TIME_RE.match(line)
            t = float(tm.group("t")) if tm else float("nan")
            mm = MEAN_RE.match(line)
            if mm:
                curve.append((float(mm.group("t")), float(mm.group("m"))))
            cm = CASCADE_RE.search(line)
            if cm:
                cascade = cm.group(1).strip()
            for key in SUMMARY_KEYS:
                m = re.match(r"^\s*[\d.]+s\s+\d+\s+" + re.escape(key) + r":\s+(.*)$", line)
                if m:
                    summary[key] = m.group(1).strip()  # last one wins = final block
            # several [finished] records can share one physical line (interleaved stdout)
            for m in FINISHED_RE.finditer(line):
                pairs = [p.split("/") for p in m.group("per").strip(",").split(",") if p]
                la = [int(a) for a, _ in pairs]
                lb = [int(b) for _, b in pairs]
                g = m.group("game")
                runs.append(Run(
                    game=g, pass_idx=counter[g], state=m.group("state"),
                    levels_done=int(m.group("done")), n_levels=int(m.group("nlev")),
                    score=float(m.group("score")), actions=int(m.group("actions")),
                    tokens=int(m.group("tokens")), finished_at_s=t,
                    level_actions=la, level_baseline=lb))
                counter[g] += 1
    return runs, curve, summary, cascade


def check(runs) -> list[str]:
    bad = []
    for r in runs:
        s = score_from_levels(r.level_actions, r.level_baseline, r.levels_done)
        if abs(s - r.score) > 0.011:
            bad.append(f"{r.game} pass{r.pass_idx}: printed {r.score:.2f} recomputed {s:.2f}")
    return bad


def per_game_means(runs):
    g = defaultdict(list)
    for r in runs:
        g[r.game].append(r.score)
    return {k: sum(v) / len(v) for k, v in sorted(g.items())}


def write_tables(prefix: str, label: str, runs, curve):
    with open(prefix + "_runs.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["label", "game", "pass", "state", "levels_done", "n_levels", "score",
                    "actions", "tokens", "finished_at_s"])
        for r in runs:
            w.writerow([label, r.game, r.pass_idx, r.state, r.levels_done, r.n_levels,
                        r.score, r.actions, r.tokens, r.finished_at_s])
    with open(prefix + "_levels.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["label", "game", "pass", "level", "weight", "completed", "actions",
                    "baseline", "level_score", "efficiency_ratio"])
        for r in runs:
            for i, (a, b) in enumerate(zip(r.level_actions, r.level_baseline)):
                done = i < r.levels_done
                w.writerow([label, r.game, r.pass_idx, i + 1, i + 1, int(done), a, b,
                            round(level_score(a, b, done), 3),
                            round(b / a, 4) if (done and a) else ""])
    with open(prefix + "_curve.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["elapsed_s", "mean_score"])
        w.writerows(curve)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("log")
    ap.add_argument("--label", default=None)
    ap.add_argument("--out-prefix", default=None)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--json", action="store_true", help="print summary as JSON")
    a = ap.parse_args(argv)
    runs, curve, summary, cascade = parse(a.log)
    label = a.label or a.log
    if not runs:
        print("no [finished] records found", file=sys.stderr)
        return 1
    pg = per_game_means(runs)
    mean_of_game_means = sum(pg.values()) / len(pg)
    info = {
        "label": label, "cascade": cascade, "n_runs": len(runs), "n_games": len(pg),
        "passes_per_game": sorted({sum(1 for r in runs if r.game == g) for g in pg}),
        "printed_final_mean": summary.get("mean score"),
        "recomputed_mean_of_game_means": round(mean_of_game_means, 2),
        "won_runs": sum(r.state == "won" for r in runs),
        "total_actions": sum(r.actions for r in runs),
        "duration": summary.get("duration"),
    }
    if a.check:
        bad = check(runs)
        info["score_check"] = "all %d runs match official formula" % len(runs) if not bad else bad
    if a.out_prefix:
        write_tables(a.out_prefix, label, runs, curve)
        info["written"] = [a.out_prefix + s for s in ("_runs.csv", "_levels.csv", "_curve.csv")]
    if a.json:
        print(json.dumps(info, indent=2))
    else:
        for k, v in info.items():
            print(f"{k:32s} {v}")
        print("\nper-game mean score:")
        for g, s in pg.items():
            print(f"  {g:16s} {s:7.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
