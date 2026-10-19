"""Noise-floor, paired-comparison and power analysis for ARC-AGI-3 offline runs.

Inputs are raw Kaggle run logs (parsed with parse_log.py), so no intermediate files are needed.

    python -I analyze.py noise   RUN.log                     # noise floor of one config
    python -I analyze.py compare A.log B.log [--label-a ..]  # paired A vs B (B - A)
    python -I analyze.py power   RUN.log [--delta 3]         # passes/games needed
    python -I analyze.py lb 29.97 26.83 [--pair 27.89 33.22] # public-LB replicate noise

Statistical model (per config): y[g,p] = mu + game_g + eps[g,p]
  y[g,p]  official per-game score of game g on pass p (0-100)
  game_g  game difficulty (fixed for a given game set)
  eps     run-to-run noise (sampling temperature, GPU contention, time budget)

Two different questions need two different error bars:
  * "fixed game set" SE   - would the same 25 games give the same number again?  (noise only)
  * "game-sampling" SE    - what does this say about a *different* game set (hidden LB)? (noise + game heterogeneity)
For an A/B decision on the hidden set, the game-sampling (paired, per-game) interval is the
honest one; the fixed-set interval is the floor you can never get below on these 25 games.

Only numpy is required. Seeded RNG, so output is reproducible.
"""
from __future__ import annotations

import argparse
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from parse_log import parse, level_score  # noqa: E402

RNG = np.random.default_rng(20261012)
Z = {0.90: 1.6449, 0.95: 1.9600, 0.80: 0.8416}


# --------------------------------------------------------------------------- loading
def load(path):
    runs, curve, summary, cascade = parse(path)
    if not runs:
        raise SystemExit(f"{path}: no [finished] records")
    by_game = defaultdict(list)
    for r in runs:
        by_game[r.game].append(r)
    return by_game, curve, summary, cascade


def score_matrix(by_game, games=None):
    games = games or sorted(by_game)
    return games, [np.array([r.score for r in by_game[g]], float) for g in games]


# --------------------------------------------------------------------------- noise
def noise_stats(by_game, n_sim=200_000):
    games, ys = score_matrix(by_game)
    G = len(games)
    means = np.array([y.mean() for y in ys])
    wvar = np.array([y.var(ddof=1) if len(y) > 1 else np.nan for y in ys])
    P = int(np.median([len(y) for y in ys]))
    mean_w = float(np.nanmean(wvar))                 # avg within-game variance of ONE run
    sd_single_fixed = math.sqrt(mean_w / G)          # analytic: one pass over these G games
    # Monte Carlo: pick one pass per game at random -> distribution of a single-pass score
    idx = [RNG.integers(0, len(y), n_sim) for y in ys]
    sims = np.mean([y[i] for y, i in zip(ys, idx)], axis=0)
    # game-sampling bootstrap of the 4-pass mean (generalisation to a new game set)
    bidx = RNG.integers(0, G, (20_000, G))
    boot = means[bidx].mean(axis=1)
    between_var = max(float(means.var(ddof=1)) - mean_w / P, 0.0)
    return dict(
        games=G, passes=P, mean=float(means.mean()),
        within_sd_per_game_run=math.sqrt(mean_w),
        between_game_sd=math.sqrt(between_var),
        icc=between_var / (between_var + mean_w),
        sd_single_pass_fixed_set_analytic=sd_single_fixed,
        sd_single_pass_fixed_set_mc=float(sims.std()),
        single_pass_range_95=(float(np.percentile(sims, 2.5)), float(np.percentile(sims, 97.5))),
        single_pass_min_max_observed_mc=(float(sims.min()), float(sims.max())),
        se_Ppass_mean_fixed_set=math.sqrt(mean_w / P / G),
        ci95_game_sampling=(float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))),
        per_game=[(g, float(m), float(math.sqrt(v)) if not np.isnan(v) else float("nan"),
                   [float(x) for x in y]) for g, m, v, y in zip(games, means, wvar, ys)],
        mean_w=mean_w,
    )


def sd_for_n_games(mean_w, n):
    return math.sqrt(mean_w / n)


def cmd_noise(a):
    by_game, curve, summary, cascade = load(a.log)
    s = noise_stats(by_game)
    print(f"config: {cascade}   final printed mean: {summary.get('mean score')}")
    print(f"games={s['games']} passes={s['passes']}  mean of game means = {s['mean']:.2f}")
    print(f"within-game run-to-run SD (one game, one run) ........ {s['within_sd_per_game_run']:.2f} pts")
    print(f"between-game SD (true difficulty spread) ............. {s['between_game_sd']:.2f} pts  ICC={s['icc']:.2f}")
    print(f"SD of a single-pass {s['games']}-game score (fixed set) ...... "
          f"{s['sd_single_pass_fixed_set_analytic']:.2f} analytic / {s['sd_single_pass_fixed_set_mc']:.2f} MC")
    lo, hi = s['single_pass_range_95']
    print(f"  95% range of a single-pass score .................. [{lo:.2f}, {hi:.2f}]  (width {hi-lo:.2f})")
    print(f"SE of the {s['passes']}-pass mean, fixed game set ............. {s['se_Ppass_mean_fixed_set']:.2f}")
    lo, hi = s['ci95_game_sampling']
    print(f"95% CI for a NEW random game set (game bootstrap) .... [{lo:.2f}, {hi:.2f}]")
    print("extrapolated single-run SD from within-game noise only (assumes no job-level shock):")
    for n in (25, 55, 110):
        print(f"  {n:4d} games: SD {sd_for_n_games(s['mean_w'], n):.2f}  -> 95% half-width {1.96*sd_for_n_games(s['mean_w'], n):.2f}")
    if a.per_game:
        print("\nper-game: mean  sd  passes")
        for g, m, sd, ys in sorted(s['per_game'], key=lambda t: -t[2]):
            print(f"  {g:16s} {m:7.2f} {sd:6.2f}  {', '.join(f'{x:.1f}' for x in ys)}")
    if curve:
        pts = [(t, m) for t, m in curve]
        print("\nscore-vs-time (mean across in-progress passes):")
        for frac in (0.25, 0.5, 0.75, 0.9, 1.0):
            t_end = pts[-1][0]
            t, m = min(pts, key=lambda p: abs(p[0] - frac * t_end))
            print(f"  {t/3600:5.2f} h  {m:6.2f}")


# --------------------------------------------------------------------------- per-level
def level_breakdown(by_game):
    """Completion-only score (every completed level scored 100) vs actual; the gap is the
    efficiency loss. Also per-level-index completion rate and mean efficiency."""
    comp_scores, act_scores = [], []
    by_idx = defaultdict(lambda: [0, 0, []])  # level index -> [attempted runs, completed, ratios]
    over_baseline = under = 0
    for g, rs in by_game.items():
        cs, as_ = [], []
        for r in rs:
            n = len(r.level_actions)
            W = n * (n + 1) / 2
            cs.append(100 * sum(i + 1 for i in range(r.levels_done)) / W)
            as_.append(r.score)
            for i, (ac, b) in enumerate(zip(r.level_actions, r.level_baseline)):
                d = by_idx[i + 1]
                d[0] += 1
                if i < r.levels_done:
                    d[1] += 1
                    d[2].append(level_score(ac, b, True))
                    if ac > b:
                        over_baseline += 1
                    else:
                        under += 1
        comp_scores.append(np.mean(cs))
        act_scores.append(np.mean(as_))
    return dict(
        completion_only=float(np.mean(comp_scores)), actual=float(np.mean(act_scores)),
        efficiency_loss=float(np.mean(comp_scores) - np.mean(act_scores)),
        completed_levels_over_baseline=over_baseline, completed_levels_at_or_under=under,
        by_index={k: (v[0], v[1], float(np.mean(v[2])) if v[2] else float("nan"))
                  for k, v in sorted(by_idx.items())},
    )


# --------------------------------------------------------------------------- compare
def paired(by_a, by_b, n_boot=20_000):
    games = sorted(set(by_a) & set(by_b))
    ya = [np.array([r.score for r in by_a[g]]) for g in games]
    yb = [np.array([r.score for r in by_b[g]]) for g in games]
    d = np.array([b.mean() - a.mean() for a, b in zip(ya, yb)])
    G = len(games)
    mean_d = float(d.mean())
    # (1) game-sampling inference: paired t and bootstrap over games
    se_g = float(d.std(ddof=1) / math.sqrt(G))
    tcrit = 2.064 if G == 25 else 1.96 + 2.5 / max(G, 1)  # t_{0.975,24}=2.064
    bidx = RNG.integers(0, G, (n_boot, G))
    boot = d[bidx].mean(axis=1)
    # (2) fixed-set (noise-only) inference: hierarchical bootstrap of passes within games
    hb = np.zeros(n_boot)
    for a, b in zip(ya, yb):
        ia = RNG.integers(0, len(a), (n_boot, len(a)))
        ib = RNG.integers(0, len(b), (n_boot, len(b)))
        hb += b[ib].mean(axis=1) - a[ia].mean(axis=1)
    hb /= G
    def _v(x):
        return x.var(ddof=1) / len(x) if len(x) > 1 else 0.0  # single pass: noise not estimable
    se_fixed = math.sqrt(sum(_v(a) + _v(b) for a, b in zip(ya, yb))) / G
    # (3) sign-flip permutation test over games (exact-ish, 2^25 too many -> MC)
    flips = RNG.choice([-1.0, 1.0], size=(100_000, G))
    perm = (flips * d).mean(axis=1)
    p_perm = float((np.abs(perm) >= abs(mean_d) - 1e-12).mean())
    # (4) within-game label permutation (fixed set): shuffle A/B labels among passes of each game
    pooled = [np.concatenate([a, b]) for a, b in zip(ya, yb)]
    nperm = 20_000
    stat = np.zeros(nperm)
    for pool, a in zip(pooled, ya):
        k = len(a)
        perms = np.argsort(RNG.random((nperm, len(pool))), axis=1)
        x = pool[perms]
        stat += x[:, k:].mean(axis=1) - x[:, :k].mean(axis=1)
    stat /= G
    p_within = float((np.abs(stat) >= abs(mean_d) - 1e-12).mean())
    wins = int((d > 0).sum()); losses = int((d < 0).sum())
    return dict(
        games=G, mean_delta=mean_d,
        ci95_t=(mean_d - tcrit * se_g, mean_d + tcrit * se_g), se_game=se_g,
        ci95_boot_games=(float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))),
        ci95_fixed_set=(float(np.percentile(hb, 2.5)), float(np.percentile(hb, 97.5))), se_fixed=se_fixed,
        p_signflip=p_perm, p_within_game_perm=p_within,
        wins=wins, losses=losses, ties=G - wins - losses,
        per_game=sorted(zip(games, [float(a.mean()) for a in ya], [float(b.mean()) for b in yb], d.tolist()),
                        key=lambda t: t[3]),
    )


def decide(res, floor):
    lo, hi = res["ci95_boot_games"]
    if lo > 0 and res["mean_delta"] >= floor:
        return "ADOPT B: CI excludes 0 and delta clears the practical floor"
    if hi < 0:
        return "REJECT B: B is worse (CI entirely below 0)"
    if lo > 0:
        return "B better but below practical floor: adopt only if free (no cost/risk)"
    return "INCONCLUSIVE: CI straddles 0 -> do not claim a difference; more passes or drop"


def cmd_compare(a):
    A, _, sa, ca = load(a.a)
    B, _, sb, cb = load(a.b)
    la, lb = a.label_a or Path(a.a).stem, a.label_b or Path(a.b).stem
    print(f"A = {la}: {ca}  (printed mean {sa.get('mean score')})")
    print(f"B = {lb}: {cb}  (printed mean {sb.get('mean score')})")
    r = paired(A, B)
    print(f"\npaired over {r['games']} games, delta = B - A = {r['mean_delta']:+.2f} pts")
    print(f"  game-sampling 95% CI (t) ........ [{r['ci95_t'][0]:+.2f}, {r['ci95_t'][1]:+.2f}]  SE {r['se_game']:.2f}")
    print(f"  game-sampling 95% CI (bootstrap)  [{r['ci95_boot_games'][0]:+.2f}, {r['ci95_boot_games'][1]:+.2f}]")
    print(f"  fixed-set 95% CI (passes boot) .. [{r['ci95_fixed_set'][0]:+.2f}, {r['ci95_fixed_set'][1]:+.2f}]  SE {r['se_fixed']:.2f}")
    print(f"  sign-flip permutation p (games) . {r['p_signflip']:.4f}")
    print(f"  within-game label-perm p ........ {r['p_within_game_perm']:.4f}")
    print(f"  games B>A / B<A / tie ........... {r['wins']} / {r['losses']} / {r['ties']}")
    print(f"  decision (floor {a.floor} pts): {decide(r, a.floor)}")
    print("\nper-game (sorted by delta):  A_mean  B_mean  delta")
    for g, ma, mb, dd in r["per_game"]:
        print(f"  {g:16s} {ma:7.2f} {mb:7.2f} {dd:+7.2f}")
    print("\nwhere did the delta come from? (completion vs efficiency)")
    for name, bg in ((la, A), (lb, B)):
        lv = level_breakdown(bg)
        print(f"  {name:12s} completion-only {lv['completion_only']:6.2f}  actual {lv['actual']:6.2f}  "
              f"efficiency loss {lv['efficiency_loss']:5.2f}  completed levels over/under human baseline "
              f"{lv['completed_levels_over_baseline']}/{lv['completed_levels_at_or_under']}")
    la_, lb_ = level_breakdown(A), level_breakdown(B)
    print("\n  level  A: completion% mean-level-score | B: completion% mean-level-score")
    for k in sorted(set(la_["by_index"]) | set(lb_["by_index"])):
        x = la_["by_index"].get(k, (0, 0, float('nan')))
        y = lb_["by_index"].get(k, (0, 0, float('nan')))
        print(f"  {k:5d}  {100*x[1]/max(x[0],1):5.1f}% {x[2]:6.1f}        | {100*y[1]/max(y[0],1):5.1f}% {y[2]:6.1f}")
    ta = sum(r.actions for v in A.values() for r in v); tb = sum(r.actions for v in B.values() for r in v)
    ka = sum(r.tokens for v in A.values() for r in v); kb = sum(r.tokens for v in B.values() for r in v)
    print(f"\n  actions A {ta}  B {tb}   tokens A {ka}  B {kb}   tokens/action A {ka/ta:.0f}  B {kb/tb:.0f}")


# --------------------------------------------------------------------------- power
def cmd_power(a):
    by_game, *_ = load(a.log)
    s = noise_stats(by_game)
    mean_w, G = s["mean_w"], s["games"]
    het = a.het_sd  # SD across games of the TRUE per-game effect (treatment heterogeneity)
    print(f"within-game run SD {math.sqrt(mean_w):.2f}, games {G}, assumed effect heterogeneity SD {het:.1f}")
    print("Minimum detectable effect (two-sided 5%, 80% power) for a paired offline A/B run:")
    print(" passes/arm   fixed-set MDE   new-game-set MDE   GPU-hours (both arms, ~8.2h per 4-pass job)")
    for P in (1, 2, 4, 8):
        se_fixed = math.sqrt(2 * mean_w / P / G)
        se_gen = math.sqrt(2 * mean_w / P / G + het ** 2 / G)
        hours = 2 * 8.2 * P / 4
        print(f"   {P:3d}        {2.80*se_fixed:6.2f}          {2.80*se_gen:6.2f}            {hours:5.1f}")
    if a.delta:
        for P in range(1, 33):
            if 2.80 * math.sqrt(2 * mean_w / P / G) <= a.delta:
                print(f"\nTo detect {a.delta} pts on the fixed 25 games you need >= {P} passes per arm.")
                break


def cmd_allocate(a):
    """Neyman allocation of a fixed slot budget (concurrency) across games: n_g ~ sigma_g.
    Games with zero observed variance still get `--min` passes as a regression guard."""
    by_game, *_ = load(a.log)
    games, ys = score_matrix(by_game)
    G = len(games)
    sd = np.array([y.std(ddof=1) for y in ys])
    pooled = float(np.sqrt(np.mean(sd ** 2)))
    sd_eff = np.maximum(sd, a.floor_sd)  # never trust a 0 SD from 4 samples completely
    N = a.slots
    n = np.maximum(a.min, np.round(N * sd_eff / sd_eff.sum())).astype(int)
    while n.sum() > N:
        n[np.argmax(n)] -= 1
    while n.sum() < N:
        n[np.argmax(sd_eff / n)] += 1
    var_uniform = float(np.sum(sd ** 2 / (N / G))) / G ** 2
    var_ney = float(np.sum(sd ** 2 / n)) / G ** 2
    print(f"slots={N} games={G} pooled within SD={pooled:.2f}")
    print(f"SE of game-mean score (one arm): uniform {N/G:.0f}/game {math.sqrt(var_uniform):.2f}  "
          f"-> Neyman {math.sqrt(var_ney):.2f}  (variance x{var_ney/var_uniform:.2f})")
    for g, s, k in sorted(zip(games, sd, n), key=lambda t: -t[2]):
        print(f"  {g:16s} sd {s:6.2f}  passes {k}")


def cmd_lb(a):
    groups = [a.scores] + (a.pair or [])
    # pooled within-group variance of identical-code public-LB submissions
    ss = df = 0.0
    for g in groups:
        x = np.array(g, float)
        if len(x) > 1:
            ss += ((x - x.mean()) ** 2).sum(); df += len(x) - 1
    sd = math.sqrt(ss / df)
    # chi-square CI for an SD with df degrees of freedom (df small -> very wide)
    from math import sqrt
    chi = {1: (0.000982, 5.024), 2: (0.0506, 7.378), 3: (0.216, 9.348), 4: (0.484, 11.143)}
    lo_q, hi_q = chi.get(int(df), (None, None))
    print(f"pooled single-submission SD = {sd:.2f} pts on {int(df)} df")
    if lo_q:
        print(f"95% CI for that SD: [{sqrt(df*sd*sd/hi_q):.2f}, {sqrt(df*sd*sd/lo_q):.2f}]")
    for n in (1, 2, 3):
        print(f"  {n} submission(s) per arm -> MDE (5%/80%) = {2.80*sd*math.sqrt(2/n):.1f} pts; "
              f"95% CI half-width on a delta = {1.96*sd*math.sqrt(2/n):.1f}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("noise"); p.add_argument("log"); p.add_argument("--per-game", action="store_true")
    p.set_defaults(fn=cmd_noise)
    p = sub.add_parser("compare"); p.add_argument("a"); p.add_argument("b")
    p.add_argument("--label-a"); p.add_argument("--label-b")
    p.add_argument("--floor", type=float, default=2.0, help="practical-significance floor in points")
    p.set_defaults(fn=cmd_compare)
    p = sub.add_parser("power"); p.add_argument("log"); p.add_argument("--delta", type=float, default=None)
    p.add_argument("--het-sd", type=float, default=8.0)
    p.set_defaults(fn=cmd_power)
    p = sub.add_parser("allocate"); p.add_argument("log")
    p.add_argument("--slots", type=int, default=50, help="runs per arm (e.g. 50 = half of a 100-slot job)")
    p.add_argument("--min", type=int, default=1); p.add_argument("--floor-sd", type=float, default=5.0)
    p.set_defaults(fn=cmd_allocate)
    p = sub.add_parser("lb"); p.add_argument("scores", type=float, nargs="+")
    p.add_argument("--pair", type=float, nargs="+", action="append",
                   help="another group of identical-code LB scores (repeatable)")
    p.set_defaults(fn=cmd_lb)
    a = ap.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
