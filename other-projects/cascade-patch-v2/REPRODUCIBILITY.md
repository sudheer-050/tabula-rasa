# Reproduce the paper

[Read the paper](PAPER.md) · [PDF](paper.pdf)

This package studies the archived cascade-patch-v2 lineage, not the newer `arcduck2` agent.

## Quick start

Download this directory with its two raw logs and scripts. Use Python 3.10+:

```bash
python -m pip install numpy==2.2.6
python reproduce.py
```

No API credentials, game server, model weights, network access during analysis, or GPU are needed. `reproduce.py` writes `paper_results.json` and `paper_verification.txt` beside itself.

The code preserves the original `analyze.py` and `parse_log.py` modules. Reproduction uses their calculation paths; any broad generalization wording in older module documentation is not a claim made by the paper.

## Claim-to-evidence map

| Claim | Evidence | Check |
| --- | --- | --- |
| Displayed public score is 29.97 on Version 2 | `paper_score.jpg`, `paper_version.jpg`, exact URL in `paper_evidence.json` | Signed-in read-only Kaggle UI observation on 2026-10-19; notebook currently private |
| 200 game-pass records, 25 games x 4 passes per job | Both raw `.log` files | Parser assertions in `reproduce.py` |
| Every score reconstructs within 0.005 | Per-level records and scorer replica | Hashes checked, all 200 differences asserted |
| Offline means and generated-work totals | Raw `[finished]` records | Aggregates compared against manifest |
| -22.2442 paired mean difference | Per-game means | Seeded 20,000-sample game bootstrap |
| Successful-level replay unused under stated semantics | `archived_cascade_v2.py` | Actual patch executed against an explicit fake interface; zero replay returns |
| Bound callback metadata is lost | Same patch | Original callback owner present, wrapped owner absent |
| Undo retained and failed attempt excluded | Same patch | Cache contents asserted after failures/resets/completions |
| Fixed completion cannot exceed completed-weight cap | Versioned scoring formula | Conditional proof in paper plus numerical cap test |

Screenshots document the score display; hashes establish artifact identity; tests establish behavior under their assumptions. None alone independently proves leaderboard authenticity, generalization, or causality.

## Raw records and privacy

The two archived logs are included byte-for-byte, with SHA-256 values in `paper_evidence.json`. They contain runtime setup, public demo identifiers, per-run metrics, and diagnostics. They were checked for credential-like values before release. No model checkpoints or private competition-game data are included.

The notebook itself is fingerprinted rather than redistributed in full. The archived custom cascade cell is extracted without source edits. A screenshot does not prove byte identity between the locally downloaded notebook and Kaggle's stored artifact.

## Rebuild the PDF

With ReportLab installed, run:

```bash
python -m pip install reportlab==4.4.9
python build_paper.py
```

The PDF builder consumes `PAPER.md`, the two figure PNGs, and the equation PNGs. The SVG figure copies preserve scalable charts. PDF layout and pagination can vary with fonts and ReportLab versions.

## Provenance and boundaries

- The verified score is **29.97**, `scriptVersionId=356244130`. The earlier conversational value 29.92 was not used as the paper's result.
- Offline results are all-pass means, not best-pass aggregates, and are not comparable numerically to the competition score.
- Four passes share each historical job's GPU. Bootstrap intervals do not eliminate job-level confounding or establish hidden-set generalization.
- The v4/summarization log identifies a combined intervention. It is not evidence that all summarization methods are harmful.
- A separately archived 26.83 leaderboard code file uses v2-style logic. This paper does not attribute that leaderboard result to the offline summary intervention.
- The newer repository's completion-ceiling sample metrics are excluded from this study.
- No new GPU run or Kaggle submission was performed for this paper.

## Attribution

The surrounding project uses Tufa Labs' Duck/TAAF framework, Daniel Franzen's modifications, Qwen/Intel/Albucino checkpoint components, and Pennyroyal/SGLang serving. The cascade notebook also credits Venkata Sathwik Reddy P. See the paper and the repository's original license/acknowledgements. Upstream source, models, and runtime bundles retain their own licenses.
