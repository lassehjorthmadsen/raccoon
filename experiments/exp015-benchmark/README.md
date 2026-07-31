# exp015 — Static PR Benchmark (BGSage Money Benchmark)

**Question**: What is the checker-play Performance Rating (PR) of our distilled
networks (exp014 2-ply, exp011b 0-ply) vs GNUBG at 0-ply and 2-ply, measured
on a static noise-free benchmark?

**Primary metric**: Checker-play PR on the BGSage money benchmark (14,693
decisions, cubeless_equity reference). Lower is better; 0.00 = perfect cubeless
play. Secondary metric: R² / MSE of predicted vs reference equity across all
327,451 candidate positions (split by reference tier).

> **Correction, 2026-07-31.** This README's original engine table (see git
> history at commit `b492581`) described the two distilled checkpoints as
> "best epoch" — which meant `best.pt`, an artifact of `train_distill.py`'s
> noisy in-loop n=40-game selector, not the properly cross-validated `ep3.pt`
> that [exp011b/exp012b](../../docs/pretraining_analysis.qmd#exp011b) actually
> selected and every other section of the project docs reports against. For
> the outcomes6 arm, `best.pt` turned out to be numerically identical to
> `ep1.pt` — the selector froze early and never updated. The bug was caught
> while [exp016](../exp016-benchmark-revisit/) was re-deriving these same
> checkpoints from scratch and couldn't reproduce this experiment's numbers;
> confirmed by reproducing the original (wrong) numbers exactly off `best.pt`
> on a subsample. All numbers below are the corrected `ep3.pt` re-run, saved
> with `--output` so the raw per-decision JSON is reproducible going forward.
> GNUBG's own rows were never affected (no checkpoint selection involved).

## Methodology

- **Benchmark**: `data/bgsage/money_benchmark/benchmark.json.gz` (mirrored from
  the sibling `bgsage` repo — see `data/README.md`)
  - 500 Sage-3P vs Sage-3P money games, 14,693 checker decisions
  - Adaptive-precision reference: Sage 3-ply (7,652), Sage 3T (3,260), Sage rollout (5,977)
  - Per-position game plan: purerace / racing / attacking / priming / anchoring
- **PR**: mean(max(0, best_cubeless_eq - chosen_cubeless_eq)) × 500
- **R²/MSE**: predicted equity vs reference cubeless_equity, across all candidate
  post-move boards (327,451 positions total), split by reference tier
- **Scope**: Checker decisions only (cube decisions skipped — all engines are cubeless)
- **Reference**: `cubeless_equity` field (not cubeful — avoids constant bias for cubeless engines)
- **Script**: `scripts/eval_benchmark_pr.py`

## Engines Tested

| Engine | Description |
|--------|-------------|
| GNUBG 0-ply | Raw NN evaluation, no search |
| GNUBG 2-ply | Full-width 2-ply search |
| Raccoon exp011b | 10×256 RaccoonNet, distilled from GNUBG 0-ply labels (`exp011b-distill/outcomes6/checkpoints/ep3.pt` — the properly cross-validated winner, not `best.pt`) |
| Raccoon exp014 | 10×256 RaccoonNet, distilled from GNUBG 2-ply labels (`exp014-distill/scalar_2ply/checkpoints/ep3.pt`, likewise not `best.pt`) |

## Results

### Move Selection (PR, n=14,693 decisions)

| Engine | PR | n | Blunders | prace | race | attack | prime | anchor |
|--------|-----|------|----------|-------|------|--------|-------|--------|
| GNUBG 2-ply | 0.56 | 14,693 | 11 | 0.05 | 0.80 | 0.50 | 0.47 | 0.76 |
| GNUBG 0-ply | 2.14 | 14,693 | 95 | 0.16 | 2.72 | 2.15 | 2.66 | 2.32 |
| exp014 (2-ply dist) | 2.16 | 14,693 | 96 | 1.13 | 2.32 | 1.97 | 2.68 | 2.44 |
| exp011b (0-ply dist) | 2.75 | 14,693 | 149 | 0.85 | 3.15 | 2.65 | 3.46 | 3.02 |

(n per game plan: prace=1,816, race=3,120, attack=4,111, prime=2,453, anchor=3,193)

### Eval Accuracy (R² / MSE on cubeless equity, n=327,451 positions)

| Engine | R² (rollout) | MSE | n | R² (3T) | MSE | n | R² (3P) | MSE | n |
|--------|-------------|-----|------|---------|-----|------|---------|-----|------|
| GNUBG 2-ply | 0.9969 | 0.0019 | 149,113 | 0.9948 | 0.0017 | 61,917 | 0.9988 | 0.0003 | 116,421 |
| GNUBG 0-ply | 0.9965 | 0.0021 | 149,113 | 0.9939 | 0.0019 | 61,917 | 0.9919 | 0.0022 | 116,421 |
| exp014 (2-ply dist) | 0.9927 | 0.0044 | 149,113 | 0.9850 | 0.0048 | 61,917 | 0.9814 | 0.0051 | 116,421 |
| exp011b (0-ply dist) | 0.9935 | 0.0039 | 149,113 | 0.9870 | 0.0042 | 61,917 | 0.9839 | 0.0044 | 116,421 |

## Comparability Note

These PR numbers are **not directly comparable** to BGSage's published figures
(Sage 3T = 0.21 PR) because we use cubeless_equity reference and score checker
decisions only, while BGSage uses cubeful equity and includes cube decisions.
The 3P reference tier is Sage 3-ply (a different engine from our GNUBG teacher).

## Conclusion

Move selection and value-fit **disagree** on which distilled net is stronger.
On PR, exp014 (2-ply dist) beats exp011b (0-ply dist) — 2.16 vs 2.75 — and
comes within noise of GNUBG 0-ply itself (2.14). But on rollout-tier R²/MSE,
the ordering flips: exp011b fits the external rollout reference slightly
better (R²=0.9935 vs 0.9927, MSE=0.0039 vs 0.0044). This doesn't contradict
exp014's own held-out-R² result (tied at 0.9983, but measured against each
net's *own* training target, not this external reference) — it's a genuine
demonstration that global value-fit and within-decision move-selection are
different skills that can disagree (see also
[exp016](../exp016-benchmark-revisit/)'s value-head panel, same pattern).

Neither raccoon network matches GNUBG 0-ply at move selection (PR 2.14)
despite exp014 having trained on GNUBG 2-ply labels — though on most game
plans (racing, attacking) exp014 is at or near GNUBG-0-ply's own PR. Both
networks are sharply weaker specifically on pure-race positions (PR 0.85–1.13,
5–7× GNUBG-0-ply's 0.16) — the largest relative gap of any game plan, and
still unexplained.
