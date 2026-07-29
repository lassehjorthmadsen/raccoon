# exp015 — Static PR Benchmark (BGSage Money Benchmark)

**Question**: What is the checker-play Performance Rating (PR) of our distilled
networks (exp014 2-ply, exp011b 0-ply) vs GNUBG at 0-ply and 2-ply, measured
on a static noise-free benchmark?

**Primary metric**: Checker-play PR on the BGSage money benchmark (14,693
decisions, cubeless_equity reference). Lower is better; 0.00 = perfect cubeless
play. Secondary metric: R² / MSE of predicted vs reference equity across all
327,451 candidate positions (split by reference tier).

## Methodology

- **Benchmark**: `../bgsage/data/money_benchmark/benchmark.json.gz`
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
| Raccoon exp011b | 10×256 RaccoonNet, distilled from GNUBG 0-ply labels (outcomes6 head, best epoch) |
| Raccoon exp014 | 10×256 RaccoonNet, distilled from GNUBG 2-ply labels (scalar head, best epoch) |

## Results

### Move Selection (PR, n=14,693 decisions)

| Engine | PR | n | Blunders | prace | race | attack | prime | anchor |
|--------|-----|------|----------|-------|------|--------|-------|--------|
| GNUBG 2-ply | 0.56 | 14,693 | 11 | 0.05 | 0.80 | 0.50 | 0.47 | 0.76 |
| GNUBG 0-ply | 2.14 | 14,693 | 95 | 0.16 | 2.72 | 2.15 | 2.66 | 2.32 |
| exp014 (2-ply dist) | 2.51 | 14,693 | 133 | 1.62 | 2.44 | 2.47 | 3.05 | 2.74 |
| exp011b (0-ply dist) | 3.53 | 14,693 | 227 | 1.51 | 3.90 | 3.36 | 4.31 | 3.94 |

(n per game plan: prace=1,816, race=3,120, attack=4,111, prime=2,453, anchor=3,193)

### Eval Accuracy (R² / MSE on cubeless equity, n=327,451 positions)

| Engine | R² (rollout) | MSE | n | R² (3T) | MSE | n | R² (3P) | MSE | n |
|--------|-------------|-----|------|---------|-----|------|---------|-----|------|
| GNUBG 2-ply | 0.9969 | 0.0019 | 149,113 | 0.9948 | 0.0017 | 61,917 | 0.9988 | 0.0003 | 116,421 |
| GNUBG 0-ply | 0.9965 | 0.0021 | 149,113 | 0.9939 | 0.0019 | 61,917 | 0.9919 | 0.0022 | 116,421 |
| exp014 (2-ply dist) | 0.9919 | 0.0049 | 149,113 | 0.9835 | 0.0053 | 61,917 | 0.9811 | 0.0052 | 116,421 |
| exp011b (0-ply dist) | 0.9831 | 0.0101 | 149,113 | 0.9658 | 0.0109 | 61,917 | 0.9561 | 0.0121 | 116,421 |

## Comparability Note

These PR numbers are **not directly comparable** to BGSage's published figures
(Sage 3T = 0.21 PR) because we use cubeless_equity reference and score checker
decisions only, while BGSage uses cubeful equity and includes cube decisions.
The 3P reference tier is Sage 3-ply (a different engine from our GNUBG teacher).

## Conclusion

2-ply distillation (exp014) vs 0-ply distillation (exp011b): exp014 is stronger
on both metrics — PR 2.51 vs 3.53, R²(rollout) 0.9919 vs 0.9831. The stronger
teacher produces a better student.

Neither raccoon network matches GNUBG 0-ply at move selection (PR 2.14) despite
exp014 having trained on GNUBG 2-ply labels. The gap is consistent across game
plans. Both networks are weakest on pure-race positions (PR 1.5+) where GNUBG
scores near-perfectly (0.05–0.16).
