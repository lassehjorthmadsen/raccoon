# Project Goal

Build a backgammon AI that clearly outplays GNUBG at **money game without the
Jacoby rule**, as the stepping stone to match play.

The Jacoby exclusion is deliberate, not a simplification. Jacoby zeroes gammons
while the cube is centred, which changes correct checker play as well as correct
cube play: the equity-maximising move becomes bolder, trading gammon chances of
both signs for raw win probability. Match play — the long-term target — has no
Jacoby, so non-Jacoby money is the closer neighbour and the honest rehearsal.
Everything below the cube layer is unaffected: the network is a cubeless
evaluator either way.

## Assumptions

Research and challenge these if needed.

- GNUBG uses TD reinforcement learning. A superior system will likely need a more modern framework (AlphaZero-style policy+value + MCTS).
- GNUBG outputs cubeless outcome probability distributions (win/lose normal, gammon, backgammon) then applies Janowski formulas for cube actions. A superior system may need a more sophisticated model, perhaps outputting cubeful equities directly.

## A known gap between the goal and the benchmark

The BGSage money benchmark, our primary static metric, was generated **with Jacoby
on**. That rule zeroes gammons while the cube is centred, so on the 5,657 of 14,693
checker decisions where the cube is still in the middle, the reference engine played
boldly — trading gammon chances of both signs for raw win probability — and the
rollout continuations behind those labels were played the same way.

Regenerating it without Jacoby is not a re-score. Jacoby changes which moves get
played, so the 500 generating games diverge and the whole position set changes;
every tier would have to be rebuilt, including 5,977 rollout positions at 1,296
trials each. That is not worth it for a labelling caveat on a proxy metric, so we
do not plan to.

What we do instead is report both. `scripts/eval_benchmark_pr.py` breaks PR down by
cube location, including an `owned (Jacoby-clean)` subset of 9,036 decisions where
the cube has already been turned and the rule is spent. For the shipped net the two
agree closely — PR 0.950 over the full benchmark against 0.978 on the
Jacoby-clean subset — so the caveat is small in practice. Note that the split
confounds Jacoby labelling with game phase (a centred cube means an earlier
position), so it bounds the taint rather than measuring it. The full-benchmark
number stays the headline, for comparability with every figure published so far.

## Requirements

- Trained using AlphaZero-style self-play with a ResNet policy-value network and MCTS
- Uses OpenSpiel for backgammon game logic
- Terminal interface for human play, using the same standard output as GNUBG
- Automated play against GNUBG CLI for benchmarking
- Game logging in standard backgammon notation
- Training metadata logging: network version, architecture, performance metrics
- Text-based communication protocol (RGP) inspired by UCI, for future frontend integration

## Success Criteria

Raccoon has a positive average win in non-Jacoby money games against GNUBG at
"world class" settings over 1000+ games, with a 95% confidence interval above and
not including zero.

Measured cubelessly today, where Jacoby is a non-concept because the cube never
turns and gammons always count — so every play result so far already satisfies
the non-Jacoby framing. Once the cube ships, the same criterion applies with the
cube live.
