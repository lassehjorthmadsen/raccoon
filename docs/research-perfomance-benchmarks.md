# Query 

I'm building a neural network to play backgammon. Research the strength of current networks: GNUBG, XG, Sage, and BGBlitz. I'm especially interested in how far they are from perfect play in money game

# Response

## Bottom line

The best recent apples-to-apples **money-game** evidence puts the strongest practical Sage and XG settings extremely close to rollout-quality play:

* **Sage 3T:** **0.21 PR**
* **XG Roller++:** **0.32 PR**
* **Sage/XG 4-ply:** **0.41 / 0.46 PR**
* **Sage/XG 3-ply:** **0.58 / 0.57 PR**

That benchmark used 500 money games with Jacoby and beavers enabled, covering 17,535 decisions. It built a layered reference: clear decisions were accepted at Sage 3-ply, closer ones at Sage 3T, and the hardest were rolled out in batches of 1,296 up to 20,736 trials. ([BGSage][1])

The important qualification is that these are distances from a **strong Sage-based rollout reference**, not mathematically perfect play. Full rollouts are the closest available practical oracle, but they still use approximate policies and evaluators. The study explicitly acknowledges possible “home-field” bias toward Sage. ([BGSage][1])

## Current strength by engine

| Engine and strongest relevant mode |                                                                                                                 Best public evidence for money play | Assessment                                                                                                                                                                                                                              |
| ---------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------: | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Open Sage 3T**                   |                                                                                                                **0.21 PR**; checker 0.18, cube 0.36 | Best currently documented result on a substantial, reproducible money-game corpus. Probably a small edge over XG++, but not conclusively established because the reference uses Sage evaluations and rollouts.                          |
| **XG Roller++**                    |                                                                                                                **0.32 PR**; checker 0.31, cube 0.38 | Essentially in the same top tier as Sage. The currently distributed desktop version is still XG 2.10, released in February 2013, but its search and truncated-rollout machinery remains highly competitive. ([Extreme Gammon][2])       |
| **GNUbg high-ply**                 |                                                                   Historical 2012 benchmark: **0.35 PR at GNU 4-ply**, 0.46 at 3-ply, 0.58 at 2-ply | Historically almost exactly XG 4-ply strength. There is no comparable current public money-game test. GNU’s terminology is offset: its 2-ply corresponds approximately to other engines’ 3-ply. ([Extreme Gammon][3])                   |
| **BGBlitz TachiAI VI**             | No current public money-game benchmark. Developer reports **0.29 average PR at 5-ply**, 0.33 at 4-ply and 0.42 at 3-ply over twenty 5-point matches | Plausibly competitive with the leading engines, but presently unrankable for money play: the sample is small, is match rather than money play, and the announcement does not describe an external adjudication protocol. ([BGBlitz][4]) |

The first two rows are the only current direct comparison. The GNUbg and BGBlitz numbers should not be used to construct a four-engine ranking.

### Sage

The shared benchmark gives a particularly useful strength ladder:

| Sage mode            |       PR |
| -------------------- | -------: |
| Raw 1-ply evaluation |     2.59 |
| 2-ply                |     1.64 |
| 3-ply                |     0.58 |
| 4-ply                |     0.41 |
| 1T truncated rollout |     0.50 |
| 2T truncated rollout |     0.26 |
| 3T truncated rollout | **0.21** |

This shows that the leaf network is only part of the system: moving from raw evaluation to 3T reduces measured error by roughly a factor of twelve. ([BGSage][1])

Sage is also unusually useful for a new project because its Python/C++ engine, weights, multi-ply search, rollouts and training framework are public. Its model development uses specialized game-plan networks rather than one monolithic evaluator; recent stages use player/opponent game-plan-pair networks and separate backgame specialists. ([GitHub][5])

### XG

XG Roller++ is not a full rollout. XG documents it as a short variance-reduced rollout using 360 trials, with high-ply decisions early in each path and lower-ply decisions later. Checker and cube evaluations use somewhat different truncation settings. ([Extreme Gammon][6])

The 2026 shared benchmark indicates that XG remains:

* effectively tied with Sage at ordinary 3-ply;
* approximately 0.05 PR behind at 4-ply;
* approximately 0.11 PR behind at the strongest truncated-rollout setting.

Those differences are small enough that corpus construction and adjudicator bias matter. On 290 real tournament matches, Sage and XG assigned almost identical player ratings: mean difference +0.002 PR, with a reported 95% interval of ±0.03. That is match-rating agreement rather than proof of equal money-game strength, but it confirms that their practical evaluations are extremely close. ([BGSage][1])

### GNUbg

The best common benchmark remains Michael Depreli’s 2012 study of 500 money games:

* GNUbg 4-ply: **0.35 PR**
* GNUbg 3-ply: **0.46 PR**
* GNUbg 2-ply: **0.58 PR**
* XG 4-ply in the same study: **0.33 PR**
* XG Roller++: **0.11 PR**

That placed GNUbg 4-ply essentially level with XG 4-ply. However, the reference rollouts in that study were produced by XG, so its exceptionally low XGR++ score should also be regarded as partly a home-field result. ([Extreme Gammon][3])

The current GNUbg weight file identifies a compact **250-input, 128-hidden-unit, 5-output** evaluator. This is a useful reminder that a relatively small feed-forward network combined with carefully engineered features, bearoff databases, cube handling and search can remain world-class. ([GitHub][7])

Because no modern neutral benchmark has re-tested current GNUbg against Sage, XG and TachiAI VI, the honest description is **historically XG-4-ply class, current exact strength unknown**.

### BGBlitz

BGBlitz has changed too much for its 2012 result to describe the current engine. The old BGBlitz 2.8 scored 0.90 PR at 4-ply and 1.04 at 3-ply in the Depreli benchmark. Since then, BGBlitz has added more specialized networks, cube improvements, and, in March 2026, TachiAI VI. The developer says TachiAI VI improves on its predecessor by approximately 0.3–0.5 PR depending on depth. BGBlitz 3.4.2 subsequently added 4-ply rollouts and further cube-handling improvements. ([Extreme Gammon][3])

The reported 0.29 average PR at 5-ply sounds comparable to Sage 3T or XG++, but it cannot safely be interpreted that way. It comes from only twenty 5-point matches, not thousands of independently adjudicated money decisions. Until TachiAI VI is scored on the Sage money corpus or another neutral corpus, its money-game gap is unknown.

## Translating PR into loss per game

PR is defined as:

[
\text{PR}=500\times\text{mean equity error per counted decision}.
]

The current money benchmark contains 17,535 decisions over 500 games, or approximately **17.535 counted decisions per player per game**. Therefore, at that decision density:

[
\text{normalized equity loss per 100 games}
\approx 3.507\times\text{PR}.
]

This gives:

| Engine/mode    |   PR | Approximate normalized equity loss per 100 games |
| -------------- | ---: | -----------------------------------------------: |
| Sage 3T        | 0.21 |                                         **0.74** |
| XG Roller++    | 0.32 |                                         **1.12** |
| Sage 4-ply     | 0.41 |                                             1.44 |
| XG 4-ply       | 0.46 |                                             1.61 |
| Sage 3-ply     | 0.58 |                                             2.03 |
| XG 3-ply       | 0.57 |                                             2.00 |
| Sage raw 1-ply | 2.59 |                                             9.08 |

These are **PR-equivalent, cube-normalized equity units**, not literal cash winnings. Actual monetary loss depends on cube levels, cube ownership, resignations, the game distribution and the exact definition of counted decisions. The conversion is nevertheless useful for interpreting scale: Sage 3T and XG++ are losing on the order of **one normalized point per hundred games** against the benchmark oracle. ([BGSage][1])

## How far are they from genuinely perfect play?

There are three different answers:

### Demonstrated gap

Against the strongest published practical reference:

* Sage 3T is **0.21 PR** away.
* XG++ is **0.32 PR** away.
* Strong fixed-depth search is approximately **0.4–0.6 PR** away.

That is the firmest quantitative answer.

### Unknown common-mode gap

The benchmark cannot detect positions where Sage, XG and the rollout policy all make the same mistake. Those common-mode errors are especially plausible in unusual backgames, deep containment positions, complex recube situations and positions poorly represented in training data.

Of the 16,889 benchmark positions, 7,652 were accepted at Sage 3-ply, 3,260 at Sage 3T and 5,977 were fully rolled out. Even the full rollouts used Sage 3-ply decisions along the simulation paths. Consequently, 0.21 and 0.32 are neither upper nor lower bounds on distance from perfect play. ([BGSage][1])

### Practical engineering estimate

A reasonable planning estimate for Sage 3T or XG++ is:

> **Approximately one normalized equity point per 100 ordinary money games from perfect-quality play, with at least a factor-of-two uncertainty.**

In other words, roughly **0.5–2 normalized points per 100 games**, or about **0.15–0.6 PR** at the current benchmark’s decision density, is a sensible order-of-magnitude band. This is not a confidence interval. It reflects uncertainty about oracle bias and shared errors.

One reason not to claim greater precision is that XGR++ scored 0.11 PR in the 2012 XG-adjudicated benchmark but 0.32 in the current Sage-adjudicated benchmark. Different corpora, rollout policies and home-engine effects moved the absolute figure by about 0.2 PR, even though XG’s practical strength did not suddenly change by that amount. ([Extreme Gammon][3])

The engines’ direct disagreements are nevertheless sparse. In a separate 200-game money sample, Sage 3T and XG++ disagreed on only 130 of 7,404 decisions, or 1.8%; only 31 disagreements exceeded 0.01 equity. Eleven larger disagreements were rolled out: Sage was best in six, XG in two, and neither engine found the rollout winner in three. This strongly supports “very close,” while also demonstrating that both engines still miss moves. ([BGSage][1])

## Targets for your network

For a new money-game engine evaluated under the same PR convention, useful milestones are:

| Evaluation level                          |                       Competitive target |
| ----------------------------------------- | ---------------------------------------: |
| Raw leaf evaluator                        |                              **≤2.5 PR** |
| 2-ply                                     |                          **≤1.5–1.8 PR** |
| 3-ply                                     |                              **≤0.6 PR** |
| 4-ply                                     |                             **≤0.45 PR** |
| Strong truncated rollout                  |                              **≤0.3 PR** |
| Credible improvement over current leaders | **<0.2 PR**, using a neutral adjudicator |

The most important design lesson is that **search and specialization currently buy more than merely enlarging the network**. Sage’s raw evaluator is around 2.6 PR, yet the full 3T system reaches 0.21. The remaining errors also suggest prioritizing cube modeling: at Sage 3T, checker PR is 0.18 but cube PR is 0.36; for XG++ the corresponding figures are 0.31 and 0.38. ([BGSage][1])

A credible evaluation protocol should generate positions from several engines rather than only your own self-play, collect candidate moves from all engines, adjudicate close positions with both mixed-policy and engine-specific rollouts, and report checker PR, cube PR, game-plan breakdowns, runtime and rule settings separately. Otherwise, once you approach 0.3 PR, benchmark home-field effects can easily become as large as the improvement you are trying to measure.

[1]: https://www.bgsage.ai/botperformance/ "https://www.bgsage.ai/botperformance/"
[2]: https://www.extremegammon.com/download.aspx "https://www.extremegammon.com/download.aspx"
[3]: https://www.extremegammon.com/studies.aspx "https://www.extremegammon.com/studies.aspx"
[4]: https://www.bgblitz.com/news.html "https://www.bgblitz.com/news.html"
[5]: https://github.com/markbgsage/bgsage "https://github.com/markbgsage/bgsage"
[6]: https://www.extremegammon.com/support.aspx "https://www.extremegammon.com/support.aspx"
[7]: https://github.com/gnubg/dev/raw/refs/heads/main/gnubg/gnubg.weights "https://github.com/gnubg/dev/raw/refs/heads/main/gnubg/gnubg.weights"
