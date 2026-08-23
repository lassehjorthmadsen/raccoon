# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Raccoon is a backgammon AI aiming to beat GNUBG at cubeless money game. OpenSpiel provides game logic; all ML/search code is written from scratch in Python/PyTorch.

**The project began as AlphaZero (ResNet policy-value net + MCTS self-play) and that machinery is still here and runnable — but it is not what the current engine does.** The shipped net was trained by distilling GNUBG onto the value head (`scripts/train_distill.py`), and it selects moves by **0-ply value lookahead** (`raccoon/train/lookahead.py`): evaluate the value head on every legal afterstate, no tree, no policy head. Read `docs/architecture.md` before assuming which path a change affects. When editing that file or this one, keep the two in step — they contradicting each other is the failure mode this note exists to prevent.

## Commands

```bash
make setup          # Install in editable mode with dev deps
make test           # Run all tests (pytest tests/ -v)
make smoke          # Quick sanity check: 2 iterations, 3 games, 10 sims
make train          # Full training run
make eval           # Checkpoint vs checkpoint evaluation
make eval-gnubg     # Automated benchmark against GNUBG (gnubg-nn engine; level=world → full-width 2-ply)
make play           # Interactive terminal play
make download-wildbg # Fetch wildbg-training labeled positions (CC0) into data/wildbg/
make pull-data       # Fetch data/bglab/ + data/bgsage/ from GCS — run once on a fresh machine
make push-data       # Push local changes to data/bglab/ or data/bgsage/ back to GCS
make pretrain-smoke  # 1 epoch on 2k positions — sanity check before a full pretrain
make pretrain NAME=… # Full supervised pretraining on wildbg data

# Run a single test file or test
python3 -m pytest tests/test_encoder.py -v
python3 -m pytest tests/test_mcts.py::test_name -v

# Training with custom params (--experiment-name required; outputs go to experiments/<name>/)
python3 scripts/train.py --experiment-name my-run --iterations 100 --games-per-iter 50 \
  --simulations 100 --lr 0.001 --channels 128 --num-blocks 6 --checkpoint-every 10

# Resume training from a checkpoint (architecture is read from the checkpoint)
python3 scripts/train.py --experiment-name my-run --iterations 100 \
  --resume experiments/my-run/checkpoints/iter_0200.pt

# Supervised pretraining on wildbg data, then continue with self-play
./scripts/download_wildbg.sh   # one-off; ~17 MB into data/wildbg/
python3 scripts/pretrain.py --experiment-name pretrain-wildbg-v1 --epochs 20
python3 scripts/train.py --experiment-name exp007-pretrained \
  --resume experiments/pretrain-wildbg-v1/checkpoints/pretrained.pt --simulations 800

# Stage-2 policy pretraining via 1-ply V-lookahead on bglab match positions.
# Requires the match archive at data/bglab/{lasse,Llabba}/{raw,analyzed} (plain
# files, no git tracking — see data/README.md for how to refresh from source).
# Stage 1 (synthesis) is the long step — ~8 h on iMac CPU, ~minutes on T4 GPU.
python3 scripts/synthesize_policy_dataset.py \
  --pretrained experiments/pretrain-wildbg-v1/checkpoints/pretrained.pt \
  --out data/bglab/cache/policy_cache.npz
python3 scripts/pretrain_policy.py --experiment-name pretrain-wildbg-v2 \
  --base-checkpoint experiments/pretrain-wildbg-v1/checkpoints/pretrained.pt \
  --cache data/bglab/cache/policy_cache.npz --epochs 10

# Value-only distillation of GNUBG onto a fresh net — this is where the shipped
# net came from. One arm per invocation (--value-head scalar|outcomes6).
python3 scripts/train_distill.py --cache-dir data/distill/0ply/run1 \
  --experiment-name exp011-distill/scalar --value-head scalar --epochs 2

# TD(lambda) self-play: value head only, moves chosen by 0-ply lookahead
python3 scripts/train_td.py --experiment-name exp010-td --batches 100

# Score any checkpoint or exported ONNX on the BGSage benchmark (the primary metric)
python3 scripts/eval_benchmark_pr.py --checkpoint experiments/<name>/checkpoints/<ep>.pt
```

### Exporting to the browser engine

The site at raccoonbg.com lives in a **separate repo, `raccoon-website`**, whose
position relative to this one is a per-machine fact: siblings on the Linux dev box,
opposite sides of the WSL boundary on the Windows ones. `--out-dir` defaults to the
sibling layout, so on a sibling checkout this needs no flags at all:

```bash
python3 scripts/export_web_model.py --checkpoint experiments/exp018-distill/checkpoints/ep22.pt
python3 scripts/export_web_fixtures.py
python3 scripts/eval_benchmark_pr.py --onnx ../raccoon-website/app/models/exp018-ep22-fp32.onnx
```

Where the repos are not siblings, export `RACCOON_WEBSITE=/path/to/raccoon-website`
once (e.g. in `.bashrc`) and the same commands work unchanged; `--out-dir` still
overrides both.

**A wrong guess is refused, not created.** Both exporters used to `mkdir -p` their
output, so a default that was wrong on this machine produced a plausible-looking
empty tree, reported success, and left the site shipping the previous weights.
`raccoon/web_export.py` now creates the leaf directory but never its ancestry: if the
parent is missing the run aborts immediately, before the checkpoint load, naming the
resolved absolute path. That catches a wrong default, a moved checkout, and a typo'd
`--out-dir` alike.

The exported graph carries the **value head only** — nothing in the web engine reads
the policy head, since it plays 0-ply. The fixtures are the contract its JS ports of
movegen and the encoder are tested against, so a divergence fails that repo's CI
instead of quietly making the demo play a different game from the one benchmarked.
The last command re-scores the exported file; the site quotes *that* number.

## Architecture

The project follows a milestone-based plan (see `docs/plan.md` for full details). The seven milestones build on each other: env → model → search → training → internal eval → GNUBG benchmark → strength scaling.

### Core Pipeline

1. **`raccoon/env/`** — OpenSpiel wrapper + custom tensor encoder + action mapping
   - `game_wrapper.py`: Wraps `pyspiel.load_game("backgammon")`, handles perspective flipping so the network always sees the board from the current player's view
   - `encoder.py`: Converts board state to **(26, 2, 12)** float32 tensor (26 channels, 2 rows, 12 columns). Channels: 4 checker planes per player, bar/borne-off/dice broadcast planes, mid-doubles flag, plus handcrafted features (pip count, blots, anchors, contact pressure). `CHANNEL_NAMES` is the authoritative registry of channel meanings; `dump_tensor()` pretty-prints the planes for debugging.
   - `actions.py`: Legal action masking over OpenSpiel's 1352 action space

2. **`raccoon/model/network.py`** — `RaccoonNet`: ResNet with shared trunk → policy head (1352 logits) + value head. Default: 6 residual blocks, 128 channels; the shipped net is 10 blocks, 256 channels. `predict()` handles masking + softmax for MCTS inference.
   - The value head is `"scalar"` (one tanh output) **or** `"outcomes6"` (six logits over `[win, win_g, win_bg, lose, lose_g, lose_bg]`), fixed at construction and recorded in the checkpoint. The shipped net is `outcomes6`.
   - **Every value in this codebase is money-equity/3 in [-1, 1] from the to-move player's POV** (win = ±1/3, gammon = ±2/3, backgammon = ±1), trained on **pre-roll** positions. `value_equity()` applies whichever conversion the head needs, so the two head types are interchangeable at play time — downstream code must go through it and never branch on head type.

3. **`raccoon/search/`** — two independent searches; **neither is used by the shipped 0-ply engine**
   - `mcts.py`: AlphaZero MCTS with PUCT selection, used by self-play training. Chance nodes (dice rolls) are sampled and skipped — the tree only contains decision/terminal nodes. Temperature controls exploration vs exploitation.
   - `expectimax.py`: filtered expectimax (exp021). Chance nodes expanded **full width** over the 21 distinct rolls (doubles 1/36, non-doubles 2/36), opponent replies greedily, one batched forward pass per level. `depth=0` reduces to plain 0-ply. Operates on **gnubg-nn 2x25 boards, not OpenSpiel states** — two slot-orientation conventions are easy to get backwards and are pinned by tests. Design study: `docs/search.qmd`.
     - **Never compare a searched value with an unsearched one.** `search_values` returns `SearchResult(values, searched, evaluated)`; take the argmax over `searched` only. Searching a move takes the opponent's best reply — a minimum over ~20 noisy estimates — so it marks the move down ~0.005 equity, while a pruned move keeps its unmarked static value and can overtake a searched move it was behind. Pruned entries carry their static value so every candidate has a number to *report*, never so it can be *chosen*. Letting them compete cost 0.054–0.176 PR (exp023). The same trap in general form: filtering on one estimator and reporting another.

4. **`raccoon/train/`** — move selection and three training routes
   - `lookahead.py`: **0-ply value lookahead — this is what picks moves.** Enumerate legal moves, evaluate V on each pre-roll child, negate when the child is the opponent's to move, rank. "0-ply" is GNUBG's numbering (static eval, no further search), so a net playing this way is directly comparable to `gnubg` at ply 0; TD-Gammon's papers call this "1-ply". The perspective/negation logic is subtle and lives here once — TD self-play, policy synthesis, and the browser port all reuse it.
   - `self_play.py` / `replay_buffer.py` / `coach.py`: the AlphaZero loop. Plays games recording (observation, MCTS policy, outcome); circular buffer of recent positions; SGD + checkpointing. `coach.py` logs full config (architecture, hyperparams, system info) to JSONL.
   - `td_selfplay.py`: TD(λ) self-play (exp010). Plays by 0-ply lookahead with dice supplying exploration, regresses the value head toward forward-view TD(λ) targets. No policy head, no tree. Loop lives in `scripts/train_td.py`.
   - `parallel_self_play.py`, `inference_server.py`: throughput plumbing for the above.

5. **`raccoon/eval/`** — Evaluation infrastructure
   - `arena.py`: Checkpoint vs checkpoint matches
   - `gnubg_harness.py`: Automated cubeless money game matches against GNUBG's evaluation engine (the `gnubg-nn` library — GNUBG's real nets, programmatically queryable; default `level=world` → full-width 2-ply)
   - `vr_arena.py`: the same net-vs-GNUBG games with a variance-reduced ppg estimator
   - `luck.py`: dice-luck control variate (XG/GNUBG/BGSage style) that `vr_arena` builds on
   - `doubles.py`: what Raccoon's two-step doubles execution costs in play (exp020)
   - `gnubg_adapter.py`, `game_log.py`, `match_log.py`: gnubg-nn bridging and match/game recording

6. **`raccoon/protocol/rgp.py`** — Raccoon Game Protocol: text-based stdin/stdout protocol inspired by UCI for future GUI frontends

7. **`raccoon/cli/play.py`** — Terminal interface for human vs Raccoon play

### Key Design Details

- Board encoding is always from the **current player's perspective** (perspective flip applied in the wrapper)
- Handcrafted channels are **normalised by default**. Raw pip (~95) and contact (~52) are ~100x the scale of the base planes, which lets them dominate the input convolution and destabilises value-head training (Stage 6 of `docs/pretraining_analysis.qmd`). `FEATURE_SCALES` holds the divisors; `normalize=False` only for feature-math tests.
- MCTS never evaluates the network at chance nodes — it samples dice and advances to the next decision node
- The loss depends on which route you are running, and they are not interchangeable:
  - **AlphaZero self-play** (`scripts/train.py`): cross-entropy(policy) + MSE(value) + L2 (via optimizer weight_decay). Value target blends the terminal outcome with MCTS root Q — `--value-bootstrap-alpha` controls the mix (1.0 = pure outcome, 0.0 = pure Q).
  - **TD(λ)** (`scripts/train_td.py`): value head only, regressed toward forward-view TD(λ) targets.
  - **Distillation** (`scripts/train_distill.py`): value head only. `scalar` arm minimises MSE against equity/3; `outcomes6` arm minimises cross-entropy against the six-outcome distribution. One arm per invocation so the A/B isolates the target definition.

## Experiment Conventions

Keep experiments clean and their conclusions buildable. (Hard-won: an earlier 0-ply-distillation arc fragmented one hypothesis across four confusingly-named experiments with mixed metrics/n's and a biased proxy — days of confusion.)

- **One hypothesis per experiment**, stated up front as a single testable question (e.g. *"does TD self-play improve a net already at near-parity with GNUBG-0-ply?"*). A required benchmark is that experiment's *completion*, not a separate experiment.
- **One primary metric, fixed before the run** — with its full protocol: opponent/benchmark, ply/tier, **n**, CI where applicable (e.g. *"cubeless-money ppg vs GNUBG-0-ply, n=6000, ±0.046"* or *"PR vs BGSage money benchmark, rollout tier, n=14693"*). Report every checkpoint on it; don't switch metric or n mid-stream.
- **The primary metric must be unbiased, or scored against a validated high-quality static benchmark.** Two accepted families:
  - **Raw play results** against a real opponent (e.g. GNUBG ppg). Near-parity comparisons need **n≥6000** (n=400 → ±0.18, too coarse); **select *and* confirm** a "best" at full n (a noisy selection inflates it — winner's curse).
  - **PR or MSE/R² against the BGSage money benchmark** (`scripts/eval_benchmark_pr.py`, `data/bgsage/money_benchmark/`) — 14,693 checker decisions with cubeless-equity references at three quality tiers (rollout > 3T > 3P). PR = mean move-selection error × 500 over all tiers (n=14,693); value-head MSE/R² is reported on the **rollout** tier (n≈149k candidate positions — the strongest reference) with 3T/3P as supporting cross-checks. Discovered in exp015: static benchmarking this way gives far more power than raw ppg at zero play cost, so prefer it for near-parity comparisons whenever a checkpoint can be scored against it.
  - Any other proxy metric is supporting-only, clearly labeled, and used only if unbiased (or the bias quantified).
- **Exception: a matched-arm fit-quality proxy, when neither raw play nor the BGSage benchmark is feasible** (e.g. comparing against a bespoke target the benchmark can't isolate). A held-out R² of each arm against its own training target, on an identical split, can stand in as the primary metric — state this explicitly as a deliberate exception, not silently. Valid only when both arms train/evaluate on identical positions and an identical split, so any measurement bias (e.g. holdout leakage) hits both symmetrically and cancels in the comparison. Gives the **sign** of a strength difference, not its ppg magnitude (exp014).
- **Every reported number carries its provenance — (checkpoint, metric, n).** No bare numbers.
- **When the hypothesized effect is small on a raw-ppg metric, report power, not just the CI.** A fixed n/CI calibrated for one comparison (checkpoint vs. teacher) understates its own noise when reused for a smaller one (checkpoint vs. checkpoint has √2× the SE); before calling a result a clean null, check whether n was actually large enough to detect the realistic effect size (needed n for 80% power ≈ `2·(1.96+0.84)²·σ²/δ²`) — "no significant difference" against an underpowered test is much weaker evidence than it sounds. (BGSage PR/MSE at n=14,693/~149k is rarely the bottleneck, but still state n.)
- **One experiment = one name = one write-up**, ending in a one-line conclusion: *"[hypothesis] → [answer]: [net] = [X ± CI] (metric, n)."* A re-run that completes a truncated attempt **supersedes** it (don't co-list both).
- **Supporting metrics are welcome, just demarcated** (training loss/SSE, coarse in-loop evals, epoch curves, secondary findings) — they never drive the headline conclusion or final selection.

## Hardware

- **Local dev (Windows work PC)**: Windows PC (WSL2, Intel i7-1365U, 6 cores / 12 threads, 16 GB RAM, no GPU). Defaults are tuned small: 6 ResNet blocks, 128 channels, 100 MCTS simulations.
- **Local dev (Windows home PC)**: Windows PC (WSL2, Intel i7-8550U, 4 cores / 8 threads, 8 GB RAM, no GPU). Same small defaults; smoke runs only.
- **Local dev (iMac)**: 2013 iMac (`lasse-iMac14-2`), Ubuntu 24.04, Intel i5-4570 (4 cores), 16 GB RAM, GT 755M (no CUDA — CPU only). Same small defaults as the Windows box; expect smoke runs only.
- **Cloud training**: GCP spot VM with T4 GPU (`raccoon-gpu` in `europe-west1-b`). Auto-detects CUDA. See `docs/gcp_guide.md` for workflow.

## Key Files

- `goal.md` — Project goal, assumptions, requirements, and success criteria
- `README.md` — Design decisions, tech stack, and references
- `docs/plan.md` — Full implementation plan with per-milestone specs, interfaces, and test requirements
- `docs/gcp_guide.md` — GCP training workflow, commands, costs, and troubleshooting
- `data/README.md` — Layout of shared datasets (`data/wildbg/`, `data/bglab/`, `data/distill/`); `data/distill/` holds the GNUBG self-play distillation caches organized by label ply then generation run, with GCS provenance
- `experiments/<name>/` — All experiment output lives here. Same layout on VM, in GCS (`gs://raccoon-training-lhm/experiments/`), and locally; `data/distill/` mirrors the same way under `gs://raccoon-training-lhm/data/distill/`. Subdirectories split by **who writes the file**:
  - `logs/` — **committed.** What a running process emitted: append-only jsonl keyed by iteration/epoch (`training_log.jsonl`, `pretrain_log.jsonl`, `eval_log.jsonl`, `td_log.jsonl`, …) plus raw stdout captures. Written by `coach.py`, `pretrain.py`, `train_td.py`, `train_distill.py`, `evaluate.py`, `eval_gnubg.py`.
  - `results/` — **committed.** What an offline scoring pass concluded: idempotent artifacts keyed by engine/checkpoint, rewritten on re-score. Written by `scripts/eval_benchmark_pr.py --output` (BGSage benchmark). A stream goes in `logs/`, a conclusion in `results/` — a scoring run writes both.
  - `checkpoints/`, `caches/`, `dumps/`, `scratch/` — gitignored (weights, intermediate npz); back these up to GCS instead.

  `logs/` and `results/` are whitelisted in `.gitignore` and **must be committed in the same commit as the write-up that uses them**: `docs/pretraining_analysis.qmd` and `docs/training_analysis.qmd` compute their tables and figures from these files at render time, so a missing one breaks — or silently blanks — the page on every other machine (docs CI hides this, since `_freeze` serves cached output). The whitelist is depth-specific (`!experiments/*/logs/**`, `!experiments/*/*/logs/**`), so a layout nested deeper than a per-round/per-arm subdirectory needs a new negation rule.
