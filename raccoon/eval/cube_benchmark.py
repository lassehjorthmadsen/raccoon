"""Scoring cube decisions against the BGSage money benchmark.

``scripts/eval_benchmark_pr.py`` scores checker play on this benchmark and
discards everything else at load time. The same file also carries 2,196
``kind == "cube"`` entries -- positions where a cube decision was live, each with
the three cubeful equities a doubler compares (no double / double-take /
double-pass) and reference double and take actions. That is 2,842 scorable
sub-decisions, since a position can pose the question to the doubler, to the
receiver, or to both.

This module turns a cube policy into a PR figure on those decisions. PR is the
average equity thrown away per decision times 500; lower is better, and 0 means
never choosing wrong. The error formulas are ported from BGSage's own scorer
(``bgsage/scripts/benchmark_money.py: _score_cube``) so the numbers are directly
comparable to the ones BGSage publishes.

Two properties of the references are worth knowing before quoting a number:

* They are **not optimal cubeful equity**. They come from BGSage's cubeful n-ply
  search and cubeful rollouts, which use Janowski at their own leaves. Scoring a
  cube model against them measures the gap that deeper cubeful search closes,
  which is useful and is what we want here, but it is not ground truth.
* **Only a third of them are measurements.** ``eval_level`` says which: entries
  marked ``Rollout`` (700 of 2,196) had ``equity_nd`` produced by 1,296 trials
  played to completion with the cube live, so it is an observed average with a
  standard error. Entries marked ``3-ply`` (1,496) had it produced by a cubeful
  search that applies Janowski at its own leaves, so scoring a Janowski variant
  against them is close to scoring it against itself -- and empirically it shows,
  with the published model scoring PR 0.649 there against 5.406 on the measured
  set. :func:`measured_only` selects the usable subset.
* **Tier is not the same split, and confounds two things.** The benchmark spends
  more compute on closer decisions, so tier mixes difficulty with reference
  provenance. Split on ``eval_level`` for validity, not on ``tier``.
"""

import gzip
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np

# Kept in step with scripts/eval_benchmark_pr.py, which defines the same
# constants for the checker half of this benchmark.
BLUNDER_THRESHOLD = 0.08
PR_MULTIPLIER = 500
TIERS = ["rollout", "3T", "3P"]
GAME_PLANS = ["purerace", "racing", "attacking", "priming", "anchoring"]

DEFAULT_BENCHMARK = str(
    Path(__file__).resolve().parents[2] / "data" / "bgsage"
    / "money_benchmark" / "benchmark.json.gz"
)

# A cube decision is always pre-roll, so these entries carry ``dice: null``.
# Which side of the decision a sub-decision represents:
DOUBLER = "double"
RECEIVER = "take"


@dataclass(frozen=True)
class CubeDecision:
    """One scored sub-decision: which entry it came from, and what it cost."""

    entry_index: int
    role: str          # DOUBLER or RECEIVER
    error: float       # equity thrown away, >= 0


def load_cube_decisions(path: str = DEFAULT_BENCHMARK) -> tuple[list[dict], dict]:
    """Load the ``kind == "cube"`` entries and the benchmark metadata.

    The checker-side loader in ``scripts/eval_benchmark_pr.py`` is the sibling of
    this function; the few lines of gzip/json are duplicated rather than shared
    so that ``raccoon.eval`` does not have to import from ``scripts``.
    """
    with gzip.open(path, "rt") as f:
        data = json.load(f)
    entries = [d for d in data["decisions"] if d["kind"] == "cube"]
    return entries, data["meta"]


def measured_only(entries: Sequence[dict]) -> list[dict]:
    """Just the entries whose reference equities were actually rolled out.

    The rest were produced by a cubeful search that uses Janowski at its leaves,
    which makes them close to circular for judging a Janowski variant. Scoring on
    them understates the model's error roughly eightfold.

    A caveat that survives this filter: the rollout's *payoff* is empirical, but
    the cube decisions made during it come from a 3-ply bot that also bottoms out
    in Janowski. So these measure equity under Janowski-quality cube play rather
    than optimal cube play. Policy error can only lower realised equity and its
    cost is second-order near a decision threshold, so the references are a
    slight understatement of what a better model could reach, not an overstatement.
    """
    return [e for e in entries if e["eval_level"] == "Rollout"]


def split_by_game(entries: Sequence[dict], frac: float = 0.5,
                  seed: int = 0) -> tuple[list[dict], list[dict]]:
    """Split entries into (fit, confirm) halves along *game* boundaries.

    Splitting on positions would leak: several cube decisions can come from the
    same game, and positions within a game are strongly correlated, so a
    parameter tuned on one of them is partly tuned on its siblings. The ``seed``
    field identifies the game a position came from (500 distinct games), so
    splitting on it keeps the two halves genuinely independent.
    """
    games = sorted({e["seed"] for e in entries})
    rng = np.random.default_rng(seed)
    shuffled = list(games)
    rng.shuffle(shuffled)
    n_fit = int(round(frac * len(shuffled)))
    fit_games = set(shuffled[:n_fit])
    fit = [e for e in entries if e["seed"] in fit_games]
    confirm = [e for e in entries if e["seed"] not in fit_games]
    return fit, confirm


def score_cube_decisions(
    entries: Sequence[dict],
    decide: Callable[[dict], tuple[bool, bool]],
) -> list[CubeDecision]:
    """Score a cube policy over the benchmark's cube entries.

    Args:
        entries: cube entries from :func:`load_cube_decisions`.
        decide: maps an entry to ``(should_double, should_take)``.

    Returns:
        One :class:`CubeDecision` per scorable sub-decision, in entry order with
        the doubler's before the receiver's. That ordering is deterministic, so
        two policies scored over the same entries produce aligned lists and can
        be compared decision by decision -- which is what a paired comparison
        needs.

    Two of the 2,196 entries are beavers, where ``equity_dt`` holds the beaver
    equity rather than the plain take equity. BGSage's scorer does not special-
    case them and neither does this one, so the figures stay comparable.
    """
    out: list[CubeDecision] = []
    for i, entry in enumerate(entries):
        nd, dt, dp = entry["equity_nd"], entry["equity_dt"], entry["equity_dp"]
        should_double, should_take = decide(entry)

        if entry.get("has_double"):
            # Doubling is worth min(dt, dp) because the opponent picks the reply.
            optimal = max(nd, min(dt, dp))
            actual = min(dt, dp) if should_double else nd
            out.append(CubeDecision(i, DOUBLER, max(0.0, optimal - actual)))

        if entry.get("has_take"):
            # Equities are the doubler's, so the receiver is minimising.
            optimal = min(dt, dp)
            actual = dt if should_take else dp
            out.append(CubeDecision(i, RECEIVER, max(0.0, actual - optimal)))
    return out


def error_array(decisions: Iterable[CubeDecision]) -> np.ndarray:
    """The per-sub-decision errors, in scoring order."""
    return np.array([d.error for d in decisions], dtype=np.float64)


def _pr_stats(errors: np.ndarray) -> dict:
    """PR with its 95% interval, plus the blunder rate."""
    if errors.size == 0:
        return {"n": 0, "pr": float("nan"), "pr_ci95": float("nan"),
                "blunder_rate": float("nan")}
    se = errors.std(ddof=1) / np.sqrt(errors.size) if errors.size > 1 else 0.0
    return {
        "n": int(errors.size),
        "pr": float(PR_MULTIPLIER * errors.mean()),
        "pr_ci95": float(PR_MULTIPLIER * 1.96 * se),
        "blunder_rate": float((errors > BLUNDER_THRESHOLD).mean()),
    }


def aggregate(decisions: Sequence[CubeDecision],
              entries: Sequence[dict]) -> dict:
    """Headline PR plus the breakdowns, matching eval_benchmark_pr.py's shape.

    Per-tier PR is reported but must not be read as a quality ladder: the
    benchmark assigns tiers by how close a decision is, so the tiers differ in
    difficulty and in what an error costs.
    """
    errors = error_array(decisions)
    result = _pr_stats(errors)

    def subset(pred) -> dict:
        sel = np.array([pred(entries[d.entry_index], d) for d in decisions],
                       dtype=bool)
        return _pr_stats(errors[sel])

    result["by_role"] = {
        role: subset(lambda e, d, r=role: d.role == r)
        for role in (DOUBLER, RECEIVER)
    }
    result["by_tier"] = {
        tier: subset(lambda e, d, t=tier: e["tier"] == t) for tier in TIERS
    }
    result["by_cube_owner"] = {
        owner: subset(lambda e, d, o=owner: e["cube_owner"] == o)
        for owner in ("centered", "player")
    }
    result["by_game_plan"] = {
        plan: subset(lambda e, d, p=plan: e.get("game_plan") == p)
        for plan in GAME_PLANS
    }
    return result


def paired_delta(baseline: Sequence[CubeDecision],
                 variant: Sequence[CubeDecision]) -> dict:
    """Paired PR difference (baseline - variant), positive meaning the variant is better.

    Both lists must come from :func:`score_cube_decisions` over the *same*
    entries, so element i is the same sub-decision in both. Pairing removes the
    position-to-position variation that dominates an unpaired comparison, and on
    this benchmark it is the difference between a usable interval and a useless
    one.
    """
    a, b = error_array(baseline), error_array(variant)
    if a.shape != b.shape:
        raise ValueError(
            f"cannot pair {a.shape} decisions against {b.shape}: "
            "both variants must be scored over the same entries")
    diff = PR_MULTIPLIER * (a - b)
    se = diff.std(ddof=1) / np.sqrt(diff.size) if diff.size > 1 else 0.0
    return {
        "n": int(diff.size),
        "delta_pr": float(diff.mean()),
        "delta_ci95": float(1.96 * se),
        "decisions_changed": int((a != b).sum()),
    }


# ---------------------------------------------------------------------------
# Internal consistency
# ---------------------------------------------------------------------------
#
# For one and the same position, owning the cube cannot be worth less than
# having it centred, which in turn cannot be worth less than the opponent
# owning it: E_O >= E_C >= E_U. Nothing in the fitting enforces this, because
# each benchmark entry is *either* centred *or* player-owned, so a model whose
# parameters are fitted per cube state is fitted on disjoint position sets and
# is free to come out incoherent. PR will not notice — every decision it scores
# involves only one cube location at a time.

# A small violation rate is expected and is not a fitting failure. Under the
# Jacoby rule the centred dead-cube term is 2p-1 while the owned and opponent
# terms use the full cubeless equity, so wherever gammons are lopsided that
# asymmetry alone can invert a pair -- in either direction. Measured floor for
# single-index models: 3.5% at the published x=0.68, 2.9% at x=0.775.
#
# That floor is worth noticing rather than waving away: the shipped model is
# mildly incoherent about the centred cube even at its own published constant,
# which is one more symptom of the centred equity being the weak component.
JACOBY_ORDERING_FLOOR = 0.035

# The limit sits well clear of that floor, so it catches structural incoherence
# -- a model whose parameters were fitted per cube state and never had to agree
# -- rather than the Jacoby artifact. Fitting per state measures 38.9%.
ORDERING_LIMIT = 0.10


def ordering_violations(entries: Sequence[dict], equities_for) -> dict:
    """How often a cube model breaks E_O >= E_C >= E_U.

    Args:
        entries: cube entries.
        equities_for: maps an entry to ``(E_O, E_C, E_U)`` — the same position
            valued at all three cube locations, which is the comparison the
            benchmark itself never makes.

    Returns:
        Violation counts and rates, plus ``coherent``: False when the rate
        exceeds :data:`ORDERING_LIMIT`. A model that fails this should be
        disqualified rather than ranked, however good its PR looks — a lower
        error on decisions it can score does not license an ordering that cannot
        happen.
    """
    n = len(entries)
    bad_oc = bad_cu = 0
    worst = 0.0
    for entry in entries:
        e_o, e_c, e_u = equities_for(entry)
        if e_c > e_o + 1e-12:
            bad_oc += 1
            worst = max(worst, e_c - e_o)
        if e_u > e_c + 1e-12:
            bad_cu += 1
    rate = (bad_oc + bad_cu) / n if n else 0.0
    return {
        "n": n,
        "centred_above_owned": bad_oc,
        "opponent_above_centred": bad_cu,
        "violation_rate": rate,
        "worst_overshoot": worst,
        "coherent": rate <= ORDERING_LIMIT,
        "limit": ORDERING_LIMIT,
        "jacoby_floor": JACOBY_ORDERING_FLOOR,
    }
