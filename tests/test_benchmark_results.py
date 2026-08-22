"""The results directory must not lose a measurement to a reused engine label.

Result files are named after the engine label alone, so scoring a different
sample under the same label silently replaces the old file. That happened once:
a full-benchmark GNUBG 1-ply number (n=14,693) was destroyed by a 2,000-position
re-score. ``save_results`` now refuses unless the two runs measured the same
thing, or ``--overwrite`` is given.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from eval_benchmark_pr import save_results  # noqa: E402


def make_result(label="engine", n=2000, checkpoint="ckpt/ep22.pt", search=None, pr=1.0):
    r = {"engine_label": label, "pr": pr, "n": n, "blunders": 0, "checkpoint": checkpoint}
    if search is not None:
        r["search"] = search
    return r


def test_rescoring_the_same_measurement_is_idempotent(tmp_path):
    save_results([make_result(pr=1.0)], str(tmp_path))
    # A re-score legitimately produces a slightly different PR (different
    # timings, a re-trained tie-break) but measures the same thing.
    save_results([make_result(pr=1.02)], str(tmp_path))

    saved = json.loads((tmp_path / "engine.json").read_text())
    assert saved["pr"] == 1.02


@pytest.mark.parametrize(
    "changed",
    [
        {"n": 14693},
        {"checkpoint": "ckpt/ep30.pt"},
        {"search": {"depth": 2, "k": 8}},
    ],
)
def test_a_different_measurement_under_the_same_label_is_refused(tmp_path, changed):
    save_results([make_result(search={"depth": 1, "k": 8})], str(tmp_path))
    before = (tmp_path / "engine.json").read_text()

    incoming = {"search": {"depth": 1, "k": 8}, **changed}
    with pytest.raises(SystemExit) as exc:
        save_results([make_result(**incoming)], str(tmp_path))

    field = next(iter(changed))
    assert field in str(exc.value)
    assert "--overwrite" in str(exc.value)
    assert (tmp_path / "engine.json").read_text() == before, "file was modified anyway"


def test_overwrite_flag_replaces_it(tmp_path):
    save_results([make_result()], str(tmp_path))
    save_results([make_result(n=14693)], str(tmp_path), overwrite=True)

    assert json.loads((tmp_path / "engine.json").read_text())["n"] == 14693


def test_timings_alone_do_not_count_as_a_different_measurement(tmp_path):
    cfg = {"depth": 1, "k": 8, "evals_per_decision": 1650.0, "sec_per_decision": 4.25}
    save_results([make_result(search=cfg)], str(tmp_path))

    slower = dict(cfg, evals_per_decision=1650.0, sec_per_decision=9.9)
    with pytest.raises(SystemExit):
        save_results([make_result(search=slower)], str(tmp_path))


def test_a_clash_in_one_file_blocks_the_whole_batch(tmp_path):
    """Otherwise a two-engine run could half-write and leave results inconsistent."""
    save_results([make_result(label="a"), make_result(label="b")], str(tmp_path))
    b_before = (tmp_path / "b.json").read_text()

    with pytest.raises(SystemExit):
        save_results(
            [make_result(label="a", pr=2.0), make_result(label="b", n=14693)],
            str(tmp_path),
        )

    assert json.loads((tmp_path / "a.json").read_text())["pr"] == 1.0
    assert (tmp_path / "b.json").read_text() == b_before


def test_holdout_is_disjoint_from_the_sample_and_covers_the_rest():
    """A rule tuned on the sample must be confirmed on positions it never saw."""
    from eval_benchmark_pr import subsample, subsample_complement

    decisions = [{"i": i} for i in range(500)]
    sample = subsample(decisions, 120, 21)
    holdout = subsample_complement(decisions, 120, 21)

    assert len(sample) == 120
    assert len(holdout) == 380
    ids = {id(d) for d in sample}
    assert not any(id(d) in ids for d in holdout), "holdout overlaps the sample"
    assert len(ids | {id(d) for d in holdout}) == len(decisions), "together they must cover it"
    # Benchmark order is preserved, so shards of the holdout stay representative.
    assert [d["i"] for d in holdout] == sorted(d["i"] for d in holdout)
