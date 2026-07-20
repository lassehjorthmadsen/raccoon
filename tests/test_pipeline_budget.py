"""Tests for scripts/pipeline_budget.py — cost projection + round-boundary stop."""
import json
import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "pipeline_budget.py"

CONTINUE = 0
STOP = 10


def run(*args) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *map(str, args)],
        capture_output=True, text=True,
    )


def record_round(state, seconds):
    assert run("record-round", "--state", state, "--seconds", seconds).returncode == 0


def record_eval(state, rnd, equity):
    assert run("record-eval", "--state", state,
               "--round", rnd, "--equity", equity).returncode == 0


def check(state, completed, total, **kw):
    args = ["check", "--state", state, "--completed", completed, "--total", total]
    for k, v in kw.items():
        args += [f"--{k.replace('_', '-')}", v]
    return run(*args)


def test_record_round_accumulates(tmp_path):
    record_round(tmp_path, 3600)      # 1.0 h
    record_round(tmp_path, 1800)      # 0.5 h
    data = json.loads((tmp_path / "budget.json").read_text())
    assert data["consumed_hours"] == 1.5
    assert data["timed_rounds"] == 2


def test_projection_uses_measured_average(tmp_path):
    # two rounds at 2 h each -> avg 2 h; 3 of 5 rounds left -> 4 + 2*3 = 10 h
    record_round(tmp_path, 7200)
    record_round(tmp_path, 7200)
    r = check(tmp_path, completed=2, total=5, rate=5.0, max_budget=0)
    assert r.returncode == CONTINUE
    assert "projecting 10.0h" in r.stdout
    assert "kr.50" in r.stdout          # 10 h * kr.5/h


def test_preflight_uses_calibration_before_any_round(tmp_path):
    # no timed rounds yet -> fall back to --calib-hours-per-round for the estimate
    r = check(tmp_path, completed=0, total=10, rate=5.0,
              calib_hours_per_round=9.0, max_budget=0)
    assert r.returncode == CONTINUE
    assert "projecting 90.0h" in r.stdout


def test_budget_cap_stops_before_next_round(tmp_path):
    # consumed 90 h (kr.450 at kr.5/h); avg 45 h -> next round would hit kr.675
    record_round(tmp_path, 45 * 3600)
    record_round(tmp_path, 45 * 3600)
    over = check(tmp_path, completed=2, total=10, rate=5.0, max_budget=600)
    assert over.returncode == STOP
    assert "STOP" in over.stdout and "max-budget" in over.stdout

    under = check(tmp_path, completed=2, total=10, rate=5.0, max_budget=2000)
    assert under.returncode == CONTINUE


def test_projected_over_budget_warns_but_does_not_stop(tmp_path):
    # one 9 h round, 25 total -> projects ~225 h = kr.1125, over an 800 cap, but
    # consumed is tiny so it must WARN and continue, not stop.
    record_round(tmp_path, 9 * 3600)
    r = check(tmp_path, completed=1, total=25, rate=5.0, max_budget=800)
    assert r.returncode == CONTINUE
    assert "NOTE" in r.stdout and "exceeds" in r.stdout


def test_wall_hours_cap(tmp_path):
    record_round(tmp_path, 10 * 3600)
    assert check(tmp_path, completed=1, total=5, max_wall=15).returncode == STOP
    assert check(tmp_path, completed=1, total=5, max_wall=100).returncode == CONTINUE


def test_plateau_stops_when_no_improvement(tmp_path):
    # best is round 6 (-0.37); next four evals never beat it -> plateau
    for rnd, eq in [(2, -0.69), (4, -0.62), (6, -0.37),
                    (8, -0.52), (10, -0.65), (14, -0.81), (18, -0.50)]:
        record_eval(tmp_path, rnd, eq)
    record_round(tmp_path, 3600)       # need some consumed time for the check
    r = check(tmp_path, completed=1, total=25, patience=4, min_delta=0.05)
    assert r.returncode == STOP
    assert "Plateau" in r.stdout


def test_plateau_not_triggered_while_improving(tmp_path):
    for rnd, eq in [(2, -0.69), (4, -0.62), (6, -0.50), (8, -0.37)]:
        record_eval(tmp_path, rnd, eq)
    record_round(tmp_path, 3600)
    r = check(tmp_path, completed=1, total=25, patience=3, min_delta=0.05)
    assert r.returncode == CONTINUE


def test_plateau_needs_baseline_before_window(tmp_path):
    # exactly `patience` evals and no earlier baseline -> cannot declare plateau
    for rnd, eq in [(2, -0.5), (4, -0.5), (6, -0.5)]:
        record_eval(tmp_path, rnd, eq)
    record_round(tmp_path, 3600)
    assert check(tmp_path, completed=1, total=25, patience=3).returncode == CONTINUE


def test_disabled_caps_never_stop(tmp_path):
    record_round(tmp_path, 1000 * 3600)      # absurd consumption
    for rnd, eq in [(2, -0.5), (4, -0.6), (6, -0.7), (8, -0.8)]:
        record_eval(tmp_path, rnd, eq)
    # all caps default to 0/off -> always continue
    assert check(tmp_path, completed=4, total=25).returncode == CONTINUE
