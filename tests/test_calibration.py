import numpy as np
import pytest

from hzz4l.calibration import brier_skill_score, weighted_ece


def test_ece_near_zero_for_calibrated_scores():
    rng = np.random.default_rng(0)
    p = rng.uniform(0, 1, 200_000)
    y = (rng.uniform(0, 1, p.size) < p).astype(int)
    assert weighted_ece(y, p, np.ones_like(p)) < 0.01


def test_ece_detects_overconfidence():
    rng = np.random.default_rng(1)
    p = np.full(50_000, 0.9)
    y = (rng.uniform(0, 1, p.size) < 0.5).astype(int)      # true rate 0.5, claimed 0.9
    assert weighted_ece(y, p, np.ones_like(p)) == pytest.approx(0.4, abs=0.01)


def test_ece_respects_weights():
    # Same predictions, but the badly calibrated half carries ~zero weight -> ECE ~ 0.
    p = np.concatenate([np.full(1000, 0.2), np.full(1000, 0.9)])
    y = np.concatenate([np.zeros(1000), np.zeros(1000)])   # second half is badly wrong
    w = np.concatenate([np.ones(1000), np.full(1000, 1e-9)])
    assert weighted_ece(y, p, w) == pytest.approx(0.2, abs=1e-3)


def test_brier_skill_score_constant_predictor_is_zero():
    rng = np.random.default_rng(2)
    y = (rng.uniform(0, 1, 10_000) < 0.3).astype(int)
    base = y.mean()
    assert brier_skill_score(y, np.full(y.size, base), np.ones(y.size), base) == pytest.approx(0.0, abs=1e-12)


def test_brier_skill_score_perfect_predictor_is_one():
    y = np.array([0, 1, 1, 0, 1])
    assert brier_skill_score(y, y.astype(float), np.ones(5), 0.5) == pytest.approx(1.0)
