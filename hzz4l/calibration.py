"""Weighted calibration metrics (same definitions as in the notebook)."""
import numpy as np


def weighted_ece(y_true, p, weights, n_bins=10):
    """Expected calibration error with per-event weights, equal-width probability bins."""
    y_true, p, weights = map(np.asarray, (y_true, p, weights))
    edges = np.linspace(0, 1, n_bins + 1)
    bin_idx = np.clip(np.digitize(p, edges) - 1, 0, n_bins - 1)
    total = weights.sum()
    ece = 0.0
    for b in range(n_bins):
        m = bin_idx == b
        wb = weights[m].sum()
        if wb == 0:
            continue
        conf = np.average(p[m], weights=weights[m])
        acc = np.average(y_true[m], weights=weights[m])
        ece += wb / total * abs(acc - conf)
    return ece


def brier_skill_score(y_true, p, weights, base_rate):
    """Brier skill score against the constant base-rate predictor."""
    y_true, p, weights = map(np.asarray, (y_true, p, weights))
    b = np.average((p - y_true) ** 2, weights=weights)
    b0 = np.average((base_rate - y_true) ** 2, weights=weights)
    return 1 - b / b0
