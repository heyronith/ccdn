"""Preregistered horizon evaluation and persistence rule for Cycle 2D."""
from __future__ import annotations

import math

import numpy as np

from scripts.bootstrap_cycle2d import HORIZONS


def _ols_slope(values, start, end):
    if start < 0 or end >= len(values) or end <= start:
        return None
    x = np.arange(start, end + 1, dtype=np.float64)
    y = np.asarray(values[start:end + 1], dtype=np.float64)
    x_centered = x - x.mean()
    return float(np.dot(x_centered, y - y.mean()) / np.dot(x_centered, x_centered))


def evaluate_calibration_horizon(task_rows, horizon):
    """Evaluate one fixed horizon, using accuracy values only from tasks [0,H)."""
    horizon = int(horizon)
    scoped = sorted((row for row in task_rows if int(row["task_index"]) < horizon),
                    key=lambda row: int(row["task_index"]))
    indices = [int(row["task_index"]) for row in scoped]
    complete = len(scoped) == horizon and indices == list(range(horizon))
    result = {
        "horizon": horizon,
        "tasks_present": len(scoped),
        "exact_horizon_complete": complete,
        "peak20": None,
        "peak_window_start": None,
        "peak_window_end": None,
        "final20": None,
        "drop": None,
        "post_peak_ols_slope": None,
        "final50_ols_slope": None,
        "peak_before_final20": False,
        "gate_passed": False,
    }
    if not complete or horizon < 50:
        return result
    accuracies = np.asarray([float(row["online_accuracy"]) for row in scoped], dtype=np.float64)
    if not np.isfinite(accuracies).all() or np.any(accuracies < 0) or np.any(accuracies > 1):
        return result
    rolling = np.convolve(accuracies, np.ones(20, dtype=np.float64) / 20.0, mode="valid")
    peak_start = int(np.argmax(rolling))
    peak_end = peak_start + 19
    peak20 = float(rolling[peak_start])
    final20 = float(accuracies[horizon - 20:horizon].mean())
    drop = peak20 - final20
    post_peak = _ols_slope(accuracies, peak_end, horizon - 1)
    final50 = _ols_slope(accuracies, horizon - 50, horizon - 1)
    peak_before = peak_end < horizon - 20
    passed = (drop >= 0.05 and peak_before and post_peak is not None and post_peak < 0
              and final50 is not None and final50 < 0)
    result.update({
        "peak20": peak20, "peak_window_start": peak_start,
        "peak_window_end": peak_end, "final20": final20, "drop": drop,
        "post_peak_ols_slope": post_peak, "final50_ols_slope": final50,
        "peak_before_final20": peak_before, "gate_passed": bool(passed),
    })
    return result


def calibration_outcome(evaluations, horizons=HORIZONS):
    """Apply the preregistered adjacent-pass confirmation rule to completed horizons."""
    by_horizon = {int(row["horizon"]): row for row in evaluations}
    for previous, following in zip(horizons[:-1], horizons[1:]):
        if (by_horizon.get(previous, {}).get("gate_passed") is True
                and by_horizon.get(following, {}).get("gate_passed") is True):
            return {"state": "CONFIRMED_FAILURE_HORIZON", "selected_horizon": previous,
                    "confirmation_horizon": following}
    last = max(by_horizon, default=0)
    if last == horizons[-1]:
        if by_horizon[last].get("gate_passed") is True:
            return {"state": "UNCONFIRMED_PASS_AT_800", "selected_horizon": None,
                    "confirmation_horizon": None}
        return {"state": "NO_CONFIRMED_FAILURE_THROUGH_800", "selected_horizon": None,
                "confirmation_horizon": None}
    return {"state": "IN_PROGRESS", "selected_horizon": None,
            "confirmation_horizon": None}
