from scripts.cycle2d_calibration import calibration_outcome, evaluate_calibration_horizon
from scripts.bootstrap_cycle2d import HORIZONS


def rows_from(values):
    return [{"task_index": index, "online_accuracy": value}
            for index, value in enumerate(values)]


def passing_curve(horizon=200, high=.85, low=.80):
    values = [high] * horizon
    decline_start = horizon - 50
    decline_end = horizon - 20
    for task in range(decline_start, decline_end):
        values[task] = high - (high - low) * (task - decline_start) / (decline_end - decline_start)
    values[decline_end:] = [low] * 20
    return values


def test_drop_below_threshold_fails():
    result = evaluate_calibration_horizon(rows_from(passing_curve(low=.8001)), 200)
    assert result["drop"] < .05
    assert result["gate_passed"] is False


def test_drop_exactly_point_zero_five_passes_without_rounding():
    values = passing_curve(high=.051, low=.001)
    result = evaluate_calibration_horizon(rows_from(values), 200)
    assert result["drop"] == .05
    assert result["gate_passed"] is True


def test_peak_inside_final_twenty_fails():
    result = evaluate_calibration_horizon(rows_from([.7] * 180 + [.9] * 20), 200)
    assert result["peak_window_end"] >= 180
    assert result["peak_before_final20"] is False
    assert result["gate_passed"] is False


def test_nonnegative_post_peak_slope_fails_independently():
    values = [.9] * 20 + [.65] * 130
    values += [.85 - .05 * (task - 150) / 30 for task in range(150, 180)] + [.8] * 20
    result = evaluate_calibration_horizon(rows_from(values), 200)
    assert result["drop"] >= .05
    assert result["final50_ols_slope"] < 0
    assert result["post_peak_ols_slope"] >= 0
    assert result["gate_passed"] is False


def test_nonnegative_final_fifty_slope_fails_independently():
    values = [.9] * 20
    values += [.9 - .18 * (task - 20) / 130 for task in range(20, 150)]
    values += [.72 + .08 * (task - 150) / 49 for task in range(150, 200)]
    result = evaluate_calibration_horizon(rows_from(values), 200)
    assert result["drop"] >= .05
    assert result["post_peak_ols_slope"] < 0
    assert result["final50_ols_slope"] >= 0
    assert result["gate_passed"] is False


def test_incomplete_horizon_is_not_evaluated_as_pass():
    result = evaluate_calibration_horizon(rows_from(passing_curve()[:-1]), 200)
    assert result["exact_horizon_complete"] is False
    assert result["gate_passed"] is False


def test_only_tasks_before_horizon_are_used():
    curve = rows_from(passing_curve())
    curve.extend({"task_index": index, "online_accuracy": 0.0}
                 for index in range(200, 210))
    result = evaluate_calibration_horizon(curve, 200)
    assert result["tasks_present"] == 200


def test_earliest_persistent_pair_is_selected():
    values = [
        {"horizon": h, "gate_passed": passed}
        for h, passed in zip(HORIZONS, [True, True, True, False, True, True])
    ]
    assert calibration_outcome(values) == {
        "state": "CONFIRMED_FAILURE_HORIZON", "selected_horizon": 200,
        "confirmation_horizon": 250}
    later_pair = [
        {"horizon": h, "gate_passed": passed}
        for h, passed in zip(HORIZONS, [False, True, True, False, True, True])
    ]
    assert calibration_outcome(later_pair)["selected_horizon"] == 250
    four_six = [
        {"horizon": h, "gate_passed": passed}
        for h, passed in zip(HORIZONS, [False, False, False, True, True, False])
    ]
    assert calibration_outcome(four_six)["selected_horizon"] == 400


def test_isolated_pass_followed_by_failure_is_not_confirmed():
    values = [{"horizon": h, "gate_passed": passed}
              for h, passed in zip(HORIZONS, [False, False, True, False, False, False])]
    assert calibration_outcome(values)["state"] == "NO_CONFIRMED_FAILURE_THROUGH_800"


def test_800_only_pass_is_unconfirmed():
    values = [{"horizon": h, "gate_passed": h == 800} for h in HORIZONS]
    assert calibration_outcome(values) == {
        "state": "UNCONFIRMED_PASS_AT_800", "selected_horizon": None,
        "confirmation_horizon": None}
