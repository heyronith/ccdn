from ccdn.experiments.online_permuted_mnist import summarize_online


def _rows(values):
    return [{"task_index": i, "online_accuracy": value,
             "optimizer_updates": 60000} for i, value in enumerate(values)]


def test_preregistered_gate_requires_drop_early_peak_and_sustained_decline():
    values = [0.95 - i * 0.0005 for i in range(150)]
    result = summarize_online(_rows(values), [], 150, 1.0, False)
    assert result["peak_20_task_accuracy"] > result["final_20_task_accuracy"]
    assert result["plasticity_drop"] >= 0.05
    assert result["peak_window_end"] < 130
    assert result["sustained_downward_trend"] is True
    assert result["loss_of_plasticity_gate_passed"] is True


def test_plateau_does_not_pass_gate():
    result = summarize_online(_rows([0.9] * 150), [], 150, 1.0, False)
    assert abs(result["plasticity_drop"]) < 1e-12
    assert result["loss_of_plasticity_gate_passed"] is False
