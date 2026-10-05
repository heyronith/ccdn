import torch

from ccdn.metrics.plasticity import online_plasticity_diagnostics
from ccdn.models.dense_mlp import DenseMLP


def test_dead_units_weight_magnitude_and_effective_rank_are_diagnostic_only():
    torch.manual_seed(25)
    model = DenseMLP(input_size=4, hidden_sizes=(3, 3, 3), output_size=2,
                     initialization="published_kaiming")
    with torch.no_grad():
        model.layers[0].weight[0].zero_()
        model.layers[0].bias[0] = -1.0
        model.layers[0].weight[1].zero_()
        model.layers[0].bias[1] = 0.0
    before = {key: value.clone() for key, value in model.state_dict().items()}
    x = torch.rand(30, 4)
    result = online_plasticity_diagnostics(model, x, batch_size=7)
    assert result["dead_unit_fraction_layer_0"] == 2 / 3
    assert result["dead_unit_fraction_overall"] >= 2 / 9
    expected_weight_mean = sum(layer.weight.detach().abs().sum()
                               for layer in model.layers) / sum(
                                   layer.weight.numel() for layer in model.layers)
    assert result["mean_absolute_weight"] == float(expected_weight_mean)
    assert all(result[f"effective_rank_layer_{i}"] >= 0 for i in range(3))
    for key, value in model.state_dict().items():
        assert torch.equal(value, before[key])
