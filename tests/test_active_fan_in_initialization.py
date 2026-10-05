import math

import torch

from ccdn.models.sparse_mlp import SparseMLP


def _model(seed):
    torch.manual_seed(seed)
    return SparseMLP(input_size=24, hidden_sizes=(12, 8), output_size=4,
                     density=0.3, seed=seed,
                     initialization="active_fan_in_kaiming")


def test_active_fan_in_initialization_preserves_mask_and_zeros_inactive_weights():
    model = _model(7)
    for layer in model.layers:
        assert torch.count_nonzero(layer.weight[~layer.mask]) == 0
        assert torch.count_nonzero(layer.bias) == 0
    model_again = _model(7)
    for first, second in zip(model.layers, model_again.layers):
        torch.testing.assert_close(first.mask, second.mask)
        torch.testing.assert_close(first.weight, second.weight)
        torch.testing.assert_close(first.bias, second.bias)


def test_active_fan_in_scaling_uses_each_rows_realized_fan_in():
    model = _model(11)
    for layer_index, layer in enumerate(model.layers):
        is_output = layer_index == len(model.layers) - 1
        gain = 1.0 if is_output else math.sqrt(2)
        for row in range(layer.out_features):
            active = layer.mask[row]
            fan_in = int(active.sum())
            if fan_in:
                expected_bound = math.sqrt(3) * gain / math.sqrt(fan_in)
                assert float(layer.weight[row, active].detach().abs().max()) <= expected_bound
            assert torch.count_nonzero(layer.weight[row, ~active]) == 0
