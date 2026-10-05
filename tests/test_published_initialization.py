import math

import torch

from ccdn.models.dense_mlp import DenseMLP


def test_published_kaiming_uses_relu_and_linear_output_bounds_with_zero_bias():
    torch.manual_seed(8)
    model = DenseMLP(input_size=20, hidden_sizes=(12, 9, 7), output_size=4,
                     initialization="published_kaiming")
    for layer in model.layers[:-1]:
        bound = math.sqrt(6 / layer.in_features)
        assert float(layer.weight.detach().abs().max()) <= bound
        assert torch.count_nonzero(layer.bias) == 0
    output = model.layers[-1]
    assert float(output.weight.detach().abs().max()) <= math.sqrt(3 / output.in_features)
    assert torch.count_nonzero(output.bias) == 0


def test_default_dense_initialization_remains_the_default_mode():
    torch.manual_seed(19)
    default = DenseMLP(input_size=10, hidden_sizes=(6,), output_size=3)
    torch.manual_seed(19)
    explicit = DenseMLP(input_size=10, hidden_sizes=(6,), output_size=3,
                        initialization="default")
    for first, second in zip(default.parameters(), explicit.parameters()):
        torch.testing.assert_close(first, second)
