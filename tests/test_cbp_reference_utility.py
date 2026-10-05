import torch

from ccdn.models.dense_mlp import DenseMLP
from ccdn.official_reference.continual_backprop import ContinualBackpropReference


def test_adaptable_contribution_utility_and_bias_correction():
    model = DenseMLP(2, (2,), 1)
    alg = ContinualBackpropReference(model, decay_rate=0.5,
                                     maturity_threshold=99)
    with torch.no_grad():
        model.layers[0].weight.copy_(torch.tensor([[1., 2.], [2., 1.]]))
        model.layers[1].weight.copy_(torch.tensor([[2., 4.]]))
    features = torch.tensor([[1., 0.], [0., 2.]])
    alg.age[0].fill_(1)
    alg._update_utility(0, features)
    corrected_mean = features.mean(0)
    in_mag = model.layers[0].weight.abs().mean(1)
    out_mag = model.layers[1].weight.abs().mean(0)
    expected = 0.5 * out_mag * (features - corrected_mean).abs().mean(0) / in_mag
    torch.testing.assert_close(alg.utility[0], expected)
    torch.testing.assert_close(alg.bias_corrected_utility[0], expected / 0.5)
    torch.testing.assert_close(alg.mean_activation[0], corrected_mean * 0.5)
