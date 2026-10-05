import torch

from ccdn.models.dense_mlp import DenseMLP
from ccdn.official_reference.continual_backprop import ContinualBackpropReference


def test_fractional_replacement_reinitializes_and_resets_unit_state():
    torch.manual_seed(21)
    model = DenseMLP(5, (4,), 3)
    alg = ContinualBackpropReference(model, replacement_rate=0.1,
                                     decay_rate=0.9, maturity_threshold=0,
                                     accumulate=True)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    x = torch.randn(2, 5)
    for _ in range(3):
        model(x)
        alg.after_backward()
        alg.after_optimizer_step(optimizer)
    assert alg.replacement_count == 1
    layer, outgoing = model.layers[:2]
    unit = alg.last_replaced_units[0][1]
    assert torch.count_nonzero(outgoing.weight[:, unit]) == 0
    assert torch.all(alg.age[0][unit] == 0)
    assert torch.all(alg.utility[0][unit] == 0)
    assert torch.all(alg.mean_activation[0][unit] == 0)
    assert abs(alg.accumulated_replacements[0] - 0.2) < 1e-6
