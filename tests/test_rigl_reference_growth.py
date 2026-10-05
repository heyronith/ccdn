import torch

from ccdn.models.sparse_mlp import SparseMLP
from ccdn.official_reference.rigl import RigLReference


def test_reference_grows_high_gradient_inactive_edge_at_zero_and_resets_slot():
    model = SparseMLP(6, (), 1, density=4 / 6, seed=1)
    layer = model.layers[0]
    with torch.no_grad():
        layer.mask.fill_(False)
        layer.mask[0, :4] = True
        layer.weight.copy_(torch.tensor([[0.8, 0.02, 0.5, 0.01, 0.0, 0.0]]))
    optimizer = torch.optim.SGD(model.parameters(), lr=.1, momentum=.9)
    optimizer.state[layer.weight]["momentum_buffer"] = torch.ones_like(layer.weight)
    port = RigLReference(model)
    _, grown = port.update_layer(0, torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.2, 0.9]]), .25, optimizer)
    assert grown == [5]
    assert layer.mask[0, 5]
    assert layer.weight[0, 5] == 0
    assert optimizer.state[layer.weight]["momentum_buffer"][0, 5] == 0
    assert int(layer.mask.sum()) == 4
