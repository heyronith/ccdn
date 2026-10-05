import torch

from ccdn.models.sparse_mlp import SparseMLP
from ccdn.official_reference.rigl import RigLReference


def test_reference_prunes_lowest_active_magnitude_and_conserves_capacity():
    model = SparseMLP(6, (), 1, density=4 / 6, seed=1)
    layer = model.layers[0]
    with torch.no_grad():
        layer.mask.fill_(False)
        layer.mask[0, :4] = True
        layer.weight.copy_(torch.tensor([[0.8, 0.02, 0.5, 0.01, 0.0, 0.0]]))
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    port = RigLReference(model)
    pruned, _ = port.update_layer(0, torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.2, 0.9]]), .25, optimizer)
    assert pruned == [3]
    assert int(layer.mask.sum()) == 4
