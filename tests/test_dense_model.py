import torch
from ccdn.models.dense_mlp import DenseMLP

def test_dense_shapes_and_width_configuration():
    m=DenseMLP(hidden_sizes=(12,7)); assert m(torch.randn(4,784)).shape==(4,10)
    assert sum(p.numel() for p in m.parameters())==784*12+12+12*7+7+7*10+10
