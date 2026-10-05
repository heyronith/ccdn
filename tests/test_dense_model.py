import math
import torch
from ccdn.models.dense_mlp import DenseMLP

def test_dense_shapes_and_width_configuration():
    m=DenseMLP(hidden_sizes=(12,7)); assert m(torch.randn(4,784)).shape==(4,10)
    assert sum(p.numel() for p in m.parameters())==784*12+12+12*7+7+7*10+10

def test_replacement_initialization_uses_full_layer_fan_in_bounds():
    torch.manual_seed(123)
    m=DenseMLP(input_size=64,hidden_sizes=(16,),output_size=40)
    first,second=m.layers
    old_first=first.weight.clone(); old_first_bias=first.bias.clone(); old_second=second.weight.clone()
    replaced=5
    m.reset_unit(0,replaced)
    incoming_bound=1/math.sqrt(first.in_features)
    outgoing_bound=1/math.sqrt(second.in_features)
    assert torch.all(first.weight[replaced].abs()<=incoming_bound)
    assert first.bias[replaced].abs()<=incoming_bound
    # The outgoing layer has 16 inputs. A slice-based Kaiming init would use fan-in 1.
    assert second.in_features==16 and torch.all(second.weight[:,replaced].abs()<=outgoing_bound)
    assert torch.equal(first.weight[:replaced],old_first[:replaced])
    assert torch.equal(first.weight[replaced+1:],old_first[replaced+1:])
    assert torch.equal(first.bias[:replaced],old_first_bias[:replaced])
    assert torch.equal(first.bias[replaced+1:],old_first_bias[replaced+1:])
    assert torch.equal(second.weight[:,:replaced],old_second[:,:replaced])
    assert torch.equal(second.weight[:,replaced+1:],old_second[:,replaced+1:])
