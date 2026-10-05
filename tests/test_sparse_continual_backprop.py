import math
import torch
from ccdn.models.sparse_mlp import SparseMLP
from ccdn.baselines.continual_backprop import ContinualBackprop

def test_sparse_unit_replacement_preserves_masks_budget_and_clears_momentum():
    torch.manual_seed(8)
    model=SparseMLP(input_size=24,hidden_sizes=(16,12),output_size=4,density=.35,seed=17)
    algorithm=ContinualBackprop(model,replacement_interval=1,replacement_fraction=.0625,maturity=0,utility_decay=1.0)
    for utility in algorithm.utility: utility.fill_(10); utility[0]=0
    optimizer=torch.optim.SGD(model.parameters(),lr=0,momentum=.9)
    masks=[layer.mask.clone() for layer in model.layers]
    counts=[layer.active_count for layer in model.layers]
    for parameter in model.parameters(): optimizer.state[parameter]['momentum_buffer']=torch.ones_like(parameter)
    optimizer.zero_grad(); model(torch.randn(5,24)).sum().backward(); algorithm.after_backward(); optimizer.step(); algorithm.after_optimizer_step(optimizer)
    assert (0,0) in algorithm.last_replaced_units
    assert [layer.active_count for layer in model.layers]==counts
    assert all(torch.equal(layer.mask,mask) for layer,mask in zip(model.layers,masks))
    for layer,mask in zip(model.layers,masks): assert torch.count_nonzero(layer.weight[~mask])==0
    for li,unit in algorithm.last_replaced_units:
        incoming=model.layers[li]; outgoing=model.layers[li+1]
        active_in=incoming.mask[unit]; active_out=outgoing.mask[:,unit]
        assert torch.all(incoming.weight[unit,active_in].abs()<=1/math.sqrt(incoming.in_features))
        assert torch.all(outgoing.weight[active_out,unit].abs()<=1/math.sqrt(outgoing.in_features))
        assert incoming.bias[unit].abs()<=1/math.sqrt(incoming.in_features)
        assert torch.count_nonzero(optimizer.state[incoming.weight]['momentum_buffer'][unit])==0
        assert torch.count_nonzero(optimizer.state[outgoing.weight]['momentum_buffer'][:,unit])==0
        assert optimizer.state[incoming.bias]['momentum_buffer'][unit]==0
