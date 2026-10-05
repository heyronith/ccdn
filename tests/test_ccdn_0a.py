import torch
from ccdn.models.sparse_mlp import SparseMLP
from ccdn.algorithms.ccdn_0a import CCDN0A

def _step(model,algorithm,optimizer):
    optimizer.zero_grad(set_to_none=True)
    input_size=model.layers[0].in_features; classes=model.layers[-1].out_features
    loss=torch.nn.functional.cross_entropy(model(torch.randn(12,input_size)),torch.randint(0,classes,(12,)))
    loss.backward(); algorithm.after_backward(); optimizer.step(); algorithm.after_optimizer_step(optimizer)

def test_ccdn_utility_rewire_budget_state_and_momentum():
    torch.manual_seed(10)
    model=SparseMLP(input_size=8,hidden_sizes=(8,),output_size=3,density=.5,seed=4)
    algorithm=CCDN0A(model,structural_interval=1,turnover_fraction=.2,utility_decay=.99)
    optimizer=torch.optim.SGD(model.parameters(),lr=.01,momentum=.9)
    counts=[layer.active_count for layer in model.layers]
    for parameter in model.parameters(): optimizer.state[parameter]['momentum_buffer']=torch.ones_like(parameter)
    prior_masks=[layer.mask.clone() for layer in model.layers]
    for _ in range(3):
        _step(model,algorithm,optimizer)
        assert [layer.active_count for layer in model.layers]==counts
        assert all(torch.count_nonzero(layer.weight[~layer.mask])==0 for layer in model.layers)
    assert algorithm.rewire_event_count==3
    assert algorithm.total_edges_pruned==algorithm.total_edges_grown>0
    assert algorithm.cumulative_edge_turnover==algorithm.total_edges_pruned*2
    assert any(not torch.equal(layer.mask,mask) for layer,mask in zip(model.layers,prior_masks))
    assert any(torch.any(utility[layer.mask]>0) for utility,layer in zip(algorithm.utility,model.layers))
    for utility,layer in zip(algorithm.utility,model.layers): assert torch.count_nonzero(utility[~layer.mask])==0
    for index,positions in algorithm.last_pruned:
        layer=model.layers[index]; flat=layer.mask.flatten(); flat_utility=algorithm.utility[index].flatten()
        assert all(not bool(flat[p]) and flat_utility[p]==0 for p in positions)
        assert torch.count_nonzero(optimizer.state[layer.weight]['momentum_buffer'].flatten()[positions])==0
    for index,positions in algorithm.last_grown:
        layer=model.layers[index]; flat=layer.mask.flatten(); flat_utility=algorithm.utility[index].flatten()
        assert all(bool(flat[p]) and flat_utility[p]==0 for p in positions)
        assert torch.count_nonzero(optimizer.state[layer.weight]['momentum_buffer'].flatten()[positions])==0
    metrics=algorithm.metrics()
    for key in ('rewire_event_count','total_edges_pruned','total_edges_grown','cumulative_edge_turnover','mean_active_utility','std_active_utility','layer_0_active_edges','layer_1_active_edges'):
        assert key in metrics

def test_ccdn_inactive_utility_is_ignored_and_zeroed():
    model=SparseMLP(input_size=5,hidden_sizes=(4,),output_size=2,density=.4,seed=9)
    algorithm=CCDN0A(model,structural_interval=100,utility_decay=.5)
    for utility,layer in zip(algorithm.utility,model.layers): utility.fill_(123)
    _step(model,algorithm,torch.optim.SGD(model.parameters(),lr=.01))
    for utility,layer in zip(algorithm.utility,model.layers):
        assert torch.count_nonzero(utility[~layer.mask])==0
        assert torch.all(utility[layer.mask]>=0)
