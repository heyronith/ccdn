import torch
from ccdn.models.sparse_mlp import SparseMLP
from ccdn.baselines.selective_reset import SelectiveReset
from ccdn.utils.checkpointing import save_checkpoint,load_checkpoint

def test_checkpoint_restores_algorithm_model_optimizer_and_step(tmp_path):
    m=SparseMLP(input_size=4,hidden_sizes=(3,),output_size=2,density=.5); a=SelectiveReset(m,reset_interval=1); o=torch.optim.SGD(m.parameters(),lr=.1,momentum=.9)
    x=torch.randn(2,4); o.zero_grad(); m(x).sum().backward(); a.after_backward(); o.step(); a.after_optimizer_step(o)
    path=tmp_path/"c.pt"; save_checkpoint(path,m,a,o,a.global_step,{"x":1}); expected=m(x).detach(); expected_util=[z.clone() for z in a.utility]
    m2=SparseMLP(input_size=4,hidden_sizes=(3,),output_size=2,density=.5); a2=SelectiveReset(m2,reset_interval=1); o2=torch.optim.SGD(m2.parameters(),lr=.1,momentum=.9)
    state=load_checkpoint(path,m2,a2,o2,restore_rng=False)
    assert state["global_step"]==1 and torch.allclose(m2(x),expected)
    assert all(torch.equal(u,v) for u,v in zip(a2.utility,expected_util))

def test_ccdn_checkpoint_restores_masks_weights_utility_and_structural_counters(tmp_path):
    from ccdn.algorithms.ccdn_0a import CCDN0A
    def make():
        model=SparseMLP(input_size=4,hidden_sizes=(5,),output_size=2,density=.5,seed=31)
        algorithm=CCDN0A(model,structural_interval=1,turnover_fraction=.2)
        optimizer=torch.optim.SGD(model.parameters(),lr=.01,momentum=.9)
        return model,algorithm,optimizer
    model,algorithm,optimizer=make()
    optimizer.zero_grad(); model(torch.randn(6,4)).sum().backward(); algorithm.after_backward(); optimizer.step(); algorithm.after_optimizer_step(optimizer)
    expected_masks=[layer.mask.clone() for layer in model.layers]
    expected_weights=[layer.weight.detach().clone() for layer in model.layers]
    expected_utility=[utility.clone() for utility in algorithm.utility]
    expected_state=algorithm.state_dict()
    path=tmp_path/'ccdn.pt'
    save_checkpoint(path,model,algorithm,optimizer,algorithm.global_step,{'model':{'type':'ccdn_0a'}})
    model2,algorithm2,optimizer2=make()
    state=load_checkpoint(path,model2,algorithm2,optimizer2,restore_rng=False)
    assert state['global_step']==algorithm.global_step
    assert all(torch.equal(layer.mask,mask) for layer,mask in zip(model2.layers,expected_masks))
    assert all(torch.equal(layer.weight,weight) for layer,weight in zip(model2.layers,expected_weights))
    assert all(torch.equal(a,b) for a,b in zip(algorithm2.utility,expected_utility))
    for key in ('rewire_event_count','total_edges_pruned','total_edges_grown','cumulative_edge_turnover','global_step'):
        assert algorithm2.state_dict()[key]==expected_state[key]
