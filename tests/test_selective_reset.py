import torch
from ccdn.models.sparse_mlp import SparseMLP
from ccdn.baselines.selective_reset import SelectiveReset

def test_reset_cadence_preserves_topology_and_changes_active_weight():
    m=SparseMLP(input_size=4,hidden_sizes=(3,),output_size=2,density=.7,seed=2); a=SelectiveReset(m,reset_interval=2,reset_fraction=.1,utility_decay=0)
    opt=torch.optim.SGD(m.parameters(),lr=0)
    masks=[l.mask.clone() for l in m.layers]; initial=[l.weight.clone() for l in m.layers]
    for step in range(2):
        opt.zero_grad(); m(torch.ones(2,4)).sum().backward(); a.after_backward(); opt.step(); a.after_optimizer_step(opt)
    assert a.reset_count>=1 and all(torch.equal(l.mask,z) for l,z in zip(m.layers,masks))
    assert any(not torch.equal(l.weight,z) for l,z in zip(m.layers,initial))
