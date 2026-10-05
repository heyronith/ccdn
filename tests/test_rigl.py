import torch
from ccdn.models.sparse_mlp import SparseMLP
from ccdn.baselines.rigl import RigL

def test_rigl_conserves_edges_changes_connections_and_clears_state():
    m=SparseMLP(input_size=4,hidden_sizes=(4,),output_size=2,density=.5,seed=4); a=RigL(m,rewire_interval=1,rewire_fraction=.25); opt=torch.optim.SGD(m.parameters(),lr=.1,momentum=.9)
    old=[l.mask.clone() for l in m.layers]; counts=[l.active_count for l in m.layers]
    opt.zero_grad(); m(torch.randn(8,4)).sum().backward(); opt.step(); a.after_optimizer_step(opt)
    assert [l.active_count for l in m.layers]==counts
    assert any(not torch.equal(l.mask,z) for l,z in zip(m.layers,old))
    for l in m.layers: assert torch.count_nonzero(l.weight[~l.mask])==0
