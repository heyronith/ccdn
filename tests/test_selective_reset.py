import math
import torch
from ccdn.models.sparse_mlp import SparseMLP
from ccdn.baselines.selective_reset import SelectiveReset

def test_reset_cadence_preserves_topology_and_clears_momentum():
    m=SparseMLP(input_size=4,hidden_sizes=(3,),output_size=2,density=.7,seed=2)
    a=SelectiveReset(m,reset_interval=1,reset_fraction=.01,utility_decay=1.0)
    opt=torch.optim.SGD(m.parameters(),lr=0,momentum=.9)
    masks=[l.mask.clone() for l in m.layers]
    for p in m.parameters(): opt.state[p]["momentum_buffer"]=torch.ones_like(p)
    opt.zero_grad(); m(torch.ones(2,4)).sum().backward(); a.after_backward(); opt.step(); a.after_optimizer_step(opt)
    assert a.reset_count>=1 and a.last_reset_positions
    assert all(torch.equal(l.mask,z) for l,z in zip(m.layers,masks))
    for li,row,col in a.last_reset_positions:
        layer=m.layers[li]; bound=1/math.sqrt(layer.in_features)
        assert layer.mask[row,col]
        assert abs(float(layer.weight[row,col].detach()))<=bound
        assert opt.state[layer.weight]["momentum_buffer"][row,col]==0
