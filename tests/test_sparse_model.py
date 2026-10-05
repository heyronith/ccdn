import torch
from ccdn.models.sparse_mlp import SparseMLP
from ccdn.baselines.static_sparse import StaticSparse

def test_sparse_mask_fixed_and_inactive_have_no_effect_or_update():
    m=SparseMLP(input_size=4,hidden_sizes=(3,),output_size=2,density=.5,seed=3); alg=StaticSparse(m); masks=[l.mask.clone() for l in m.layers]
    for l in m.layers:
        with torch.no_grad(): l.weight[~l.mask]=1000
    x=torch.randn(2,4); y=m(x); opt=torch.optim.SGD(m.parameters(),lr=.1,momentum=.9)
    y.sum().backward(); opt.step(); alg.after_optimizer_step(opt)
    assert all(torch.equal(l.mask,mask) for l,mask in zip(m.layers,masks))
    assert all(torch.count_nonzero(l.weight[~l.mask])==0 for l in m.layers)
    assert m.active_count==sum(int(z.sum()) for z in masks)+sum(l.bias.numel() for l in m.layers)
