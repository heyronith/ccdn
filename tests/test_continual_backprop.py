import torch
from ccdn.models.dense_mlp import DenseMLP
from ccdn.baselines.continual_backprop import ContinualBackprop

def test_mature_low_utility_units_replace_and_state_clears():
    m=DenseMLP(input_size=4,hidden_sizes=(4,),output_size=2); a=ContinualBackprop(m,replacement_interval=1,replacement_fraction=.25,maturity=0,utility_decay=0)
    with torch.no_grad(): a.utility[0].fill_(0); m.layers[0].weight[0].fill_(100)
    opt=torch.optim.SGD(m.parameters(),lr=.01,momentum=.9)
    for p in m.parameters(): opt.state[p]["momentum_buffer"]=torch.ones_like(p)
    before=m.layers[0].weight.clone(); opt.zero_grad(); loss=m(torch.ones(2,4)).sum(); loss.backward(); a.after_backward(); opt.step(); a.after_optimizer_step(opt)
    assert a.replacement_count>=1 and m.hidden_sizes==(4,)
    assert not torch.equal(before,m.layers[0].weight)
    assert a.age[0].min()==0
