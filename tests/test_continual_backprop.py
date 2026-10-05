import torch
from ccdn.models.dense_mlp import DenseMLP
from ccdn.baselines.continual_backprop import ContinualBackprop

def test_mature_low_utility_units_replace_and_clear_incoming_outgoing_momentum():
    m=DenseMLP(input_size=8,hidden_sizes=(16,40),output_size=3)
    a=ContinualBackprop(m,replacement_interval=1,replacement_fraction=.0625,maturity=0,utility_decay=1.0)
    for u in a.utility: u.fill_(10); u[0]=0
    opt=torch.optim.SGD(m.parameters(),lr=0,momentum=.9)
    for p in m.parameters(): opt.state[p]["momentum_buffer"]=torch.ones_like(p)
    opt.zero_grad(); loss=m(torch.ones(2,8)).sum(); loss.backward(); a.after_backward(); opt.step(); a.after_optimizer_step(opt)
    assert a.replacement_count>=1 and a.last_replaced_units
    assert (0,0) in a.last_replaced_units
    for li,unit in a.last_replaced_units:
        incoming=m.layers[li].weight; outgoing=m.layers[li+1].weight
        assert torch.count_nonzero(opt.state[incoming]["momentum_buffer"][unit,:])==0
        assert torch.count_nonzero(opt.state[outgoing]["momentum_buffer"][:,unit])==0
        assert torch.count_nonzero(opt.state[m.layers[li].bias]["momentum_buffer"][unit])==0
    assert m.hidden_sizes==(16,40)
    assert a.age[0][0]==0
