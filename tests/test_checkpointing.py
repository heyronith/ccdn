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
