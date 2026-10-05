import torch
from ccdn.utils.reproducibility import seed_everything
from ccdn.experiments.train import build

def test_same_seed_same_initialization_and_short_step():
    c={"experiment":{"seed":44},"model":{"type":"static_dense","hidden_sizes":[8,8]},"optimizer":{"lr":.01}}
    seed_everything(44); m1,a1,o1=build(c,torch.device("cpu")); seed_everything(44); m2,a2,o2=build(c,torch.device("cpu"))
    for x,y in zip(m1.parameters(),m2.parameters()): assert torch.equal(x,y)
    xx=torch.randn(3,784); yy=torch.tensor([1,2,3]); seed_everything(9)
    for m,o,a in [(m1,o1,a1),(m2,o2,a2)]: o.zero_grad(); loss=torch.nn.functional.cross_entropy(m(xx),yy); loss.backward(); a.after_backward(); o.step(); a.after_optimizer_step(o)
    for x,y in zip(m1.parameters(),m2.parameters()): assert torch.equal(x,y)

def test_all_primary_sparse_methods_share_seeded_initial_state():
    from ccdn.algorithms.ccdn_0a import CCDN0A
    from ccdn.baselines import StaticSparse,SelectiveReset,RigL,ContinualBackprop
    types=('static_sparse','selective_reset','rigl','continual_backprop_sparse','ccdn_0a')
    states=[]
    for typ in types:
        cfg={'experiment':{'seed':77},'model':{'type':typ,'hidden_sizes':[12,9],'density':.25},'optimizer':{'lr':.01,'momentum':0}}
        seed_everything(77); model,algorithm,_=build(cfg,torch.device('cpu'))
        states.append({key:value.detach().clone() for key,value in model.state_dict().items()})
    for state in states[1:]:
        assert state.keys()==states[0].keys()
        assert all(torch.equal(state[key],states[0][key]) for key in state)
