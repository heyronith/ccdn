import torch
from torch import nn
from ccdn.metrics.resources import account
from ccdn.baselines.static_dense import StaticDense

def test_manual_dense_model_resource_counts():
    m=nn.Sequential(nn.Linear(2,3),nn.Linear(3,1)); a=StaticDense(m); o=torch.optim.SGD(m.parameters(),lr=.1)
    r=account(m,a,o)
    # (2*3+3) + (3*1+1) = 13
    assert r["total_possible_parameters"]==13 and r["logical_active_parameters"]==13
    assert r["actual_tensor_parameters"]==13 and r["estimated_total_training_state_memory_bytes"]>=r["model_tensor_memory_bytes"]
