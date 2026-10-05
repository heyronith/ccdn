from __future__ import annotations
import torch

class Baseline:
    """Algorithm policy hooks; all cadence uses uninterrupted global optimizer steps."""
    name = "base"
    def __init__(self, model): self.model=model; self.global_step=0
    def before_backward(self, loss): pass
    def after_backward(self): pass
    def after_optimizer_step(self, optimizer): self.global_step += 1
    def metrics(self): return {}
    def state_dict(self): return {"global_step":self.global_step}
    def load_state_dict(self,state): self.global_step=int(state.get("global_step",0))

def clear_optimizer_entries(optimizer, parameter, mask):
    """Clear SGD/Adam per-element state where a parameter is reset or disconnected."""
    state=optimizer.state.get(parameter,{})
    for value in state.values():
        if torch.is_tensor(value) and value.shape == parameter.shape: value.masked_fill_(mask,0)

def enforce_sparse_state(model,optimizer):
    """Project inactive weights and matching optimizer slots to exact zero."""
    with torch.no_grad():
        for layer in getattr(model,"layers",[]):
            if not hasattr(layer,"mask"): continue
            layer.weight.mul_(layer.mask)
            inactive=~layer.mask
            clear_optimizer_entries(optimizer,layer.weight,inactive)
