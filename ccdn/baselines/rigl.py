"""Minimal RigL-style global-step rewiring; dense masked kernels are used."""
import torch
from .base import Baseline, clear_optimizer_entries
class RigL(Baseline):
    name="rigl"
    def __init__(self,model,rewire_interval=100,rewire_fraction=0.1):
        super().__init__(model); self.interval=int(rewire_interval); self.fraction=float(rewire_fraction); self.rewire_count=0
    def after_optimizer_step(self,optimizer):
        super().after_optimizer_step(optimizer)
        if self.interval<=0 or self.global_step%self.interval: return
        with torch.no_grad():
            for i,l in enumerate(self.model.layers):
                m=l.mask.flatten(); active=m.nonzero().flatten(); inactive=(~m).nonzero().flatten()
                n=min(active.numel(),inactive.numel(),max(1,round(active.numel()*self.fraction))) if active.numel() and inactive.numel() else 0
                if not n: continue
                prune=active[torch.argsort(l.weight.flatten()[active].abs())[:n]]
                inp=self.model._layer_inputs[i]; delta=self.model._layer_deltas[i]
                if inp is not None and delta is not None:
                    # dL/dW approximation for every dense candidate, including masked edges.
                    scores=(delta.transpose(0,1)@inp).abs().flatten()
                else: scores=torch.zeros_like(l.weight.flatten())
                grow=inactive[torch.argsort(scores[inactive],descending=True)[:n]]
                pm=torch.zeros_like(m); pm[prune]=True; gm=torch.zeros_like(m); gm[grow]=True
                l.mask.flatten()[prune]=False; l.mask.flatten()[grow]=True
                l.weight.flatten()[prune]=0; l.weight.flatten()[grow]=0
                clear_optimizer_entries(optimizer,l.weight,pm.reshape_as(l.weight)|gm.reshape_as(l.weight))
                self.rewire_count+=1
                assert int(l.mask.sum())==active.numel()
    def metrics(self): return {"rewire_count":self.rewire_count}
    def state_dict(self): return {**super().state_dict(),"rewire_count":self.rewire_count}
    def load_state_dict(self,s): super().load_state_dict(s); self.rewire_count=int(s["rewire_count"])
