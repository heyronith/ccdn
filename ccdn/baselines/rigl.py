"""Minimal RigL-style global-step rewiring; dense masked kernels are used."""
import torch
from .base import Baseline, clear_optimizer_entries, enforce_sparse_state
from ccdn.models.sparse_mlp import dense_candidate_gradient_scores
class RigL(Baseline):
    name="rigl"
    def __init__(self,model,rewire_interval=100,rewire_fraction=0.1):
        super().__init__(model); self.interval=int(rewire_interval); self.fraction=float(rewire_fraction); self.rewire_count=0; self.last_pruned=[]; self.last_grown=[]
    def after_optimizer_step(self,optimizer):
        super().after_optimizer_step(optimizer)
        enforce_sparse_state(self.model,optimizer)
        if self.interval<=0 or self.global_step%self.interval: return
        with torch.no_grad():
            self.last_pruned=[]; self.last_grown=[]
            for i,l in enumerate(self.model.layers):
                m=l.mask.flatten(); active=m.nonzero().flatten(); inactive=(~m).nonzero().flatten()
                n=min(active.numel(),inactive.numel(),max(1,round(active.numel()*self.fraction))) if active.numel() and inactive.numel() else 0
                if not n: continue
                prune=active[torch.argsort(l.weight.flatten()[active].abs())[:n]]
                # dL/dW approximation for every dense candidate, including masked edges.
                scores=dense_candidate_gradient_scores(self.model,i).flatten()
                grow=inactive[torch.argsort(scores[inactive],descending=True)[:n]]
                pm=torch.zeros_like(m); pm[prune]=True; gm=torch.zeros_like(m); gm[grow]=True
                l.mask.flatten()[prune]=False; l.mask.flatten()[grow]=True
                l.weight.flatten()[prune]=0; l.weight.flatten()[grow]=0
                clear_optimizer_entries(optimizer,l.weight,pm.reshape_as(l.weight)|gm.reshape_as(l.weight))
                self.last_pruned.append((i,prune.tolist())); self.last_grown.append((i,grow.tolist()))
                self.rewire_count+=1
                assert int(l.mask.sum())==active.numel()
    def metrics(self): return {"rewire_count":self.rewire_count}
    def state_dict(self): return {**super().state_dict(),"rewire_count":self.rewire_count,"last_pruned":self.last_pruned,"last_grown":self.last_grown}
    def load_state_dict(self,s): super().load_state_dict(s); self.rewire_count=int(s["rewire_count"]); self.last_pruned=[(int(i),list(v)) for i,v in s.get("last_pruned",[])]; self.last_grown=[(int(i),list(v)) for i,v in s.get("last_grown",[])]
