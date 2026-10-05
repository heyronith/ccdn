"""Lightweight Continual Backprop reproduction, not an exact paper reproduction.

Uses mean absolute activation-times-backpropagated-sensitivity as unit utility and
replaces mature bottom-utility hidden units on a fixed global optimizer cadence.
"""
import torch
from .base import Baseline, clear_optimizer_entries
class ContinualBackprop(Baseline):
    name="continual_backprop"
    def __init__(self,model,replacement_interval=100,replacement_fraction=0.01,maturity=100,utility_decay=0.99):
        super().__init__(model); self.interval=int(replacement_interval); self.fraction=float(replacement_fraction); self.maturity=int(maturity); self.decay=float(utility_decay)
        self.utility=[torch.zeros(n,device=next(model.parameters()).device) for n in model.hidden_sizes]
        self.age=[torch.zeros(n,dtype=torch.long,device=next(model.parameters()).device) for n in model.hidden_sizes]; self.replacement_count=0; self.last_replaced_units=[]
    def after_backward(self):
        with torch.no_grad():
            for i,(u,act) in enumerate(zip(self.utility,self.model.last_activations)):
                delta=self.model._layer_deltas[i]
                if delta is not None:
                    salience=(act*delta).abs().mean(0)
                    u.mul_(self.decay).add_(salience,alpha=1-self.decay)
    def after_optimizer_step(self,optimizer):
        super().after_optimizer_step(optimizer)
        for a in self.age: a.add_(1)
        if self.interval<=0 or self.global_step%self.interval: return
        self.last_replaced_units=[]
        for li,u in enumerate(self.utility):
            eligible=(self.age[li]>=self.maturity).nonzero().flatten()
            n=min(eligible.numel(), max(1,round(self.model.hidden_sizes[li]*self.fraction))) if self.fraction>0 else 0
            if not n: continue
            units=eligible[torch.argsort(u[eligible])[:n]]
            for unit_t in units:
                unit=int(unit_t); self.model.reset_unit(li,unit)
                incoming=self.model.layers[li].weight; outgoing=self.model.layers[li+1].weight
                mi=torch.zeros_like(incoming,dtype=torch.bool); mi[unit,:]=True
                mo=torch.zeros_like(outgoing,dtype=torch.bool); mo[:,unit]=True
                clear_optimizer_entries(optimizer,incoming,mi); clear_optimizer_entries(optimizer,outgoing,mo)
                if self.model.layers[li].bias is not None:
                    b=self.model.layers[li].bias; clear_optimizer_entries(optimizer,b,torch.arange(b.numel(),device=b.device)==unit)
                self.utility[li][unit]=0; self.age[li][unit]=0; self.replacement_count+=1; self.last_replaced_units.append((li,unit))
    def metrics(self): return {"replacement_count":self.replacement_count}
    def state_dict(self): return {**super().state_dict(),"utility":self.utility,"age":self.age,"replacement_count":self.replacement_count,"last_replaced_units":self.last_replaced_units}
    def load_state_dict(self,s):
        super().load_state_dict(s); self.utility=[x.clone() for x in s["utility"]]; self.age=[x.clone() for x in s["age"]]; self.replacement_count=int(s["replacement_count"]); self.last_replaced_units=[tuple(x) for x in s.get("last_replaced_units",[])]
