import torch
from .base import Baseline, clear_optimizer_entries
class SelectiveReset(Baseline):
    name="selective_reset"
    def __init__(self,model,reset_interval=100,reset_fraction=0.01,utility_decay=0.99):
        super().__init__(model); self.reset_interval=int(reset_interval); self.reset_fraction=float(reset_fraction); self.utility_decay=float(utility_decay)
        self.utility=[torch.zeros_like(l.weight) for l in model.layers]; self.reset_count=0; self.last_reset_positions=[]
    def after_backward(self):
        with torch.no_grad():
            for u,l in zip(self.utility,self.model.layers):
                if l.weight.grad is not None: u.mul_(self.utility_decay).add_((l.weight*l.weight.grad).abs(),alpha=1-self.utility_decay)
    def after_optimizer_step(self,optimizer):
        super().after_optimizer_step(optimizer)
        if self.reset_interval>0 and self.global_step%self.reset_interval==0:
            self.last_reset_positions=[]
            entries=[]
            for li,(u,l) in enumerate(zip(self.utility,self.model.layers)):
                idx=l.mask.flatten().nonzero().flatten()
                entries.extend((float(u.flatten()[j]),li,int(j)) for j in idx)
            n=min(len(entries),max(1,round(len(entries)*self.reset_fraction))) if entries and self.reset_fraction>0 else 0
            for _,li,j in sorted(entries)[:n]:
                layer=self.model.layers[li]; row,col=divmod(j,layer.weight.shape[1])
                layer.reset_weight(row,col); self.utility[li][row,col]=0
                m=torch.zeros_like(layer.weight,dtype=torch.bool); m[row,col]=True; clear_optimizer_entries(optimizer,layer.weight,m)
                self.reset_count+=1; self.last_reset_positions.append((li,row,col))
    def metrics(self): return {"reset_count":self.reset_count}
    def state_dict(self): return {**super().state_dict(),"utility":self.utility,"reset_count":self.reset_count,"last_reset_positions":self.last_reset_positions}
    def load_state_dict(self,s):
        super().load_state_dict(s); self.utility=[x.clone() for x in s["utility"]]; self.reset_count=int(s["reset_count"]); self.last_reset_positions=[tuple(x) for x in s.get("last_reset_positions",[])]
