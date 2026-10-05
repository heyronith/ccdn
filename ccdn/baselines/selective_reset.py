import torch
from .base import Baseline, clear_optimizer_entries, enforce_sparse_state
class SelectiveReset(Baseline):
    name="selective_reset"
    def __init__(self,model,reset_interval=100,reset_fraction=0.01,utility_decay=0.99,selection_scope="global"):
        super().__init__(model); self.reset_interval=int(reset_interval); self.reset_fraction=float(reset_fraction); self.utility_decay=float(utility_decay)
        if selection_scope not in {"global", "per_layer"}: raise ValueError("selection_scope must be global or per_layer")
        self.selection_scope=selection_scope
        self.utility=[torch.zeros_like(l.weight) for l in model.layers]; self.reset_count=0; self.last_reset_positions=[]
        self.event_steps=[]; self.utility_elements_updated=0; self.edges_ranked=0; self.weights_reset=0
    def after_backward(self):
        with torch.no_grad():
            for u,l in zip(self.utility,self.model.layers):
                if l.weight.grad is not None:
                    u.mul_(self.utility_decay).add_((l.weight*l.weight.grad).abs()*l.mask,alpha=1-self.utility_decay)
                    u.mul_(l.mask)
                    self.utility_elements_updated += int(l.mask.sum().item())
    def after_optimizer_step(self,optimizer):
        super().after_optimizer_step(optimizer)
        enforce_sparse_state(self.model, optimizer)
        if self.reset_interval>0 and self.global_step%self.reset_interval==0:
            self.last_reset_positions=[]
            groups=[]
            if self.selection_scope == "per_layer":
                for li,(u,l) in enumerate(zip(self.utility,self.model.layers)):
                    idx=l.mask.flatten().nonzero().flatten()
                    n=min(idx.numel(),max(1,round(idx.numel()*self.reset_fraction))) if idx.numel() and self.reset_fraction>0 else 0
                    self.edges_ranked += int(idx.numel())
                    groups.extend((float(u.flatten()[j]),li,int(j)) for j in (idx[torch.argsort(u.flatten()[idx])[:n]] if n else []))
            else:
                entries=[]
                for li,(u,l) in enumerate(zip(self.utility,self.model.layers)):
                    idx=l.mask.flatten().nonzero().flatten()
                    entries.extend((float(u.flatten()[j]),li,int(j)) for j in idx)
                self.edges_ranked += len(entries)
                n=min(len(entries),max(1,round(len(entries)*self.reset_fraction))) if entries and self.reset_fraction>0 else 0
                groups=sorted(entries)[:n]
            for _,li,j in groups:
                layer=self.model.layers[li]; row,col=divmod(j,layer.weight.shape[1])
                if getattr(self.model, "initialization", "legacy") == "active_fan_in_kaiming":
                    fan_in=int(layer.mask[row].sum().item())
                    gain=1.0 if getattr(layer,"is_output",False) else 2.0**0.5
                    bound=(3.0**0.5)*gain/(fan_in**0.5)
                    with torch.no_grad(): layer.weight[row,col].uniform_(-bound,bound)
                else:
                    layer.reset_weight(row,col)
                self.utility[li][row,col]=0
                m=torch.zeros_like(layer.weight,dtype=torch.bool); m[row,col]=True; clear_optimizer_entries(optimizer,layer.weight,m)
                self.reset_count+=1; self.last_reset_positions.append((li,row,col))
            if groups: self.event_steps.append(self.global_step)
    def metrics(self): return {"reset_count":self.reset_count,"reset_event_steps":list(self.event_steps),"topology_events":len(self.event_steps),"weights_reset":self.reset_count,"utility_elements_updated":self.utility_elements_updated,"edges_ranked":self.edges_ranked}
    def state_dict(self): return {**super().state_dict(),"utility":self.utility,"reset_count":self.reset_count,"last_reset_positions":self.last_reset_positions,"selection_scope":self.selection_scope,"event_steps":self.event_steps,"utility_elements_updated":self.utility_elements_updated,"edges_ranked":self.edges_ranked}
    def load_state_dict(self,s):
        super().load_state_dict(s); self.utility=[x.clone() for x in s["utility"]]; self.reset_count=int(s["reset_count"]); self.last_reset_positions=[tuple(x) for x in s.get("last_reset_positions",[])]
        self.selection_scope=s.get("selection_scope",self.selection_scope); self.event_steps=[int(x) for x in s.get("event_steps",[])]; self.utility_elements_updated=int(s.get("utility_elements_updated",0)); self.edges_ranked=int(s.get("edges_ranked",0))
