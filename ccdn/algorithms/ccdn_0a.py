"""CCDN-0A: utility-retaining sparse rewiring, with no other CCDN mechanisms."""
import torch

from ccdn.baselines.base import Baseline, clear_optimizer_entries, enforce_sparse_state
from ccdn.models.sparse_mlp import dense_candidate_gradient_scores


class CCDN0A(Baseline):
    """Fixed-budget, per-layer utility pruning with gradient-guided regrowth.

    This initial algorithm contains only ordinary gradient learning, EMA
    connection utility, and cadence-based structural rewiring. It has no task
    signals, consolidation, reserve score, or trainability regularization.
    """
    name="ccdn_0a"

    def __init__(self, model, structural_interval=64, turnover_fraction=0.05, utility_decay=0.99):
        super().__init__(model)
        self.structural_interval=int(structural_interval)
        self.turnover_fraction=float(turnover_fraction)
        self.utility_decay=float(utility_decay)
        self.utility=[torch.zeros_like(layer.weight) for layer in model.layers]
        self.rewire_event_count=0
        self.total_edges_pruned=0
        self.total_edges_grown=0
        self.cumulative_edge_turnover=0
        self.last_pruned=[]
        self.last_grown=[]
        self.structural_event_steps=[]
        self.utility_elements_updated=0
        self.candidate_gradient_elements_scored=0
        self.edges_ranked=0

    def after_backward(self):
        with torch.no_grad():
            for utility,layer in zip(self.utility,self.model.layers):
                mask=layer.mask
                if layer.weight.grad is not None:
                    utility.mul_(self.utility_decay).add_((layer.weight*layer.weight.grad).abs()*mask,alpha=1-self.utility_decay)
                    self.utility_elements_updated += int(mask.sum().item())
                utility.mul_(mask)

    def after_optimizer_step(self,optimizer):
        super().after_optimizer_step(optimizer)
        enforce_sparse_state(self.model,optimizer)
        if self.structural_interval<=0 or self.global_step%self.structural_interval:
            return
        event_pruned=event_grown=0
        self.last_pruned=[]; self.last_grown=[]
        with torch.no_grad():
            for index,(layer,utility) in enumerate(zip(self.model.layers,self.utility)):
                flat_mask=layer.mask.flatten()
                active=flat_mask.nonzero().flatten()
                inactive=(~flat_mask).nonzero().flatten()
                n=min(active.numel(),inactive.numel(),max(1,round(active.numel()*self.turnover_fraction))) if active.numel() and inactive.numel() and self.turnover_fraction>0 else 0
                if not n:
                    continue
                flat_utility=utility.flatten()
                self.edges_ranked += int(active.numel())
                prune=active[torch.argsort(flat_utility[active])[:n]]
                candidate=dense_candidate_gradient_scores(self.model,index).flatten()
                self.candidate_gradient_elements_scored += int(candidate.numel())
                grow=inactive[torch.argsort(candidate[inactive],descending=True)[:n]]
                prune_mask=torch.zeros_like(flat_mask); prune_mask[prune]=True
                grow_mask=torch.zeros_like(flat_mask); grow_mask[grow]=True
                flat_mask[prune]=False; flat_mask[grow]=True
                layer.weight.flatten()[prune]=0
                layer.weight.flatten()[grow]=0
                flat_utility[prune]=0; flat_utility[grow]=0
                clear_optimizer_entries(optimizer,layer.weight,(prune_mask|grow_mask).reshape_as(layer.weight))
                self.last_pruned.append((index,prune.tolist()))
                self.last_grown.append((index,grow.tolist()))
                event_pruned+=n; event_grown+=n
                if int(layer.mask.sum())!=int(active.numel()):
                    raise RuntimeError(f"CCDN-0A changed active-edge budget in layer {index}")
        if event_pruned:
            self.rewire_event_count+=1
            self.structural_event_steps.append(self.global_step)
            self.total_edges_pruned+=event_pruned
            self.total_edges_grown+=event_grown
            self.cumulative_edge_turnover+=event_pruned+event_grown

    def metrics(self):
        values={
            "rewire_event_count":self.rewire_event_count,
            "topology_events":self.rewire_event_count,
            "total_edges_pruned":self.total_edges_pruned,
            "edges_pruned":self.total_edges_pruned,
            "total_edges_grown":self.total_edges_grown,
            "edges_grown":self.total_edges_grown,
            "cumulative_edge_turnover":self.cumulative_edge_turnover,
            "structural_event_steps":list(self.structural_event_steps),
            "utility_elements_updated":self.utility_elements_updated,
            "candidate_gradient_elements_scored":self.candidate_gradient_elements_scored,
            "edges_ranked":self.edges_ranked,
        }
        active_values=[]
        for index,(layer,utility) in enumerate(zip(self.model.layers,self.utility)):
            mask=layer.mask
            selected=utility[mask]
            active_values.append(selected)
            values[f"layer_{index}_active_edges"]=int(mask.sum().item())
        if active_values:
            all_active=torch.cat(active_values)
            values["mean_active_utility"]=float(all_active.mean().item())
            values["std_active_utility"]=float(all_active.std(unbiased=False).item())
            values["median_active_utility"]=float(all_active.median().item())
        else:
            values["mean_active_utility"]=0.0; values["std_active_utility"]=0.0; values["median_active_utility"]=0.0
        return values

    def state_dict(self):
        return {
            **super().state_dict(),
            "utility":self.utility,
            "rewire_event_count":self.rewire_event_count,
            "total_edges_pruned":self.total_edges_pruned,
            "total_edges_grown":self.total_edges_grown,
            "cumulative_edge_turnover":self.cumulative_edge_turnover,
            "last_pruned":self.last_pruned,
            "last_grown":self.last_grown,
            "structural_event_steps":self.structural_event_steps,
            "utility_elements_updated":self.utility_elements_updated,
            "candidate_gradient_elements_scored":self.candidate_gradient_elements_scored,
            "edges_ranked":self.edges_ranked,
        }

    def load_state_dict(self,state):
        super().load_state_dict(state)
        self.utility=[value.clone() for value in state["utility"]]
        self.rewire_event_count=int(state["rewire_event_count"])
        self.total_edges_pruned=int(state["total_edges_pruned"])
        self.total_edges_grown=int(state["total_edges_grown"])
        self.cumulative_edge_turnover=int(state["cumulative_edge_turnover"])
        self.last_pruned=[(int(i),list(v)) for i,v in state.get("last_pruned",[])]
        self.last_grown=[(int(i),list(v)) for i,v in state.get("last_grown",[])]
        self.structural_event_steps=[int(x) for x in state.get("structural_event_steps",[])]
        self.utility_elements_updated=int(state.get("utility_elements_updated",0))
        self.candidate_gradient_elements_scored=int(state.get("candidate_gradient_elements_scored",0))
        self.edges_ranked=int(state.get("edges_ranked",0))
