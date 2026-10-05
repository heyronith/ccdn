from __future__ import annotations
import torch
from torch import nn


def dense_candidate_gradient_scores(model, layer_index):
    """Dense |dL/dW| estimate for every masked candidate edge."""
    inp=model._layer_inputs[layer_index]
    delta=model._layer_deltas[layer_index]
    layer=model.layers[layer_index]
    if inp is None or delta is None:
        return torch.zeros_like(layer.weight)
    return (delta.transpose(0,1) @ inp).abs()

class SparseLinear(nn.Module):
    def __init__(self, in_features, out_features, density=0.2, bias=True, generator=None):
        super().__init__()
        self.in_features, self.out_features = in_features, out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features)) if bias else None
        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)
        if self.bias is not None: nn.init.uniform_(self.bias, -in_features ** -0.5, in_features ** -0.5)
        n = max(1, min(self.weight.numel(), round(self.weight.numel() * density)))
        scores = torch.rand(self.weight.shape, generator=generator)
        mask = torch.zeros(self.weight.numel(), dtype=torch.bool)
        mask[scores.flatten().topk(n).indices] = True
        self.register_buffer("mask", mask.reshape_as(self.weight))
        self.weight.register_hook(lambda grad: grad * self.mask)
        with torch.no_grad(): self.weight.mul_(self.mask)
    def forward(self, x): return torch.nn.functional.linear(x, self.weight * self.mask, self.bias)
    def reset_weight(self, out_index: int, in_index: int):
        """Reinitialize one stored value at this layer's Linear fan-in scale."""
        bound=1.0/self.in_features**0.5
        with torch.no_grad(): self.weight[out_index, in_index].uniform_(-bound,bound)
    @property
    def active_count(self): return int(self.mask.sum().item())

class SparseMLP(nn.Module):
    def __init__(self, input_size=784, hidden_sizes=(256,256), output_size=10, density=0.2, seed=0):
        super().__init__()
        dims=[input_size,*hidden_sizes,output_size]
        gen=torch.Generator().manual_seed(seed)
        self.layers=nn.ModuleList(SparseLinear(a,b,density,generator=gen) for a,b in zip(dims[:-1],dims[1:]))
        self.hidden_sizes=tuple(hidden_sizes); self.last_activations=[]
        self._layer_inputs=[None]*len(self.layers); self._layer_deltas=[None]*len(self.layers)
    def forward(self,x):
        x=x.reshape(x.shape[0],-1); self.last_activations=[]
        for i,layer in enumerate(self.layers):
            self._layer_inputs[i]=x.detach(); x=layer(x)
            if x.requires_grad: x.register_hook(lambda grad,idx=i:self._save_delta(idx,grad))
            if i<len(self.layers)-1:
                x=torch.relu(x); self.last_activations.append(x)
        return x
    def _save_delta(self,idx,grad): self._layer_deltas[idx]=grad.detach()
    def reset_unit(self, layer_index: int, unit_index: int):
        """Reinitialize active weights and bias for one hidden unit, retaining masks."""
        layer=self.layers[layer_index]
        incoming=layer.mask[unit_index].nonzero().flatten()
        if incoming.numel():
            bound=1.0/layer.in_features**0.5
            values=torch.empty(incoming.numel(),device=layer.weight.device,dtype=layer.weight.dtype).uniform_(-bound,bound)
            with torch.no_grad():
                layer.weight[unit_index,incoming]=values
        with torch.no_grad():
            if layer.bias is not None:
                bound=1.0/layer.in_features**0.5
                layer.bias[unit_index].uniform_(-bound,bound)
            if incoming.numel() < layer.in_features:
                layer.weight[unit_index,~layer.mask[unit_index]]=0
        nxt=self.layers[layer_index+1]
        outgoing=nxt.mask[:,unit_index].nonzero().flatten()
        if outgoing.numel():
            bound=1.0/nxt.in_features**0.5
            values=torch.empty(outgoing.numel(),device=nxt.weight.device,dtype=nxt.weight.dtype).uniform_(-bound,bound)
            with torch.no_grad():
                nxt.weight[outgoing,unit_index]=values
        with torch.no_grad():
            if outgoing.numel() < nxt.out_features:
                nxt.weight[~nxt.mask[:,unit_index],unit_index]=0
    @property
    def active_count(self): return sum(l.active_count for l in self.layers)+sum(l.bias.numel() for l in self.layers if l.bias is not None)
    @property
    def logical_total(self): return sum(l.weight.numel()+ (l.bias.numel() if l.bias is not None else 0) for l in self.layers)
