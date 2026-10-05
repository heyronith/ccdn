from __future__ import annotations
import torch
from torch import nn

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
    @property
    def active_count(self): return sum(l.active_count for l in self.layers)+sum(l.bias.numel() for l in self.layers if l.bias is not None)
    @property
    def logical_total(self): return sum(l.weight.numel()+ (l.bias.numel() if l.bias is not None else 0) for l in self.layers)
