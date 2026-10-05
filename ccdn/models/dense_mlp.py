from __future__ import annotations
import torch
from torch import nn

class DenseMLP(nn.Module):
    """Configurable fully-connected MLP. Hidden-layer IO is exposed for algorithms."""
    def __init__(self, input_size: int = 784, hidden_sizes=(256, 256), output_size: int = 10):
        super().__init__()
        dims = [input_size, *hidden_sizes, output_size]
        self.layers = nn.ModuleList(nn.Linear(a, b) for a, b in zip(dims[:-1], dims[1:]))
        self.hidden_sizes = tuple(hidden_sizes)
        self.last_activations = []
        self._layer_inputs = [None] * len(self.layers)
        self._layer_deltas = [None] * len(self.layers)
    def forward(self, x):
        x = x.reshape(x.shape[0], -1)
        self.last_activations = []
        for i, layer in enumerate(self.layers):
            self._layer_inputs[i] = x.detach()
            x = layer(x)
            if x.requires_grad:
                x.register_hook(lambda grad, idx=i: self._save_delta(idx, grad))
            if i < len(self.layers) - 1:
                x = torch.relu(x)
                self.last_activations.append(x)
        return x
    def _save_delta(self, idx, grad):
        self._layer_deltas[idx] = grad.detach()
    def reset_unit(self, layer_index: int, unit_index: int):
        """Reinitialize hidden unit's incoming row/bias and outgoing column."""
        layer = self.layers[layer_index]
        with torch.no_grad():
            nn.init.kaiming_uniform_(layer.weight[unit_index:unit_index+1], a=5 ** 0.5)
            if layer.bias is not None: layer.bias[unit_index].zero_()
            nxt = self.layers[layer_index + 1]
            nn.init.kaiming_uniform_(nxt.weight[:, unit_index:unit_index+1], a=5 ** 0.5)
