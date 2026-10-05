import torch
from .base import Baseline
class StaticSparse(Baseline):
    name="static_sparse"
    def __init__(self,model): super().__init__(model); self._initial=[l.mask.clone() for l in model.layers]
    def after_optimizer_step(self,optimizer):
        with torch.no_grad():
            for layer in self.model.layers:
                layer.weight.mul_(layer.mask)
                for v in optimizer.state.get(layer.weight,{}).values():
                    if torch.is_tensor(v) and v.shape==layer.weight.shape: v.mul_(layer.mask)
        super().after_optimizer_step(optimizer)
