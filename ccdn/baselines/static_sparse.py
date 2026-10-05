import torch
from .base import Baseline, enforce_sparse_state
class StaticSparse(Baseline):
    name="static_sparse"
    def __init__(self,model): super().__init__(model); self._initial=[l.mask.clone() for l in model.layers]
    def after_optimizer_step(self,optimizer):
        enforce_sparse_state(self.model,optimizer)
        super().after_optimizer_step(optimizer)
