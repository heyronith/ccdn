"""PyTorch port of the pinned Google Research ``SparseRigLOptimizerBase``.

Selection mirrors ``sparse_optimizers_base.py``: floor(active*fraction), keep
largest active magnitudes, then grow from largest dense gradient magnitudes.
The local experiment's dense candidate gradient is the practical approximation
used by our pre-existing RigL baseline; upstream computes the gradient of its
masked weight tensor. When the grow ranking selects a just-pruned edge, the
official implementation retains its stored value and optimizer slot because it
was active before the update; previously inactive grown edges are zeroed and
their optimizer slots are cleared.
"""
from __future__ import annotations

import math

import torch

from ccdn.baselines.base import Baseline, clear_optimizer_entries
from ccdn.models.sparse_mlp import dense_candidate_gradient_scores


class RigLReference(Baseline):
    name = "rigl_reference"

    def __init__(self, model, begin_step=0, end_step=-1, update_freq=100,
                 init_drop_fraction=0.3, schedule="constant",
                 initial_learning_rate=None, grow_init="zeros"):
        super().__init__(model)
        self.begin_step = int(begin_step)
        self.end_step = int(end_step)
        self.update_freq = int(update_freq)
        self.init_drop_fraction = float(init_drop_fraction)
        self.schedule = schedule
        self.last_update_step = -self.update_freq
        self.initial_learning_rate = initial_learning_rate
        self.grow_init = grow_init
        self.rewire_event_count = 0
        self.total_edges_pruned = 0
        self.total_edges_grown = 0
        self.last_pruned = []
        self.last_grown = []
        self.last_drop_fraction = 0.0
        self.structural_event_steps = []
        self.candidate_gradient_elements_scored = 0
        self.edges_ranked = 0

    def drop_fraction(self, step, optimizer=None):
        if self.schedule == "constant":
            return self.init_drop_fraction
        if self.schedule == "cosine":
            if self.end_step <= self.begin_step:
                raise ValueError("cosine schedule requires positive last_update_step")
            # Matches tf.train.learning_rate_decay.cosine_decay in the pinned
            # source, which receives global_step directly as decay progress.
            progress = min(max(step / (self.end_step - self.begin_step), 0.0), 1.0)
            return self.init_drop_fraction * 0.5 * (1 + math.cos(math.pi * progress))
        if self.schedule == "lr":
            if optimizer is None or self.initial_learning_rate is None:
                raise ValueError("lr schedule needs optimizer and initial_learning_rate")
            lr = optimizer.param_groups[0]["lr"]
            return self.init_drop_fraction / self.initial_learning_rate * lr
        raise ValueError(f"unknown RigL schedule: {self.schedule}")

    @torch.no_grad()
    def update_layer(self, layer_index, gradient_scores, drop_fraction, optimizer):
        layer = self.model.layers[layer_index]
        old_mask = layer.mask.flatten().clone()
        n_active = int(old_mask.sum())
        n_prune = int(n_active * drop_fraction)  # tf.cast(float, int32) floors.
        if n_prune == 0:
            return [], []
        flat_weight = layer.weight.flatten()
        self.edges_ranked += n_active
        drop_scores = (flat_weight * old_mask).abs()
        n_keep = n_active - n_prune
        kept = torch.topk(drop_scores, n_keep, sorted=False).indices
        mask1 = torch.zeros_like(old_mask)
        mask1[kept] = True

        grow = gradient_scores.flatten().clone()
        # Match upstream score lifting, including permitting a just-dropped
        # edge to return if it ranks among the gradient's best candidates.
        grow[mask1] = grow.min() - 1
        grown = torch.topk(grow, n_prune, sorted=False).indices
        mask2 = torch.zeros_like(old_mask)
        mask2[grown] = True
        if torch.any(mask1 & mask2):
            raise RuntimeError("RigL keep and grow masks overlap")
        new_connections = mask2 & ~old_mask
        removed = old_mask & ~mask1
        layer.mask.copy_((mask1 | mask2).view_as(layer.mask))
        # Match SparseRigLOptimizerBase: previously inactive grown edges receive
        # grow_init; pruned-and-regrown edges preserve their old value.
        if self.grow_init == "zeros":
            flat_weight[new_connections] = 0
        elif self.grow_init == "random_normal":
            flat_weight[new_connections] = torch.randn_like(flat_weight[new_connections]) * flat_weight.std()
        elif self.grow_init == "random_uniform":
            scale = flat_weight.abs().mean()
            flat_weight[new_connections] = torch.empty_like(flat_weight[new_connections]).uniform_(-float(scale), float(scale))
        else:
            raise ValueError(f"unsupported RigL grow_init: {self.grow_init}")
        clear_optimizer_entries(optimizer, layer.weight,
                                new_connections.view_as(layer.weight))
        if int(layer.mask.sum()) != n_active:
            raise AssertionError("RigL active edge count changed")
        return removed.nonzero().flatten().tolist(), grown.tolist()

    def after_optimizer_step(self, optimizer):
        # The pinned optimizer checks global_step after applying the current
        # gradient; initial last_update_step=-frequency makes begin_step the
        # first eligible update (normally zero).
        step = self.global_step
        super().after_optimizer_step(optimizer)
        is_in_range = step >= self.begin_step and (self.end_step < 0 or step <= self.end_step)
        if (self.update_freq <= 0 or not is_in_range or
                step - self.last_update_step < self.update_freq):
            return
        fraction = self.drop_fraction(step, optimizer)
        self.last_drop_fraction = fraction
        self.last_update_step = step
        if fraction <= 0:
            return
        self.last_pruned, self.last_grown = [], []
        with torch.no_grad():
            for i, layer in enumerate(self.model.layers):
                scores = dense_candidate_gradient_scores(self.model, i)
                self.candidate_gradient_elements_scored += int(scores.numel())
                pruned, grown = self.update_layer(i, scores, fraction, optimizer)
                self.last_pruned.append((i, pruned))
                self.last_grown.append((i, grown))
                self.total_edges_pruned += len(pruned)
                self.total_edges_grown += len(grown)
            self.rewire_event_count += 1
            self.structural_event_steps.append(self.global_step)

    def metrics(self):
        return {"rigl_reference_rewire_events": self.rewire_event_count,
                "rigl_reference_edges_pruned": self.total_edges_pruned,
                "rigl_reference_edges_grown": self.total_edges_grown,
                "topology_events": self.rewire_event_count,
                "edges_pruned": self.total_edges_pruned,
                "edges_grown": self.total_edges_grown,
                "rigl_reference_drop_fraction": self.last_drop_fraction,
                "structural_event_steps": list(self.structural_event_steps),
                "candidate_gradient_elements_scored": self.candidate_gradient_elements_scored,
                "edges_ranked": self.edges_ranked}

    def state_dict(self):
        return {**super().state_dict(), "rewire_event_count": self.rewire_event_count,
                "total_edges_pruned": self.total_edges_pruned,
                "total_edges_grown": self.total_edges_grown,
                "last_pruned": self.last_pruned, "last_grown": self.last_grown,
                "last_drop_fraction": self.last_drop_fraction,
                "last_update_step": self.last_update_step,
                "structural_event_steps": self.structural_event_steps,
                "candidate_gradient_elements_scored": self.candidate_gradient_elements_scored,
                "edges_ranked": self.edges_ranked}

    def load_state_dict(self, state):
        super().load_state_dict(state)
        self.rewire_event_count = int(state["rewire_event_count"])
        self.total_edges_pruned = int(state["total_edges_pruned"])
        self.total_edges_grown = int(state["total_edges_grown"])
        self.last_pruned = [(int(i), list(v)) for i, v in state["last_pruned"]]
        self.last_grown = [(int(i), list(v)) for i, v in state["last_grown"]]
        self.last_drop_fraction = float(state["last_drop_fraction"])
        self.last_update_step = int(state.get("last_update_step", -self.update_freq))
        self.structural_event_steps = [int(x) for x in state.get("structural_event_steps", [])]
        self.candidate_gradient_elements_scored = int(state.get("candidate_gradient_elements_scored", 0))
        self.edges_ranked = int(state.get("edges_ranked", 0))
