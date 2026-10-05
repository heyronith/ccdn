"""PyTorch port of upstream Loss-of-Plasticity Generate-and-Test.

This deliberately follows the pinned upstream ``lop.algos.gnt.GnT`` behavior,
including its eligibility rule (age strictly greater than maturity), fractional
replacement accumulator, adaptable-contribution utility, and reset ordering.
The upstream source clears activation averages before its outgoing-bias
correction reads them; this port preserves that observable behavior for parity.
It is a validation/reference implementation, not the lightweight Cycle 1 CBP.
"""
from __future__ import annotations

from math import sqrt

import torch

from ccdn.baselines.base import Baseline


class ContinualBackpropReference(Baseline):
    name = "continual_backprop_reference"

    def __init__(self, model, replacement_rate=1e-4, decay_rate=0.99,
                 maturity_threshold=100, accumulate=True,
                 util_type="adaptable_contribution", activation="relu",
                 init="kaiming"):
        super().__init__(model)
        self.replacement_rate = float(replacement_rate)
        self.decay_rate = float(decay_rate)
        self.maturity_threshold = int(maturity_threshold)
        self.accumulate = bool(accumulate)
        self.util_type = util_type
        self.hidden_layers = len(model.hidden_sizes)
        device = next(model.parameters()).device
        self.utility = [torch.zeros(n, device=device) for n in model.hidden_sizes]
        self.bias_corrected_utility = [torch.zeros(n, device=device) for n in model.hidden_sizes]
        self.age = [torch.zeros(n, device=device) for n in model.hidden_sizes]
        self.mean_activation = [torch.zeros(n, device=device) for n in model.hidden_sizes]
        self.accumulated_replacements = [0.0 for _ in model.hidden_sizes]
        if activation == "selu":
            init = "lecun"
        self.bounds = self._bounds(activation, init)
        self.replacement_count = 0
        self.last_replaced_units = []
        self._features = None

    def _bounds(self, activation, init):
        if activation in ("swish", "elu"):
            activation = "relu"
        bounds = []
        for layer in self.model.layers[:-1]:
            if init == "default":
                value = sqrt(1 / layer.in_features)
            elif init == "xavier":
                value = torch.nn.init.calculate_gain(activation) * sqrt(
                    6 / (layer.in_features + layer.out_features))
            elif init == "lecun":
                value = sqrt(3 / layer.in_features)
            else:
                value = torch.nn.init.calculate_gain(activation) * sqrt(
                    3 / layer.in_features)
            bounds.append(value)
        bounds.append(sqrt(3 / self.model.layers[-1].in_features))
        return bounds

    def after_backward(self):
        # Official CBP holds the pre-update activations and runs GnT after SGD.
        self._features = [x.detach() for x in self.model.last_activations]

    @torch.no_grad()
    def _update_utility(self, i, features):
        self.utility[i].mul_(self.decay_rate)
        correction = 1 - self.decay_rate ** self.age[i]
        self.mean_activation[i].mul_(self.decay_rate).add_(
            features.mean(0), alpha=1 - self.decay_rate)
        corrected_act = self.mean_activation[i] / correction
        current, following = self.model.layers[i], self.model.layers[i + 1]
        output_mag = following.weight.abs().mean(0)
        input_mag = current.weight.abs().mean(1)
        if self.util_type == "weight":
            value = output_mag
        elif self.util_type == "contribution":
            value = output_mag * features.abs().mean(0)
        elif self.util_type == "adaptation":
            value = 1 / input_mag
        elif self.util_type == "zero_contribution":
            value = output_mag * (features - corrected_act).abs().mean(0)
        elif self.util_type == "adaptable_contribution":
            value = output_mag * (features - corrected_act).abs().mean(0) / input_mag
        elif self.util_type == "feature_by_input":
            value = (features - corrected_act).abs().mean(0) / input_mag
        else:
            value = torch.zeros_like(self.utility[i])
        self.utility[i].add_(value, alpha=1 - self.decay_rate)
        self.bias_corrected_utility[i].copy_(self.utility[i] / correction)
        if self.util_type == "random":
            self.bias_corrected_utility[i].copy_(torch.rand_like(self.utility[i]))

    @torch.no_grad()
    def _replace(self, i, units):
        if units.numel() == 0:
            return
        layer, following = self.model.layers[i], self.model.layers[i + 1]
        layer.weight[units] = torch.empty_like(layer.weight[units]).uniform_(
            -self.bounds[i], self.bounds[i])
        layer.bias[units] = 0
        # Preserve source ordering: the selected mean activations were cleared
        # during test_features before generate_new_features reads this value.
        following.bias.add_(
            (following.weight[:, units] * self.mean_activation[i][units] /
             (1 - self.decay_rate ** self.age[i][units])).sum(1))
        following.weight[:, units] = 0
        self.age[i][units] = 0
        self.replacement_count += units.numel()
        self.last_replaced_units.extend((i, int(u)) for u in units.tolist())

    @torch.no_grad()
    def after_optimizer_step(self, optimizer):
        super().after_optimizer_step(optimizer)
        self.last_replaced_units = []
        if self._features is None:
            return
        replacements = []
        for i, features in enumerate(self._features):
            self.age[i].add_(1)
            self._update_utility(i, features)
            eligible = (self.age[i] > self.maturity_threshold).nonzero().flatten()
            if eligible.numel() == 0 or self.replacement_rate == 0:
                continue
            amount = self.replacement_rate * eligible.numel()
            self.accumulated_replacements[i] += amount
            if self.accumulate:
                count = int(self.accumulated_replacements[i])
                self.accumulated_replacements[i] -= count
            else:
                count = int(amount)
                if amount < 1 and torch.rand(1).item() <= amount:
                    count = 1
            if count == 0:
                continue
            units = eligible[torch.topk(-self.bias_corrected_utility[i][eligible], count).indices]
            self.utility[i][units] = 0
            self.mean_activation[i][units] = 0
            replacements.append((i, units))
        # GnT tests every hidden layer before it mutates any layer weights.
        for i, units in replacements:
            self._replace(i, units)
        self._features = None

    def metrics(self):
        return {"cbp_reference_replacements": self.replacement_count}

    def state_dict(self):
        return {**super().state_dict(), "utility": self.utility,
                "bias_corrected_utility": self.bias_corrected_utility,
                "age": self.age, "mean_activation": self.mean_activation,
                "accumulated_replacements": self.accumulated_replacements,
                "replacement_count": self.replacement_count,
                "last_replaced_units": self.last_replaced_units}

    def load_state_dict(self, state):
        super().load_state_dict(state)
        for name, key in (("utility", "utility"),
                          ("bias_corrected_utility", "bias_corrected_utility"),
                          ("age", "age"), ("mean_activation", "mean_activation")):
            setattr(self, name, [x.clone() for x in state[key]])
        self.accumulated_replacements = list(state["accumulated_replacements"])
        self.replacement_count = int(state["replacement_count"])
        self.last_replaced_units = [tuple(x) for x in state["last_replaced_units"]]
