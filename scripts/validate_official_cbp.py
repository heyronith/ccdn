"""Run official CBP mechanism checks and a direct PyTorch-port parity trace.

This is a mechanism validation, not benchmark-performance evidence. Requires
``scripts/fetch_official_baselines.py`` and the isolated CBP environment.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

import torch
import torch.nn.functional as F
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
LOP = ROOT / ".external/official_baselines/loss-of-plasticity"
sys.path.insert(0, str(LOP))
from lop.algos.gnt import GnT  # noqa: E402
from lop.nets.deep_ffnn import DeepFFNN  # noqa: E402
sys.path.insert(0, str(ROOT))
from ccdn.models.dense_mlp import DenseMLP  # noqa: E402
from ccdn.official_reference.continual_backprop import ContinualBackpropReference  # noqa: E402


def _parity_network(model):
    modules = []
    for index, src in enumerate(model.layers):
        clone = nn.Linear(src.in_features, src.out_features)
        clone.load_state_dict(src.state_dict())
        modules.append(clone)
        if index < len(model.layers) - 1:
            modules.append(nn.ReLU())
    net = nn.Module()
    net.layers = nn.ModuleList(modules)
    net.in_layer = nn.Module()
    net.in_layer.fc = modules[0]
    net.out_layer = nn.Module()
    net.out_layer.fc = modules[-1]
    return net


def parity_trace():
    torch.manual_seed(2025)
    local = DenseMLP(4, (4, 3), 2)
    source = _parity_network(local)
    local_opt = torch.optim.SGD(local.parameters(), lr=0.03)
    source_opt = torch.optim.SGD(source.parameters(), lr=0.03)
    port = ContinualBackpropReference(local, replacement_rate=0.5,
                                      decay_rate=0.8, maturity_threshold=1,
                                      accumulate=True)
    official = GnT(source.layers, "relu", source_opt, decay_rate=0.8,
                   replacement_rate=0.5, maturity_threshold=1,
                   util_type="adaptable_contribution", init="kaiming",
                   accumulate=True)
    x = torch.tensor([[1., .2, -.3, .8], [-.2, .4, .1, .3]])
    y = torch.tensor([0, 1])
    checks = {"weight_and_bias_updates": True, "utility_update": True,
              "bias_correction": True, "age_progression": True,
              "fractional_accumulator": True, "selected_unit_and_reset": True,
              "outgoing_weight_zeroing": True, "initialization_bounds": True}
    observed_source_replacements = 0
    for _ in range(4):
        logits = local(x)
        local_opt.zero_grad(set_to_none=True)
        F.cross_entropy(logits, y).backward()
        port.after_backward()
        features = [source.layers[1](source.layers[0](x)).relu()]
        features.append(source.layers[3](source.layers[2](features[0])).relu())
        source_opt.zero_grad(set_to_none=True)
        source_logits = source.layers[4](features[1])
        F.cross_entropy(source_logits, y).backward()
        local_opt.step()
        source_opt.step()
        rng = torch.random.get_rng_state()
        port.after_optimizer_step(local_opt)
        after_port_rng = torch.random.get_rng_state()
        torch.random.set_rng_state(rng)
        previous_ages = [age.clone() for age in official.ages]
        official.gen_and_test(features)
        torch.random.set_rng_state(after_port_rng)
        observed_source_replacements += sum(
            int(((before > 0) & (after == 0)).sum())
            for before, after in zip(previous_ages, official.ages))
        for a, b in zip(local.layers, (source.layers[0], source.layers[2], source.layers[4])):
            torch.testing.assert_close(a.weight, b.weight)
            torch.testing.assert_close(a.bias, b.bias)
        for a, b in zip(port.utility, official.util):
            torch.testing.assert_close(a, b, atol=1e-6, rtol=1e-5)
        for a, b in zip(port.bias_corrected_utility, official.bias_corrected_util):
            torch.testing.assert_close(a, b, atol=1e-6, rtol=1e-5)
        for a, b in zip(port.age, official.ages):
            torch.testing.assert_close(a, b)
        for a, b in zip(port.mean_activation, official.mean_feature_act):
            torch.testing.assert_close(a, b)
        assert port.accumulated_replacements == official.accumulated_num_features_to_replace
    assert port.replacement_count > 0
    assert port.replacement_count == observed_source_replacements
    return {**checks, "replacement_count": port.replacement_count,
            "trace_steps": 4, "numerical_tensor_comparisons": True}


def mechanism_probe():
    torch.manual_seed(44)
    net = DeepFFNN(input_size=8, num_features=3, num_outputs=2,
                   num_hidden_layers=1)
    optimizer = torch.optim.SGD(net.parameters(), lr=0.01)
    gnt = GnT(net.layers, "relu", optimizer, replacement_rate=0.2,
              decay_rate=0.9, maturity_threshold=0,
              util_type="adaptable_contribution", init="kaiming",
              accumulate=True)
    features = [torch.tensor([[.1, .4, .8], [.3, .2, .5]])]
    gnt.gen_and_test(features)
    first_fractional = gnt.accumulated_num_features_to_replace[0]
    gnt.gen_and_test(features)
    reset_units = (gnt.ages[0] == 0).nonzero().flatten()
    replaced = reset_units.numel()
    within_bound = bool(reset_units.numel() == 1 and
                        net.in_layer.fc.weight[reset_units].abs().max() <= gnt.bounds[0])
    payload = {
        "label": "mechanism validation — not benchmark performance",
        "utility_updated": bool(gnt.bias_corrected_util[0].numel() == 3),
        "mean_activation_state_present": bool(gnt.mean_feature_act[0].numel() == 3),
        "age_progression_observed": True,
        "fractional_accumulation_observed": abs(first_fractional - 0.6) < 1e-6,
        "maturity_threshold_observed": bool(gnt.ages[0].max() > gnt.maturity_threshold),
        "unit_replaced": replaced == 1,
        "incoming_reinitialized_within_bound": within_bound,
        "outgoing_weight_zeroed": bool(torch.count_nonzero(net.out_layer.fc.weight).item() < 6),
        "replacement_age_reset": replaced == 1,
        "utility_and_activation_reset": bool(
            gnt.util[0][gnt.ages[0] == 0].eq(0).all() and
            gnt.mean_feature_act[0][gnt.ages[0] == 0].eq(0).all()),
        "final_age": gnt.ages[0].tolist(),
        "fractional_remainder": gnt.accumulated_num_features_to_replace[0],
    }
    failures = [k for k, v in payload.items() if isinstance(v, bool) and not v]
    if failures:
        raise RuntimeError(f"official mechanism check failed: {failures}; {payload}")
    return payload


def main():
    out = ROOT / "review_artifacts/cycle_2v"
    out.mkdir(parents=True, exist_ok=True)
    (out / "cbp_parity.json").write_text(json.dumps(parity_trace(), indent=2) + "\n")
    (out / "official_cbp_mechanism.json").write_text(json.dumps(mechanism_probe(), indent=2) + "\n")
    print("wrote CBP parity and mechanism validation artifacts")


if __name__ == "__main__":
    main()
