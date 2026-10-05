import importlib.util
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from ccdn.models.dense_mlp import DenseMLP
from ccdn.models.sparse_mlp import SparseMLP
from ccdn.official_reference.continual_backprop import ContinualBackpropReference
from ccdn.official_reference.rigl import RigLReference


ROOT = Path(__file__).resolve().parents[1]
LOP = ROOT / ".external/official_baselines/loss-of-plasticity"


class OfficialShape(nn.Module):
    def __init__(self, sizes):
        super().__init__()
        blocks = []
        for i, (a, b) in enumerate(zip(sizes[:-1], sizes[1:])):
            blocks.append(nn.Linear(a, b))
            if i < len(sizes) - 2:
                blocks.append(nn.ReLU())
        self.layers = nn.ModuleList(blocks)


def _as_parity_network(dense):
    result = OfficialShape([dense.layers[0].in_features,
                           *(layer.out_features for layer in dense.layers)])
    for src, index in zip(dense.layers, range(0, len(result.layers), 2)):
        result.layers[index].load_state_dict(src.state_dict())
    return result


def test_cbp_reference_matches_pinned_official_gnt_trace():
    if not (LOP / "lop/algos/gnt.py").exists():
        pytest.skip("run scripts/fetch_official_baselines.py for pinned-source parity")
    sys.path.insert(0, str(LOP))
    try:
        from lop.algos.gnt import GnT
    finally:
        sys.path.pop(0)

    torch.manual_seed(2025)
    local = DenseMLP(input_size=4, hidden_sizes=(4, 3), output_size=2)
    official_net = _as_parity_network(local)
    local_opt = torch.optim.SGD(local.parameters(), lr=0.03)
    official_opt = torch.optim.SGD(official_net.parameters(), lr=0.03)
    ref = ContinualBackpropReference(
        local, replacement_rate=0.5, decay_rate=0.8,
        maturity_threshold=1, accumulate=True)
    official = GnT(official_net.layers, "relu", official_opt,
                   decay_rate=0.8, replacement_rate=0.5,
                   maturity_threshold=1, util_type="adaptable_contribution",
                   init="kaiming", accumulate=True)
    x = torch.tensor([[1., 0.2, -0.3, 0.8], [-0.2, 0.4, 0.1, 0.3]])
    y = torch.tensor([0, 1])

    for _ in range(4):
        lo = local(x)
        local_opt.zero_grad(set_to_none=True)
        F.cross_entropy(lo, y).backward()
        ref.after_backward()
        official_features = [official_net.layers[1](official_net.layers[0](x)).relu()]
        official_features.append(official_net.layers[3](official_net.layers[2](
            official_features[0])).relu())
        oo = official_net.layers[4](official_net.layers[3](
            official_net.layers[2](official_net.layers[1](
                official_net.layers[0](x)))))
        official_opt.zero_grad(set_to_none=True)
        F.cross_entropy(oo, y).backward()
        local_opt.step()
        official_opt.step()
        rng_state = torch.random.get_rng_state()
        ref.after_optimizer_step(local_opt)
        local_rng = torch.random.get_rng_state()
        torch.random.set_rng_state(rng_state)
        official.gen_and_test(official_features)
        torch.random.set_rng_state(local_rng)
        for a, b in zip(local.layers, (official_net.layers[0], official_net.layers[2], official_net.layers[4])):
            torch.testing.assert_close(a.weight, b.weight)
            torch.testing.assert_close(a.bias, b.bias)
        for a, b in zip(ref.utility, official.util):
            torch.testing.assert_close(a, b)
        for a, b in zip(ref.age, official.ages):
            torch.testing.assert_close(a, b)
        for a, b in zip(ref.mean_activation, official.mean_feature_act):
            torch.testing.assert_close(a, b)
        assert ref.accumulated_replacements == pytest.approx(
            official.accumulated_num_features_to_replace)


def test_rigl_reference_conserves_budget_and_clears_momentum():
    model = SparseMLP(input_size=8, hidden_sizes=(4,), output_size=2,
                      density=0.5, seed=3)
    opt = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    layer = model.layers[0]
    with torch.no_grad():
        layer.weight[layer.mask] = torch.linspace(0.1, 0.8, int(layer.mask.sum()))
    for current in model.layers:
        opt.state[current.weight]["momentum_buffer"] = torch.ones_like(current.weight)
    rigl = RigLReference(model, update_freq=1, init_drop_fraction=0.25,
                         schedule="constant")
    counts = [int(x.mask.sum()) for x in model.layers]
    scores = [torch.arange(l.weight.numel(), dtype=l.weight.dtype).reshape_as(l.weight)
              for l in model.layers]
    for i, (l, score) in enumerate(zip(model.layers, scores)):
        old_mask = l.mask.clone()
        pruned, grown = rigl.update_layer(i, score, 0.25, opt)
        assert len(pruned) == len(grown)
        assert int(l.mask.sum()) == counts[i]
        newly_grown = l.mask & ~old_mask
        assert int(newly_grown.sum()) <= len(grown)
        assert torch.count_nonzero(opt.state[l.weight]["momentum_buffer"][newly_grown]) == 0
