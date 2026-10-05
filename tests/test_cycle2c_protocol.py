import inspect
import json

import torch

from ccdn.algorithms.ccdn_0a import CCDN0A
from ccdn.baselines.selective_reset import SelectiveReset
from ccdn.experiments.online_permuted_mnist import _build_online_learner, _learner_step
from ccdn.models.sparse_mlp import SparseMLP
from ccdn.official_reference.rigl import RigLReference
from ccdn.utils.reproducibility import seed_everything
from ccdn.experiments.online_permuted_mnist import run_online_experiment


class TinyStream:
    input_size = 8


def make_cfg(kind):
    cfg = {
        "experiment": {"seed": 101},
        "model": {"type": kind, "hidden_sizes": [4, 4, 4], "output_size": 3,
                  "density": .5, "initialization": "active_fan_in_kaiming"},
        "optimizer": {"learning_rate": .003, "momentum": 0, "weight_decay": 0},
        "algorithm": {},
        "runtime": {"strict_structural_cadence": True},
    }
    if kind == "selective_reset":
        cfg["algorithm"] = {"reset_interval": 8, "reset_fraction": .25,
                            "selection_scope": "per_layer", "utility_decay": .9999214848336215}
    elif kind == "rigl_reference":
        cfg["algorithm"] = {"begin_step": 7, "update_freq": 8, "init_drop_fraction": .25,
                            "schedule": "constant", "grow_init": "zeros"}
    elif kind == "ccdn_0a":
        cfg["algorithm"] = {"structural_interval": 8, "turnover_fraction": .25,
                            "utility_decay": .9999214848336215}
    return cfg


def build(kind):
    seed_everything(101, True)
    return _build_online_learner(make_cfg(kind), TinyStream(), torch.device("cpu"))


def test_cycle2c_sparse_methods_have_identical_starting_state():
    hashes = []
    for kind in ("static_sparse", "selective_reset", "rigl_reference", "ccdn_0a"):
        model, _, _ = build(kind)
        hashes.append({key: value.clone() for key, value in model.state_dict().items()})
    for state in hashes[1:]:
        assert state.keys() == hashes[0].keys()
        assert all(torch.equal(state[key], hashes[0][key]) for key in state)


def test_frozen_sparse_capacity_is_exactly_99533_parameters():
    model = SparseMLP(input_size=784, hidden_sizes=(336, 336, 336), output_size=10,
                      density=.2, seed=101, initialization="active_fan_in_kaiming")
    assert [int(layer.mask.sum()) for layer in model.layers] == [52685, 22579, 22579, 672]
    assert model.active_count == 99533


def test_no_task_boundary_is_exposed_to_learner_step():
    assert list(inspect.signature(_learner_step).parameters) == ["model", "algorithm", "optimizer", "x", "y"]


def test_structural_methods_align_on_completed_update_8192_equivalent():
    kinds = ("static_sparse", "selective_reset", "rigl_reference", "ccdn_0a")
    before = None
    algs = {}
    models = {}
    opts = {}
    for kind in kinds:
        models[kind], algs[kind], opts[kind] = build(kind)
    for step in range(8):
        x = torch.arange(8, dtype=torch.float32).reshape(1, 8) / 8
        y = torch.tensor([step % 3])
        for kind in kinds:
            _learner_step(models[kind], algs[kind], opts[kind], x, y)
        if step == 6:
            before = {kind: [p.detach().clone() for p in models[kind].parameters()] for kind in kinds}
            for kind in kinds[1:]:
                assert all(torch.equal(a, b) for a, b in zip(before[kind], before["static_sparse"]))
    assert algs["selective_reset"].event_steps == [8]
    assert algs["rigl_reference"].structural_event_steps == [8]
    assert algs["ccdn_0a"].structural_event_steps == [8]
    expected = [int(layer.mask.sum()) for layer in models["static_sparse"].layers]
    for kind in kinds:
        assert [int(layer.mask.sum()) for layer in models[kind].layers] == expected


def test_selective_reset_active_fan_in_scale_and_mask_are_preserved():
    model = SparseMLP(input_size=8, hidden_sizes=(4, 4, 4), output_size=3,
                      density=.5, seed=101, initialization="active_fan_in_kaiming")
    algorithm = SelectiveReset(model, reset_interval=1, reset_fraction=.25,
                               utility_decay=1.0, selection_scope="per_layer")
    initial_masks = [layer.mask.clone() for layer in model.layers]
    optimizer = torch.optim.SGD(model.parameters(), lr=.01, momentum=.9)
    x, y = torch.randn(2, 8), torch.tensor([0, 1])
    optimizer.zero_grad(); torch.nn.functional.cross_entropy(model(x), y).backward()
    algorithm.after_backward(); optimizer.step(); algorithm.after_optimizer_step(optimizer)
    assert all(torch.equal(layer.mask, mask) for layer, mask in zip(model.layers, initial_masks))
    assert algorithm.event_steps == [1]
    for layer in model.layers:
        assert torch.equal(layer.weight[~layer.mask], torch.zeros_like(layer.weight[~layer.mask]))
        for row in range(layer.out_features):
            bound = (3 ** .5) * (1 if layer.is_output else 2 ** .5) / max(int(layer.mask[row].sum()), 1) ** .5
            assert float(layer.weight[row].detach().abs().max()) <= bound + 1e-6


def test_ccdn_rolling_checkpoint_resume_matches_continuous_run(tmp_path):
    class Stream:
        data_source = "injected_test_data"
        input_size = 8
        images = torch.arange(80, dtype=torch.float32).reshape(10, 8) / 80
        labels = torch.arange(10) % 3
        metadata = {"fixture": True}
        def task(self, index):
            from ccdn.streams.online_permuted_mnist import OnlineTask
            return OnlineTask(index, torch.arange(8), torch.arange(10))
        def sample(self, task, pos, device="cpu"):
            return self.images[pos:pos+1].to(device), self.labels[pos:pos+1].to(device)
        def diagnostic_batch(self, task, count=2000, device="cpu"):
            return self.images[:count].to(device), self.labels[:count].to(device)
        def sequence_digest(self, tasks, examples):
            return "fixture-stream"

    config = make_cfg("ccdn_0a")
    config["stream"] = {"tasks": 3, "examples_per_task": 10}
    config["diagnostics"] = {"examples": 5}
    config["checkpointing"] = {"rolling": True}
    config["protocol_hash"] = "fixture-protocol"
    config["git_sha"] = "fixture-sha"
    continuous = tmp_path / "continuous"
    interrupted = tmp_path / "interrupted"
    run_online_experiment(config, stream=Stream(), output_dir=continuous)
    run_online_experiment(config, stream=Stream(), output_dir=interrupted,
                          max_tasks_this_invocation=1)
    checkpoint = interrupted / "checkpoint_latest.pt"
    run_online_experiment(config, stream=Stream(), output_dir=interrupted,
                          resume_from=checkpoint)
    a = json.loads((continuous / "summary.json").read_text())
    b = json.loads((interrupted / "summary.json").read_text())
    assert a["tasks_completed"] == b["tasks_completed"] == 3
    assert a["task_sequence_sha256"] == b["task_sequence_sha256"] == "fixture-stream"
    assert a["final_replacement_state"] == b["final_replacement_state"]
    final_a = torch.load(continuous / "checkpoint_latest.pt", weights_only=False)
    final_b = torch.load(interrupted / "checkpoint_latest.pt", weights_only=False)
    assert all(torch.equal(final_a["model"][k], final_b["model"][k]) for k in final_a["model"])
    assert final_a["algorithm"]["global_step"] == final_b["algorithm"]["global_step"] == 30
