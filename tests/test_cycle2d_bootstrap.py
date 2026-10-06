import copy
import json
from pathlib import Path

import pytest
import torch
import yaml

from scripts.bootstrap_cycle2d import (
    CYCLE2C_EXECUTION_SHA, CYCLE2C_SOURCE_PROTOCOL_HASH, EXPECTED_EDGES,
    EXPECTED_STREAM, bootstrap_inherited_state, build_cycle2d_config,
    file_sha256, load_source_checkpoint, validate_source_identity,
    validate_static_checkpoint_tensors,
)


def fixture_source():
    config = {
        "experiment": {"name": "cycle2c_fixture", "seed": 101, "deterministic": True},
        "stream": {"dataset_root": "./data", "download": False,
                   "tasks": 150, "examples_per_task": 60000},
        "model": {"type": "static_sparse", "hidden_sizes": [336, 336, 336],
                  "output_size": 10, "density": .2,
                  "initialization": "active_fan_in_kaiming"},
        "optimizer": {"type": "sgd", "learning_rate": .003,
                      "momentum": 0.0, "weight_decay": 0.0},
        "runtime": {"expected_task_sequence_sha256": EXPECTED_STREAM,
                    "status_path": "/fixture/status.json"},
    }
    manifest = {"git_commit": CYCLE2C_EXECUTION_SHA,
                "reviewer_approved_git_sha": CYCLE2C_EXECUTION_SHA,
                "git_branch": "cycle-2c-ccdn-failure-regime", "git_dirty": False,
                "protocol_hash": CYCLE2C_SOURCE_PROTOCOL_HASH,
                "task_sequence_sha256": EXPECTED_STREAM, "data_source": "mnist",
                "synthetic_fallback": False, "seed": 101,
                "architecture": [784, 336, 336, 336, 10], "density": .2,
                "logical_active_parameters": 99533,
                "active_edges_by_layer": EXPECTED_EDGES}
    summary = {"learner_type": "static_sparse", "tasks_requested": 150,
               "tasks_completed": 150, "total_optimizer_updates": 9000000,
               "loss_of_plasticity_gate_passed": False,
               "data_source": "mnist", "synthetic_fallback": False,
               "architecture": [784, 336, 336, 336, 10],
               "initialization": "active_fan_in_kaiming", "learning_rate": .003,
               "momentum": 0.0, "weight_decay": 0.0}
    status = {"state": "STATIC_GATE_FAILED", "summary": copy.deepcopy(summary)}
    masks, model = [], {}
    for i, edge_count in enumerate(EXPECTED_EDGES):
        mask = torch.ones(edge_count, dtype=torch.bool)
        masks.append(mask)
        model[f"layers.{i}.mask"] = mask.clone()
        model[f"layers.{i}.weight"] = torch.zeros(edge_count)
        model[f"layers.{i}.bias"] = torch.zeros(1)
    checkpoint = {"model": model, "optimizer": {"state": {}, "param_groups": [
                    {"lr": .003, "momentum": 0.0, "weight_decay": 0.0}]},
                  "algorithm": {"global_step": 9000000}, "rng_state": {},
                  "completed_task_index": 149, "lifetime_examples_seen": 9000000,
                  "config": config, "git_sha": CYCLE2C_EXECUTION_SHA,
                  "protocol_hash": CYCLE2C_SOURCE_PROTOCOL_HASH, "method": "static_sparse",
                  "task_sequence_sha256": EXPECTED_STREAM}
    tasks = [{"task_index": i, "examples_seen_in_task": 60000,
              "optimizer_updates": 60000,
              "lifetime_examples_seen": (i+1)*60000,
              "correct_predictions": 48000,
              "online_accuracy": .8} for i in range(150)]
    diagnostics = [{"task_index": i, "phase": "task_start",
                    "lifetime_examples_seen": i*60000,
                    "dead_unit_fraction_overall": 0.0,
                    "mean_absolute_weight": 0.1,
                    "effective_rank_layer_0": 1.0,
                    "effective_rank_layer_1": 1.0,
                    "effective_rank_layer_2": 1.0,
                    "initial_mask_overlap_fraction_overall": 1.0,
                    **{f"active_edges_layer_{j}": count for j, count in enumerate(EXPECTED_EDGES)}
                    } for i in range(0, 150, 5)]
    diagnostics.append({"task_index": 149, "phase": "final",
                        "lifetime_examples_seen": 9000000,
                        "active_edges_layer_0": EXPECTED_EDGES[0],
                        "active_edges_layer_1": EXPECTED_EDGES[1],
                        "active_edges_layer_2": EXPECTED_EDGES[2],
                        "active_edges_layer_3": EXPECTED_EDGES[3],
                        "initial_mask_overlap_fraction_overall": 1.0,
                        "dead_unit_fraction_overall": 0.0,
                        "mean_absolute_weight": 0.1,
                        "effective_rank_layer_0": 1.0,
                        "effective_rank_layer_1": 1.0,
                        "effective_rank_layer_2": 1.0})
    return manifest, status, summary, checkpoint, tasks, diagnostics, config, masks


def validate_fixture(parts):
    manifest, status, summary, checkpoint, tasks, diagnostics, config, _masks = parts
    return validate_source_identity(manifest, status, summary, checkpoint, tasks,
                                    diagnostics, config, CYCLE2C_SOURCE_PROTOCOL_HASH,
                                    EXPECTED_STREAM)


def test_valid_source_run_identity_passes():
    assert validate_fixture(fixture_source())


@pytest.mark.parametrize("field,value", [
    ("git_commit", "0"*40),
    ("protocol_hash", "wrong"),
    ("task_sequence_sha256", "wrong"),
    ("data_source", "synthetic"),
    ("synthetic_fallback", True),
    ("git_branch", "main"),
    ("seed", 4),
    ("logical_active_parameters", 1),
])
def test_wrong_source_manifest_identity_rejected(field, value):
    parts = fixture_source()
    parts[0][field] = value
    with pytest.raises(RuntimeError): validate_fixture(parts)


@pytest.mark.parametrize("field,value", [
    ("method", "rigl_reference"),
    ("completed_task_index", 148),
    ("lifetime_examples_seen", 8999999),
    ("git_sha", "wrong"),
    ("protocol_hash", "wrong"),
    ("task_sequence_sha256", "wrong"),
])
def test_wrong_source_checkpoint_identity_rejected(field, value):
    parts = fixture_source()
    parts[3][field] = value
    with pytest.raises(RuntimeError): validate_fixture(parts)


@pytest.mark.parametrize("field,value", [
    ("lr", .01), ("momentum", .9), ("weight_decay", .1),
])
def test_wrong_optimizer_checkpoint_state_rejected(field, value):
    parts = fixture_source()
    parts[3]["optimizer"]["param_groups"][0][field] = value
    with pytest.raises(RuntimeError, match="checkpoint optimizer state"):
        validate_fixture(parts)


@pytest.mark.parametrize("section,field,value", [
    ("model", "hidden_sizes", [512, 512, 512]),
    ("optimizer", "learning_rate", .01),
    ("optimizer", "momentum", .9),
    ("optimizer", "weight_decay", .1),
])
def test_changed_architecture_or_optimizer_rejected(section, field, value):
    parts = fixture_source()
    parts[6][section][field] = value
    parts[3]["config"] = parts[6]
    with pytest.raises(RuntimeError): validate_fixture(parts)


def test_gate_status_must_be_failed_and_method_must_be_static_sparse():
    parts = fixture_source()
    parts[1]["state"] = "COMPLETE"
    with pytest.raises(RuntimeError, match="STATIC_GATE_FAILED"):
        validate_fixture(parts)
    parts = fixture_source()
    parts[2]["learner_type"] = "ccdn_0a"
    with pytest.raises(RuntimeError, match="Static Sparse"):
        validate_fixture(parts)


def test_incomplete_task_or_diagnostic_history_rejected():
    parts = fixture_source()
    parts[4].pop()
    with pytest.raises(RuntimeError, match="task CSV"):
        validate_fixture(parts)
    parts = fixture_source()
    parts[5].pop(0)
    with pytest.raises(RuntimeError, match="diagnostic history"):
        validate_fixture(parts)


def test_altered_static_mask_rejected():
    parts = fixture_source()
    masks = parts[7]
    altered = dict(parts[3])
    altered["model"] = dict(parts[3]["model"])
    altered["model"]["layers.0.mask"] = altered["model"]["layers.0.mask"].clone()
    altered["model"]["layers.0.mask"][0] = False
    with pytest.raises(RuntimeError, match="mask changed"):
        validate_static_checkpoint_tensors(altered, masks)


def test_nonfinite_static_checkpoint_weights_rejected():
    parts = fixture_source()
    altered = dict(parts[3])
    altered["model"] = dict(parts[3]["model"])
    altered["model"]["layers.0.weight"] = altered["model"]["layers.0.weight"].clone()
    altered["model"]["layers.0.weight"][0] = float("nan")
    with pytest.raises(RuntimeError, match="non-finite weights"):
        validate_static_checkpoint_tensors(altered, parts[7])


def test_invalid_source_checkpoint_file_rejected(tmp_path):
    path = tmp_path / "bad.pt"
    path.write_bytes(b"not a torch checkpoint")
    with pytest.raises(RuntimeError, match="invalid Cycle 2C source checkpoint"):
        load_source_checkpoint(path)


def test_bootstrap_preserves_all_state_and_history_without_training(tmp_path):
    from ccdn.experiments.online_permuted_mnist import _build_online_learner
    from ccdn.streams.online_permuted_mnist import OnlineTask
    from ccdn.utils.reproducibility import capture_rng_state, seed_everything

    class Stream:
        input_size = 784
        def task(self, index):
            return OnlineTask(index, torch.arange(784), torch.arange(60000))

    stream = Stream()
    source_config = fixture_source()[6]
    source_config["experiment"].update({"device": "cpu", "deterministic": True, "cpu_threads": 1})
    source_config["model"].update({"input_size": 784})
    seed_everything(101, True)
    model, algorithm, optimizer = _build_online_learner(source_config, stream, torch.device("cpu"))
    algorithm.global_step = 9000000
    masks = [layer.mask.detach().clone() for layer in model.layers]
    checkpoint = {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                  "algorithm": algorithm.state_dict(), "rng_state": capture_rng_state(),
                  "completed_task_index": 149, "lifetime_examples_seen": 9000000,
                  "config": source_config, "runtime_seconds": 12.0}
    source_result = tmp_path / "source" / "static_sparse" / "result"
    source_result.mkdir(parents=True)
    historical_tasks = "task_index,online_accuracy\n" + "".join(
        f"{i},{i/1000:.3f}\n" for i in range(150))
    historical_diagnostics = "task_index,phase\n" + "".join(
        f"{i},task_start\n" for i in range(0, 150, 5)) + "149,final\n"
    (source_result / "task_accuracy.csv").write_text(historical_tasks)
    (source_result / "diagnostics.csv").write_text(historical_diagnostics)
    checkpoint_path = source_result / "checkpoint_latest.pt"
    torch.save(checkpoint, checkpoint_path)
    source_sha = file_sha256(checkpoint_path)
    source = {"run_root": str(tmp_path / "source"), "checkpoint_path": str(checkpoint_path),
              "checkpoint_sha256": source_sha, "checkpoint": checkpoint,
              "manifest": {"run_id": "source-run", "protocol_hash": CYCLE2C_SOURCE_PROTOCOL_HASH},
              "initial_masks": masks}
    digests = {800: "d"*64}
    config = build_cycle2d_config(source_config, digests[800], "g"*40, "p"*64)
    dest = tmp_path / "cycle2d" / "result"
    proof = bootstrap_inherited_state(source, dest, config=config, stream=stream,
                                      stream_hashes=digests, git_sha="g"*40,
                                      protocol_hash="p"*64)
    inherited = torch.load(dest / "checkpoint_latest.pt", map_location="cpu", weights_only=False)
    assert proof["training_updates_during_bootstrap"] == 0
    assert proof["next_task_index"] == 150 and proof["next_task_human_number"] == 151
    assert proof["source_checkpoint_bytes_unchanged"]
    assert proof["model_state_exact"] and proof["mask_state_exact"]
    assert proof["optimizer_state_exact"] and proof["algorithm_state_exact"]
    assert proof["rng_state_exact"]
    assert file_sha256(checkpoint_path) == source_sha
    assert inherited["algorithm"]["global_step"] == 9000000
    assert inherited["completed_task_index"] == 149
    assert all(torch.equal(inherited["model"][key], checkpoint["model"][key])
               for key in checkpoint["model"])
    assert (dest / "task_accuracy.csv").read_bytes() == (source_result / "task_accuracy.csv").read_bytes()
    assert (dest / "diagnostics.csv").read_bytes() == (source_result / "diagnostics.csv").read_bytes()
