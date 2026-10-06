#!/usr/bin/env python3
"""Explicit, no-training inheritance of the completed Cycle 2C Static Sparse state."""
from __future__ import annotations

import copy
import csv
import hashlib
import json
import math
import shutil
from datetime import datetime, timezone
from pathlib import Path

import torch
import yaml

from ccdn.experiments.online_permuted_mnist import (
    _build_online_learner, _finite_tree,
)
from ccdn.models.sparse_mlp import SparseMLP
from ccdn.streams.online_permuted_mnist import OnlinePermutedMNIST
from ccdn.utils.reproducibility import capture_rng_state, restore_rng_state, seed_everything
from scripts.run_cycle2c import EXPECTED_STREAM, file_hashes as cycle2c_file_hashes, verify_lock_file

CYCLE2C_EXECUTION_SHA = "5b48752c3df050d47aae83cba49d08486ad837aa"
CYCLE2C_SOURCE_PROTOCOL_HASH = "019e306798b19183d1b4ee32a7c3ee200ad176f100c55900c97a275eb8f01cb1"
EXPECTED_EDGES = [52685, 22579, 22579, 672]
EXPECTED_ARCH = [784, 336, 336, 336, 10]
EXPECTED_ACTIVE_PARAMETERS = 99533
HORIZONS = [200, 250, 300, 400, 600, 800]
ROOT = Path(__file__).resolve().parents[1]


def csv_rows(path):
    with Path(path).open(newline="", encoding="utf8") as handle:
        return list(csv.DictReader(handle))


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_source_checkpoint(path):
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:
        raise RuntimeError(f"invalid Cycle 2C source checkpoint: {exc}") from exc
    required = {"model", "optimizer", "algorithm", "rng_state", "completed_task_index",
                "lifetime_examples_seen", "config", "git_sha", "protocol_hash",
                "method", "task_sequence_sha256"}
    if not isinstance(payload, dict) or not required.issubset(payload):
        raise RuntimeError("invalid Cycle 2C source checkpoint: required state is missing")
    return payload


def task_sequence_digests(stream, horizons=HORIZONS, examples_per_task=60000):
    """Compute all preregistered prefix hashes in one deterministic stream pass."""
    requested = sorted(set(int(horizon) for horizon in horizons))
    if not requested or requested[0] <= 0:
        raise ValueError("horizons must be positive")
    digest = hashlib.sha256()
    result = {}
    for task_index in range(requested[-1]):
        task = stream.task(task_index)
        digest.update(task_index.to_bytes(8, "little", signed=False))
        digest.update(task.permutation.numpy().tobytes())
        digest.update(task.order[:examples_per_task].numpy().tobytes())
        if task_index + 1 in requested:
            result[task_index + 1] = digest.copy().hexdigest()
    return result


def _nested_equal(left, right):
    if torch.is_tensor(left) or torch.is_tensor(right):
        return torch.is_tensor(left) and torch.is_tensor(right) and torch.equal(left, right)
    if isinstance(left, dict) or isinstance(right, dict):
        return (isinstance(left, dict) and isinstance(right, dict) and left.keys() == right.keys()
                and all(_nested_equal(left[key], right[key]) for key in left))
    if isinstance(left, (list, tuple)) or isinstance(right, (list, tuple)):
        return (isinstance(left, (list, tuple)) and isinstance(right, (list, tuple))
                and len(left) == len(right)
                and all(_nested_equal(a, b) for a, b in zip(left, right)))
    if hasattr(left, "shape") and hasattr(right, "shape"):
        try:
            import numpy as np
            return bool(np.array_equal(left, right))
        except Exception:
            return False
    return left == right


def _check(condition, message):
    if not condition:
        raise RuntimeError(message)


def _validate_source_config(config):
    _check(config.get("experiment", {}).get("seed") == 101, "source seed mismatch")
    _check(config.get("experiment", {}).get("deterministic") is True, "source deterministic mode is disabled")
    _check(config.get("stream", {}).get("tasks") == 150, "source task horizon mismatch")
    _check(config.get("stream", {}).get("examples_per_task") == 60000, "source examples/task mismatch")
    _check(config.get("stream", {}).get("download") is False, "source config permits a dataset download/fallback")
    _check(config.get("runtime", {}).get("expected_task_sequence_sha256") == EXPECTED_STREAM,
           "source config does not freeze the approved 150-task stream")
    model = config.get("model", {})
    _check(model.get("type") == "static_sparse", "source method is not static_sparse")
    _check(model.get("hidden_sizes") == [336, 336, 336] and model.get("output_size") == 10,
           "source architecture mismatch")
    _check(model.get("density") == 0.2 and model.get("initialization") == "active_fan_in_kaiming",
           "source sparse initialization/config mismatch")
    optimizer = config.get("optimizer", {})
    _check(optimizer.get("type", "sgd").lower() == "sgd"
           and optimizer.get("learning_rate") == 0.003
           and optimizer.get("momentum") == 0.0
           and optimizer.get("weight_decay") == 0.0,
           "source optimizer configuration mismatch")


def validate_static_checkpoint_tensors(checkpoint, initial_masks):
    model_state = checkpoint.get("model", {})
    counts = []
    for index, initial_mask in enumerate(initial_masks):
        weight_key = f"layers.{index}.weight"
        mask_key = f"layers.{index}.mask"
        _check(weight_key in model_state and mask_key in model_state,
               f"checkpoint is missing sparse layer {index}")
        weight = model_state[weight_key]
        mask = model_state[mask_key].bool().cpu()
        start = initial_mask.bool().cpu()
        _check(torch.equal(mask, start), f"Static Sparse mask changed in layer {index}")
        _check(not bool((weight[~mask.to(weight.device)] != 0).any()),
               f"inactive weights are nonzero in layer {index}")
        _check(bool(torch.isfinite(weight).all()), f"non-finite weights in layer {index}")
        _check(bool(torch.isfinite(model_state[f"layers.{index}.bias"]).all()),
               f"non-finite bias in layer {index}")
        counts.append(int(mask.sum()))
    _check(counts == EXPECTED_EDGES, "checkpoint active edge count mismatch")
    return counts


def validate_source_identity(manifest, status, summary, checkpoint, task_rows,
                             diagnostics, source_config, source_protocol_hash,
                             task_sequence_sha256, *, source_sha=CYCLE2C_EXECUTION_SHA):
    """Validate all scientific identities and completed-task metadata before inheritance."""
    _check(manifest.get("git_commit") == source_sha, "wrong source Git SHA")
    _check(manifest.get("reviewer_approved_git_sha") == source_sha, "wrong reviewer-approved source SHA")
    _check(manifest.get("git_branch") == "cycle-2c-ccdn-failure-regime" and not manifest.get("git_dirty"),
           "source branch/worktree identity mismatch")
    _check(manifest.get("protocol_hash") == source_protocol_hash == CYCLE2C_SOURCE_PROTOCOL_HASH,
           "wrong source protocol hash")
    _check(manifest.get("task_sequence_sha256") == task_sequence_sha256 == EXPECTED_STREAM,
           "wrong source task-stream hash")
    _check(manifest.get("data_source") == "mnist" and manifest.get("synthetic_fallback") is False,
           "source run is not verified real MNIST")
    _check(manifest.get("seed") == 101 and manifest.get("architecture") == EXPECTED_ARCH
           and manifest.get("density") == 0.2
           and manifest.get("logical_active_parameters") == EXPECTED_ACTIVE_PARAMETERS,
           "source manifest architecture/resource identity mismatch")
    _check(manifest.get("active_edges_by_layer") == EXPECTED_EDGES, "source manifest edge budget mismatch")
    _check(status.get("state") == "STATIC_GATE_FAILED", "source outcome is not STATIC_GATE_FAILED")
    _check(summary.get("learner_type") == "static_sparse" and summary.get("tasks_requested") == 150
           and summary.get("tasks_completed") == 150 and summary.get("total_optimizer_updates") == 9000000,
           "source summary does not describe the completed Static Sparse run")
    _check(summary.get("architecture") == EXPECTED_ARCH and summary.get("initialization") == "active_fan_in_kaiming"
           and summary.get("learning_rate") == 0.003 and summary.get("momentum") == 0.0
           and summary.get("weight_decay") == 0.0,
           "source summary learner/optimizer values differ from the frozen Cycle 2C learner")
    _check(summary.get("loss_of_plasticity_gate_passed") is False
           and status.get("summary", {}).get("loss_of_plasticity_gate_passed") is False,
           "source summary does not independently report gate failure")
    _check(status.get("summary", {}).get("plasticity_drop") == summary.get("plasticity_drop"),
           "source status/summary gate measurements differ")
    _check(summary.get("data_source") == "mnist" and summary.get("synthetic_fallback") is False,
           "source summary data provenance mismatch")
    _validate_source_config(source_config)
    _check(checkpoint.get("method") == "static_sparse", "checkpoint method mismatch")
    _check(checkpoint.get("git_sha") == source_sha, "checkpoint source Git SHA mismatch")
    _check(checkpoint.get("protocol_hash") == source_protocol_hash, "checkpoint source protocol mismatch")
    _check(checkpoint.get("task_sequence_sha256") == task_sequence_sha256,
           "checkpoint task-stream hash mismatch")
    _check(checkpoint.get("completed_task_index") == 149, "checkpoint task index is not 149")
    _check(checkpoint.get("lifetime_examples_seen") == 9000000, "checkpoint lifetime is not 9,000,000")
    _check(checkpoint.get("config") == source_config, "checkpoint source config differs from recorded config")
    _check(all(key in checkpoint for key in ("model", "optimizer", "algorithm", "rng_state")),
           "checkpoint is missing model/optimizer/algorithm/RNG state")
    _check(checkpoint.get("algorithm", {}).get("global_step") == 9000000,
           "checkpoint optimizer step is not 9,000,000")
    parameter_groups = checkpoint.get("optimizer", {}).get("param_groups", [])
    _check(bool(parameter_groups) and all(group.get("lr") == 0.003
           and group.get("momentum") == 0.0 and group.get("weight_decay") == 0.0
           for group in parameter_groups), "checkpoint optimizer state differs from SGD 0.003/0/0")
    _check(len(task_rows) == 150 and [int(row["task_index"]) for row in task_rows] == list(range(150)),
           "source task CSV is incomplete or duplicated")
    _check(all(int(row["examples_seen_in_task"]) == 60000
               and int(row["optimizer_updates"]) == 60000
               and int(row["lifetime_examples_seen"]) == (index + 1) * 60000
               and 0 <= int(row["correct_predictions"]) <= 60000
               and math.isfinite(float(row["online_accuracy"]))
               and 0 <= float(row["online_accuracy"]) <= 1
               and abs(float(row["online_accuracy"]) - int(row["correct_predictions"]) / 60000) < 1e-12
               for index, row in enumerate(task_rows)), "source task CSV exposure/update counts are invalid")
    expected_diag = list(range(0, 150, 5)) + [149]
    _check([int(row["task_index"]) for row in diagnostics] == expected_diag,
           "source diagnostic history is incomplete, duplicated, or unordered")
    expected_diagnostic_fields = {"dead_unit_fraction_overall", "mean_absolute_weight",
                                  "effective_rank_layer_0", "effective_rank_layer_1",
                                  "effective_rank_layer_2", "initial_mask_overlap_fraction_overall",
                                  *(f"active_edges_layer_{i}" for i in range(4))}
    _check(all(expected_diagnostic_fields.issubset(row) for row in diagnostics),
           "source diagnostics are missing required metric columns")
    _check(all((row.get("phase") == "task_start") for row in diagnostics[:-1]),
           "source task-start diagnostic phase markers are invalid")
    _check(all(int(row["lifetime_examples_seen"]) ==
               (int(row["task_index"]) + (1 if row.get("phase") == "final" else 0)) * 60000
               for row in diagnostics), "source diagnostic lifetime metadata is invalid")
    _check(diagnostics[-1].get("phase") == "final", "source final diagnostic row is missing")
    _check([int(float(diagnostics[-1][f"active_edges_layer_{i}"])) for i in range(4)] == EXPECTED_EDGES,
           "source diagnostic edge counts mismatch")
    _check(float(diagnostics[-1]["initial_mask_overlap_fraction_overall"]) == 1.0,
           "Static Sparse initial mask overlap is not exactly one")
    _check(all(math.isfinite(float(row[field])) for row in diagnostics
               for field in expected_diagnostic_fields), "source diagnostic history contains non-finite metrics")
    _check(_finite_tree(checkpoint["model"]) and _finite_tree(checkpoint["optimizer"])
           and _finite_tree(checkpoint["algorithm"]), "source checkpoint contains non-finite state")
    _check(all(float(row["online_accuracy"]) >= 0 for row in task_rows),
           "source task curve contains invalid accuracy")
    return True


def validate_source_run(run_root, stream):
    """Resolve and validate the official completed Cycle 2C source run."""
    run_root = Path(run_root)
    verify_lock_file(cycle2c_file_hashes())
    lock = json.loads((ROOT / "configs/cycle2c/protocol_lock.json").read_text())
    _check(lock.get("protocol_hash") == CYCLE2C_SOURCE_PROTOCOL_HASH,
           "checked-out Cycle 2C protocol lock differs from approved source")
    manifest = json.loads((run_root / "run_manifest.json").read_text())
    status = json.loads((run_root / "status.json").read_text())
    result_dir = run_root / "static_sparse" / "result"
    summary = json.loads((result_dir / "summary.json").read_text())
    source_config = yaml.safe_load((result_dir / "config.yaml").read_text())
    frozen_source_config = yaml.safe_load((ROOT / "configs/cycle2c/static_sparse.yaml").read_text())
    frozen_source_config["runtime"]["status_path"] = source_config.get("runtime", {}).get("status_path")
    frozen_source_config["protocol_hash"] = CYCLE2C_SOURCE_PROTOCOL_HASH
    frozen_source_config["git_sha"] = CYCLE2C_EXECUTION_SHA
    _check(source_config == frozen_source_config,
           "source Static Sparse config differs from the frozen Cycle 2C config")
    checkpoint_path = result_dir / "checkpoint_latest.pt"
    checkpoint_sha = file_sha256(checkpoint_path)
    checkpoint = load_source_checkpoint(checkpoint_path)
    task_rows = csv_rows(result_dir / "task_accuracy.csv")
    diagnostics = csv_rows(result_dir / "diagnostics.csv")
    validate_source_identity(manifest, status, summary, checkpoint, task_rows,
                             diagnostics, source_config, lock["protocol_hash"],
                             stream.sequence_digest(150, 60000))
    _check(manifest.get("run_id") == run_root.name, "source run directory/name identity mismatch")
    unlaunched_methods = {"selective_reset", "rigl_reference", "ccdn_0a"}
    _check(not any((run_root / method).exists() for method in unlaunched_methods),
           "Cycle 2C gate-failure source unexpectedly contains comparison method outputs")
    _check(manifest.get("protocol_source_sha256") == lock.get("sha256"),
           "source run manifest protocol file hashes differ from Cycle 2C lock")
    seed_everything(101, True)
    source_model, source_algorithm, _ = _build_online_learner(source_config, stream, torch.device("cpu"))
    source_masks = [layer.mask.detach().cpu() for layer in source_model.layers]
    checkpoint_masks = [checkpoint["model"][f"layers.{i}.mask"].bool().cpu() for i in range(4)]
    _check(all(torch.equal(a, b) for a, b in zip(source_masks, checkpoint_masks)),
           "Static Sparse topology differs from its seed/config initial topology")
    _check([int(mask.sum()) for mask in checkpoint_masks] == EXPECTED_EDGES,
           "checkpoint active edge counts mismatch")
    _check(source_algorithm.global_step == 0, "invalid fresh static algorithm fixture")
    validate_static_checkpoint_tensors(checkpoint, source_masks)
    return {
        "run_root": str(run_root), "run_id": manifest["run_id"],
        "manifest": manifest, "status": status, "summary": summary,
        "source_config": source_config, "checkpoint": checkpoint,
        "checkpoint_path": str(checkpoint_path), "checkpoint_sha256": checkpoint_sha,
        "task_rows": task_rows, "diagnostic_rows": diagnostics,
        "initial_masks": source_masks,
    }


def build_cycle2d_config(source_config, full_stream_hash, git_sha, protocol_hash):
    config = copy.deepcopy(source_config)
    config["experiment"]["name"] = "cycle2d_static_sparse_lifetime_calibration"
    config["stream"]["tasks"] = 800
    config["runtime"]["expected_task_sequence_sha256"] = full_stream_hash
    config["runtime"].pop("status_path", None)
    config["protocol_hash"] = protocol_hash
    config["git_sha"] = git_sha
    config["output"] = {"root": "results/cycle2d"}
    return config


def bootstrap_inherited_state(source, destination, *, config, stream,
                              stream_hashes, git_sha, protocol_hash):
    """Create an equivalent Cycle 2D checkpoint/history without a training step."""
    destination = Path(destination)
    _check(not destination.exists(), f"Cycle 2D bootstrap destination already exists: {destination}")
    source_checkpoint_path = Path(source["checkpoint_path"])
    before_checkpoint_sha = file_sha256(source_checkpoint_path)
    _check(before_checkpoint_sha == source["checkpoint_sha256"], "source checkpoint changed after validation")
    destination.mkdir(parents=True)
    source_result = Path(source["run_root"]) / "static_sparse" / "result"
    shutil.copyfile(source_result / "task_accuracy.csv", destination / "task_accuracy.csv")
    shutil.copyfile(source_result / "diagnostics.csv", destination / "diagnostics.csv")
    (destination / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))

    seed_everything(101, True)
    model, algorithm, optimizer = _build_online_learner(config, stream, torch.device("cpu"))
    checkpoint = source["checkpoint"]
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    algorithm.load_state_dict(checkpoint["algorithm"])
    restore_rng_state(checkpoint["rng_state"])
    expected_rng = capture_rng_state()
    _check(_nested_equal(model.state_dict(), checkpoint["model"]), "inherited model state mismatch")
    _check(_nested_equal(optimizer.state_dict(), checkpoint["optimizer"]), "inherited optimizer state mismatch")
    _check(_nested_equal(algorithm.state_dict(), checkpoint["algorithm"]), "inherited algorithm state mismatch")
    _check(_nested_equal(expected_rng, checkpoint["rng_state"]), "inherited RNG state mismatch")
    _check(algorithm.global_step == 9000000, "inherited global optimizer step mismatch")
    _check([int(layer.mask.sum()) for layer in model.layers] == EXPECTED_EDGES,
           "inherited active edge budget mismatch")
    _check(all(torch.equal(layer.mask.cpu(), mask.cpu()) for layer, mask in zip(model.layers, source["initial_masks"])),
           "inherited Static Sparse topology changed")
    next_task = stream.task(150)
    _check(next_task.task_index == 150, "next task is not zero-based index 150 / Task 151")
    rng_before = capture_rng_state()
    # Checkpoint serialization and hashing should not consume random values.
    inherited = {
        "model": model.state_dict(), "optimizer": optimizer.state_dict(),
        "algorithm": algorithm.state_dict(), "rng_state": checkpoint["rng_state"],
        "completed_task_index": 149, "lifetime_examples_seen": 9000000,
        "config": copy.deepcopy(config), "runtime_seconds": checkpoint.get("runtime_seconds", 0.0),
        "protocol_hash": protocol_hash, "git_sha": git_sha, "method": "static_sparse",
        "task_sequence_sha256": stream_hashes[800],
        "inherited_from_cycle": "2C",
        "source_execution_sha": CYCLE2C_EXECUTION_SHA,
        "source_protocol_hash": CYCLE2C_SOURCE_PROTOCOL_HASH,
        "source_run_id": source["manifest"]["run_id"],
        "source_checkpoint_sha256": before_checkpoint_sha,
        "source_completed_task_index": 149,
        "source_lifetime_examples_seen": 9000000,
        "inheritance_timestamp": datetime.now(timezone.utc).isoformat(),
    }
    checkpoint_path = destination / "checkpoint_latest.pt"
    temp = destination / "checkpoint_latest.pt.tmp"
    with temp.open("wb") as handle:
        torch.save(inherited, handle)
        handle.flush()
        import os
        os.fsync(handle.fileno())
    temp.replace(checkpoint_path)
    loaded = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    _check(_nested_equal(loaded["model"], checkpoint["model"]), "serialized inherited model differs")
    _check(_nested_equal(loaded["optimizer"], checkpoint["optimizer"]), "serialized optimizer differs")
    _check(_nested_equal(loaded["algorithm"], checkpoint["algorithm"]), "serialized algorithm differs")
    _check(_nested_equal(loaded["rng_state"], checkpoint["rng_state"]), "serialized RNG differs")
    _check(file_sha256(source_checkpoint_path) == before_checkpoint_sha,
           "source Cycle 2C checkpoint bytes were modified")
    _check(_nested_equal(rng_before, capture_rng_state()), "bootstrap unexpectedly changed RNG during serialization")
    _check((destination / "task_accuracy.csv").read_bytes() == (source_result / "task_accuracy.csv").read_bytes(),
           "historical task CSV changed during inheritance")
    _check((destination / "diagnostics.csv").read_bytes() == (source_result / "diagnostics.csv").read_bytes(),
           "historical diagnostic CSV changed during inheritance")
    return {
        "status": "PASS", "training_updates_during_bootstrap": 0,
        "source_checkpoint_sha256_before": before_checkpoint_sha,
        "source_checkpoint_sha256_after": file_sha256(source_checkpoint_path),
        "source_checkpoint_bytes_unchanged": True,
        "model_state_exact": True, "mask_state_exact": True,
        "optimizer_state_exact": True, "algorithm_state_exact": True,
        "rng_state_exact": True, "global_optimizer_step": algorithm.global_step,
        "completed_task_index": 149, "lifetime_examples_seen": 9000000,
        "next_task_index": next_task.task_index, "next_task_human_number": next_task.task_index + 1,
        "active_weights_by_layer": [int(layer.mask.sum()) for layer in model.layers],
        "historical_task_csv_unchanged": True, "historical_diagnostics_unchanged": True,
        "source_protocol_hash": CYCLE2C_SOURCE_PROTOCOL_HASH,
        "new_protocol_hash": protocol_hash,
        "new_task_sequence_sha256": stream_hashes[800],
    }
