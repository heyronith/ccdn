#!/usr/bin/env python3
"""Real-MNIST paired integration preflight; never executes scientific tasks."""
from __future__ import annotations

import csv
import hashlib
import json
import time
import tempfile
from pathlib import Path

import torch
import yaml

from ccdn.experiments.online_permuted_mnist import _build_online_learner, _device, _learner_step
from ccdn.streams.online_permuted_mnist import OnlinePermutedMNIST
from ccdn.utils.reproducibility import seed_everything

ROOT = Path(__file__).resolve().parents[1]
METHODS = ("static_sparse", "selective_reset", "rigl_reference", "ccdn_0a")
EXPECTED_EDGES = [52685, 22579, 22579, 672]


def state_hash(state, masks_only=False):
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        if masks_only and not name.endswith(".mask"):
            continue
        if torch.is_tensor(value):
            digest.update(name.encode())
            digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def active_counts(model):
    return [int(layer.mask.sum()) for layer in model.layers]


def main():
    seed = 101
    stream = OnlinePermutedMNIST(seed=seed, root="./data", download=False)
    if stream.data_source != "mnist" or len(stream.labels) != 60000 or stream.input_size != 784:
        raise RuntimeError("preflight requires real 60,000-example MNIST")
    if not bool(torch.isfinite(stream.images).all()) or int(stream.labels.min()) != 0 or int(stream.labels.max()) != 9:
        raise RuntimeError("MNIST data failed finite/range validation")
    torch.set_num_threads(1)
    task = stream.task(0)
    init_hashes, init_mask_hashes, pre_event_hashes, rows = {}, {}, {}, []
    expected_pre_event = None
    for method in METHODS:
        cfg = yaml.safe_load((ROOT / f"configs/cycle2c/{method}.yaml").read_text())
        seed_everything(seed, True)
        model, algorithm, optimizer = _build_online_learner(cfg, stream, _device("cpu"))
        start_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        init_hashes[method] = state_hash(start_state)
        init_mask_hashes[method] = state_hash(start_state, masks_only=True)
        if active_counts(model) != EXPECTED_EDGES:
            raise AssertionError(f"unexpected initial edge counts for {method}: {active_counts(model)}")
        started = time.perf_counter()
        for position in range(8192):
            x, y = stream.sample(task, position, "cpu")
            _learner_step(model, algorithm, optimizer, x, y)
            if not all(bool(torch.isfinite(p).all()) for p in model.parameters()):
                raise FloatingPointError(f"nonfinite model state: {method} update {position+1}")
            if position == 8190:
                pre_event_hashes[method] = state_hash(model.state_dict())
                if method == METHODS[0]:
                    expected_pre_event = pre_event_hashes[method]
                elif pre_event_hashes[method] != expected_pre_event:
                    raise AssertionError(f"paired methods diverged before update 8192: {method}")
        elapsed = time.perf_counter() - started
        counts = active_counts(model)
        if counts != EXPECTED_EDGES:
            raise AssertionError(f"active budget changed for {method}: {counts}")
        metrics = algorithm.metrics()
        topology_changed = any(not torch.equal(layer.mask, start_state[f"layers.{i}.mask"])
                               for i, layer in enumerate(model.layers))
        event_steps = getattr(algorithm, "structural_event_steps", getattr(algorithm, "event_steps", []))
        if method != "static_sparse" and list(event_steps) != [8192]:
            raise AssertionError(f"{method} did not intervene exactly at update 8192: {event_steps}")
        if method in {"rigl_reference", "ccdn_0a"} and counts != EXPECTED_EDGES:
            raise AssertionError(f"edge conservation failed for {method}")
        if method == "static_sparse" and any(not torch.equal(layer.mask, start_state[f"layers.{i}.mask"]) for i, layer in enumerate(model.layers)):
            raise AssertionError("static sparse topology changed")
        if method == "selective_reset":
            if any(not torch.equal(layer.mask, start_state[f"layers.{i}.mask"]) for i, layer in enumerate(model.layers)):
                raise AssertionError("selective reset changed topology")
            if state_hash(model.state_dict()) == state_hash(start_state):
                raise AssertionError("selective reset did not change any model values")
        if method in {"rigl_reference", "ccdn_0a"}:
            if all(torch.equal(layer.mask, start_state[f"layers.{i}.mask"]) for i, layer in enumerate(model.layers)):
                raise AssertionError(f"{method} event did not change topology")
        if method == "ccdn_0a":
            for util, layer in zip(algorithm.utility, model.layers):
                if not bool(torch.isfinite(util).all()) or bool((util[~layer.mask] != 0).any()):
                    raise AssertionError("CCDN utility is nonfinite or nonzero on inactive connections")
        checkpoint_payload = {"model": model.state_dict(), "algorithm": algorithm.state_dict(),
                              "optimizer": optimizer.state_dict(), "global_step": algorithm.global_step}
        with tempfile.NamedTemporaryFile(suffix=".pt") as checkpoint_file:
            torch.save(checkpoint_payload, checkpoint_file.name)
            restored = torch.load(checkpoint_file.name, map_location="cpu", weights_only=False)
        model.load_state_dict(restored["model"])
        algorithm.load_state_dict(restored["algorithm"])
        optimizer.load_state_dict(restored["optimizer"])
        if state_hash(model.state_dict()) != state_hash(restored["model"]):
            raise AssertionError(f"checkpoint model roundtrip failed: {method}")
        if algorithm.global_step != 8192:
            raise AssertionError(f"checkpoint algorithm step roundtrip failed: {method}")
        rows.append({"method": method, "data_source": stream.data_source,
                     "synthetic_fallback": False, "seed": seed,
                     "updates": 8192, "initial_state_sha256": init_hashes[method],
                     "initial_mask_sha256": init_mask_hashes[method],
                     "pre_event_state_sha256": pre_event_hashes[method],
                     "topology_changed": topology_changed,
                     "reset_count": int(metrics.get("reset_count", 0)),
                     "active_edges_by_layer": json.dumps(counts),
                     "structural_event_steps": json.dumps(list(event_steps)),
                     "structural_metrics": json.dumps(metrics, sort_keys=True),
                     "checkpoint_roundtrip": True,
                     "runtime_seconds": elapsed, "finite_state": True,
                     "active_budget_conserved": counts == EXPECTED_EDGES})
    if len(set(init_hashes.values())) != 1:
        raise AssertionError("starting model weights and masks are not paired")
    out = ROOT / "review_artifacts/cycle_2c_preflight"
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "real_mnist_preflight.csv", "w", newline="", encoding="utf8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)
    report = {"status": "PASSED", "preflight_only": True, "data_source": stream.data_source,
              "synthetic_fallback": False, "task_sequence_sha256": stream.sequence_digest(150,60000),
              "examples_per_method": 8192, "initial_state_hashes": init_hashes,
              "initial_mask_hashes": init_mask_hashes,
              "pre_event_state_hashes": pre_event_hashes,
              "paired_initialization_passed": True, "pre_event_pairing_passed": True,
              "intervention_update": 8192, "methods": list(METHODS), "rows": rows}
    (out / "real_mnist_preflight.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print("Cycle 2C real-MNIST preflight PASSED (4 methods × 8,192 examples); no long scientific run executed.")


if __name__ == "__main__":
    main()
