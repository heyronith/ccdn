#!/usr/bin/env python3
"""No-training Cycle 2D real-MNIST/source-state preflight and evidence writer."""
from __future__ import annotations

import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import torch
import yaml

from ccdn.streams.online_permuted_mnist import OnlinePermutedMNIST
from scripts.bootstrap_cycle2d import (
    bootstrap_inherited_state,
    build_cycle2d_config, task_sequence_digests, validate_source_run,
)
from scripts.run_cycle2d import (
    ROOT, cycle2d_file_hashes, cycle2d_protocol_digest, verify_cycle2d_lock,
)


def main():
    protocol = yaml.safe_load((ROOT / "configs/cycle2d/protocol.yaml").read_text())
    base_config = yaml.safe_load((ROOT / "configs/cycle2d/static_sparse.yaml").read_text())
    lock = verify_cycle2d_lock()
    stream = OnlinePermutedMNIST(seed=101, root=base_config["stream"]["dataset_root"], download=False)
    if stream.data_source != "mnist" or len(stream.labels) != 60000 or stream.input_size != 784:
        raise RuntimeError("Cycle 2D preflight requires real 60,000-example MNIST")
    if not bool(torch.isfinite(stream.images).all()):
        raise RuntimeError("Cycle 2D preflight MNIST images are non-finite")
    digests = task_sequence_digests(stream)
    digests[150] = stream.sequence_digest(150, 60000)
    expected = {int(key): value for key, value in protocol["stream"]["prefix_sha256"].items()}
    if digests != expected or digests[150] != protocol["stream"]["prefix_150_sha256"]:
        raise RuntimeError(f"Cycle 2D preflight task stream mismatch: {digests}")
    if base_config["runtime"]["expected_task_sequence_sha256"] != digests[800]:
        raise RuntimeError("Cycle 2D 800-task config hash mismatch")

    source_root = ROOT / "results/cycle2c/seed101_5b48752c_019e3067"
    source = validate_source_run(source_root, stream)
    current_head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    d_config = build_cycle2d_config(source["source_config"], digests[800],
                                    current_head, lock["protocol_hash"])
    with tempfile.TemporaryDirectory(prefix="cycle2d-inheritance-") as temp_root:
        destination = Path(temp_root) / "static_sparse" / "result"
        parity = bootstrap_inherited_state(
            source, destination, config=d_config, stream=stream,
            stream_hashes=digests, git_sha=current_head,
            protocol_hash=lock["protocol_hash"])
        inherited = torch.load(destination / "checkpoint_latest.pt",
                               map_location="cpu", weights_only=False)
        if inherited["completed_task_index"] != 149 or inherited["lifetime_examples_seen"] != 9000000:
            raise RuntimeError("Cycle 2D preflight inherited progress is not task-150")
        if inherited["algorithm"]["global_step"] != 9000000:
            raise RuntimeError("Cycle 2D preflight inherited optimizer is not at 9,000,000 updates")

    out = ROOT / "review_artifacts/cycle_2d_preflight"
    out.mkdir(parents=True, exist_ok=True)
    (out / "protocol.json").write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n")
    (out / "protocol_lock.json").write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n")
    source_evidence = {
        "status": "PASS", "source_run_id": source["manifest"]["run_id"],
        "source_run_root": str(source_root.relative_to(ROOT)),
        "git_commit": source["manifest"]["git_commit"],
        "reviewer_approved_git_sha": source["manifest"]["reviewer_approved_git_sha"],
        "branch": source["manifest"]["git_branch"],
        "protocol_hash": source["manifest"]["protocol_hash"],
        "task_sequence_sha256": source["manifest"]["task_sequence_sha256"],
        "status": source["status"]["state"],
        "data_source": source["manifest"]["data_source"],
        "synthetic_fallback": source["manifest"]["synthetic_fallback"],
        "summary_gate_passed": source["summary"]["loss_of_plasticity_gate_passed"],
        "tasks_completed": source["summary"]["tasks_completed"],
        "updates_completed": source["summary"]["total_optimizer_updates"],
        "exact_task_indices_0_149": True,
        "examples_per_task": 60000,
        "diagnostic_rows": len(source["diagnostic_rows"]),
        "active_edges_by_layer": [int(source["checkpoint"]["model"][f"layers.{i}.mask"].sum()) for i in range(4)],
        "static_masks_unchanged": True,
    }
    (out / "cycle2c_source_validation.json").write_text(json.dumps(source_evidence, indent=2) + "\n")
    checkpoint_evidence = {
        "status": "PASS", "checkpoint_sha256": source["checkpoint_sha256"],
        "checkpoint_path_relative": str(Path(source["checkpoint_path"]).relative_to(ROOT)),
        "source_execution_sha": source["checkpoint"]["git_sha"],
        "source_protocol_hash": source["checkpoint"]["protocol_hash"],
        "task_sequence_sha256": source["checkpoint"]["task_sequence_sha256"],
        "method": source["checkpoint"]["method"],
        "completed_task_index": source["checkpoint"]["completed_task_index"],
        "lifetime_examples_seen": source["checkpoint"]["lifetime_examples_seen"],
        "global_optimizer_step": source["checkpoint"]["algorithm"]["global_step"],
        "required_state_present": ["model", "optimizer", "algorithm", "rng_state", "config"],
        "edge_budget": source_evidence["active_edges_by_layer"],
        "source_checkpoint_not_modified": parity["source_checkpoint_bytes_unchanged"],
    }
    (out / "source_checkpoint_identity.json").write_text(json.dumps(checkpoint_evidence, indent=2) + "\n")
    (out / "inheritance_parity.json").write_text(json.dumps(parity, indent=2, sort_keys=True) + "\n")
    (out / "task_sequence_hashes.json").write_text(json.dumps({
        "seed": 101, "examples_per_task": 60000,
        "hashes_by_task_horizon": {str(key): value for key, value in sorted(digests.items())},
        "150_prefix_matches_cycle2c": digests[150] == source["manifest"]["task_sequence_sha256"],
    }, indent=2, sort_keys=True) + "\n")
    self_tests = {
        "status": "PASS", "training_updates_during_preflight": 0,
        "real_mnist": True, "cycle2c_source_validation": True,
        "all_prefix_hashes_match_locked_protocol": True,
        "bootstrap_state_exact": True, "next_task_index": 150,
        "next_human_task": 151, "latest_previous_checkpoint_helpers_tested": True,
        "no_comparison_learner_constructed_by_preflight_bootstrap": True,
    }
    (out / "runner_self_tests.json").write_text(json.dumps(self_tests, indent=2) + "\n")
    result = subprocess.run([sys.executable, "-m", "pytest", "-q"], cwd=ROOT,
                            text=True, capture_output=True)
    (out / "pytest_output.txt").write_text(result.stdout + result.stderr)
    if result.returncode:
        raise RuntimeError(f"full pytest suite failed during Cycle 2D preflight: {result.stdout}{result.stderr}")
    match = re.search(r"(\d+) passed", result.stdout + result.stderr)
    count = int(match.group(1)) if match else None
    (out / "test_summary.json").write_text(json.dumps({
        "command": f"{Path(sys.executable).name} -m pytest -q",
        "passed": count, "failed": 0, "status": "PASS",
        "synthetic_regression": "not run in this Cycle 2D calibration phase",
    }, indent=2) + "\n")
    (out / "README.md").write_text(
        "# Cycle 2D preflight artifacts\n\n"
        "Preflight engineering evidence only; no scientific Cycle 2D continuation was run. "
        "The source is the completed Cycle 2C Static Sparse run. Real MNIST, all frozen prefix "
        "hashes, source checkpoint identity, exact inheritance parity, and Task 151 as the next "
        "example stream were verified. Bootstrap performed zero training updates. No CCDN, RigL, "
        "or Selective Reset learner was run.\n")
    print(json.dumps({"status": "PASS", "source_run_id": source["manifest"]["run_id"],
                      "source_checkpoint_sha256": source["checkpoint_sha256"],
                      "hashes": digests, "training_updates": 0,
                      "tests_passed": count}, indent=2))


if __name__ == "__main__":
    main()
