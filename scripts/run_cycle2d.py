#!/usr/bin/env python3
"""Self-monitoring, Static Sparse-only Cycle 2D continuation runner."""
from __future__ import annotations

import argparse
import csv
from contextlib import contextmanager
import hashlib
import json
import os
import signal
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
import yaml

from ccdn.experiments.online_permuted_mnist import (
    _build_online_learner, _device, _finite_tree,
    _sparse_diagnostics, _learner_step, finite_check_due,
)
from ccdn.metrics.plasticity import online_plasticity_diagnostics
from ccdn.models.sparse_mlp import SparseMLP
from ccdn.streams.online_permuted_mnist import OnlinePermutedMNIST
from ccdn.utils.reproducibility import capture_rng_state, restore_rng_state, seed_everything
from scripts.bootstrap_cycle2d import (
    CYCLE2C_EXECUTION_SHA, EXPECTED_ACTIVE_PARAMETERS, EXPECTED_EDGES,
    HORIZONS, bootstrap_inherited_state, build_cycle2d_config,
    validate_source_run, file_sha256, task_sequence_digests,
)
from scripts.cycle2d_calibration import calibration_outcome, evaluate_calibration_horizon

ROOT = Path(__file__).resolve().parents[1]


class TaskWatchdogTimeout(TimeoutError):
    pass


@contextmanager
def task_watchdog(seconds=1800):
    """Bound a silent single-task interval while preserving task checkpoints."""
    if not hasattr(signal, "setitimer"):
        yield
        return
    previous = signal.getsignal(signal.SIGALRM)
    def expired(_signum, _frame):
        raise TaskWatchdogTimeout(f"no task-boundary heartbeat for {seconds} seconds")
    signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)
LOCK_FILES = (
    "ccdn/baselines/base.py", "ccdn/baselines/static_sparse.py",
    "ccdn/models/sparse_mlp.py", "ccdn/streams/online_permuted_mnist.py",
    "ccdn/experiments/online_permuted_mnist.py", "ccdn/metrics/plasticity.py",
    "ccdn/metrics/resources.py", "ccdn/utils/reproducibility.py",
    "ccdn/utils/config.py", "scripts/run_cycle2c.py",
    "scripts/bootstrap_cycle2d.py", "scripts/cycle2d_calibration.py",
    "scripts/run_cycle2d.py", "scripts/run_cycle2d.sh",
    "scripts/preflight_cycle2d.py", "configs/cycle2c/protocol_lock.json",
    "configs/cycle2d/static_sparse.yaml", "configs/cycle2d/protocol.yaml",
    "tests/test_cycle2d_bootstrap.py", "tests/test_cycle2d_stream.py",
    "tests/test_cycle2d_gate.py", "tests/test_cycle2d_runner.py",
    "pyproject.toml",
)


def sha_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def cycle2d_file_hashes():
    return {name: sha_file(ROOT / name) for name in LOCK_FILES}


def cycle2d_protocol_digest(hashes=None):
    hashes = hashes or cycle2d_file_hashes()
    raw = json.dumps(hashes, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def verify_cycle2d_lock():
    lock = json.loads((ROOT / "configs/cycle2d/protocol_lock.json").read_text())
    actual = cycle2d_file_hashes()
    try:
        validate_lock_payload(lock, actual)
    except RuntimeError:
        changed = sorted(name for name in set(lock.get("sha256", {})) | set(actual)
                         if lock.get("sha256", {}).get(name) != actual.get(name))
        raise RuntimeError("Cycle 2D protocol lock mismatch: " + ", ".join(changed))
    return lock


def validate_approved_sha(approved_sha, head):
    if not approved_sha:
        raise RuntimeError("ABORTED_PROTOCOL_MISMATCH: reviewer-approved full SHA is required")
    if len(approved_sha) != 40 or any(char not in "0123456789abcdefABCDEF" for char in approved_sha):
        raise RuntimeError("ABORTED_PROTOCOL_MISMATCH: reviewer-approved SHA must be 40 hex characters")
    if approved_sha.lower() != head.lower():
        raise RuntimeError(f"ABORTED_PROTOCOL_MISMATCH: approved SHA {approved_sha} does not match HEAD {head}")
    return True


def validate_branch_clean(branch, clean, status=""):
    if branch != "cycle-2d-sparse-lifetime-calibration":
        raise RuntimeError(f"ABORTED_PROTOCOL_MISMATCH: wrong branch {branch}")
    if not clean:
        raise RuntimeError(f"ABORTED_PROTOCOL_MISMATCH: dirty worktree: {status!r}")
    return True


def validate_lock_payload(lock, hashes):
    digest = cycle2d_protocol_digest(hashes)
    if lock.get("sha256") != hashes or lock.get("protocol_hash") != digest:
        raise RuntimeError("Cycle 2D protocol lock mismatch")
    return digest


def validate_active_counts(counts, expected=EXPECTED_EDGES):
    if list(counts) != list(expected):
        raise RuntimeError(f"active edge budget mismatch: {list(counts)} != {list(expected)}")
    return True


def git_info():
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    branch = subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT, text=True).strip()
    status = subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=normal"], cwd=ROOT, text=True)
    return head, branch, not bool(status.strip()), status


def discover_cycle2c_run(results_root):
    results_root = Path(results_root)
    matches = []
    if results_root.exists():
        for manifest_path in results_root.glob("*/run_manifest.json"):
            try:
                manifest = json.loads(manifest_path.read_text())
            except (OSError, json.JSONDecodeError):
                continue
            if manifest.get("git_commit") == CYCLE2C_EXECUTION_SHA:
                matches.append(manifest_path.parent)
    if len(matches) != 1:
        raise RuntimeError(f"Cycle 2C source discovery expected one approved run, found {len(matches)}")
    return matches[0]


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".{os.getpid()}.{time.time_ns()}.tmp")
    with temp.open("w", encoding="utf8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temp.replace(path)


def acquire_run_lock(path, git_sha, protocol_hash):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    while True:
        record = {"pid": os.getpid(), "git_sha": git_sha,
                  "protocol_hash": protocol_hash,
                  "started_at": datetime.now(timezone.utc).isoformat()}
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        except FileExistsError:
            try:
                old = json.loads(path.read_text())
                os.kill(int(old["pid"]), 0)
            except ProcessLookupError:
                path.unlink(missing_ok=True)
                continue
            except (OSError, ValueError, KeyError, json.JSONDecodeError):
                raise RuntimeError(f"unreadable Cycle 2D lock; refusing stale-lock recovery: {path}")
            raise RuntimeError(f"another Cycle 2D process is active (PID {old['pid']})")
        with os.fdopen(fd, "w", encoding="utf8") as handle:
            json.dump(record, handle, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        return record


def load_valid_checkpoint(path):
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        required = {"model", "optimizer", "algorithm", "rng_state", "completed_task_index",
                    "lifetime_examples_seen", "config", "protocol_hash", "git_sha",
                    "method", "task_sequence_sha256", "source_checkpoint_sha256"}
        if not isinstance(payload, dict) or not required.issubset(payload):
            return None
        return payload
    except Exception:
        return None


def select_resume_checkpoint(result_dir):
    result_dir = Path(result_dir)
    latest = result_dir / "checkpoint_latest.pt"
    previous = result_dir / "checkpoint_previous.pt"
    abandoned_temp = result_dir / "checkpoint_latest.pt.tmp"
    if latest.exists():
        payload = load_valid_checkpoint(latest)
        if payload is not None:
            abandoned_temp.unlink(missing_ok=True)
            return latest, payload, False
    saved_previous = load_valid_checkpoint(previous) if previous.exists() else None
    if saved_previous is not None:
        latest.unlink(missing_ok=True)
        abandoned_temp.unlink(missing_ok=True)
        print(f"[CYCLE2D] RECOVERY: using checkpoint_previous.pt: {previous}", flush=True)
        return previous, saved_previous, True
    abandoned_temp.unlink(missing_ok=True)
    raise RuntimeError("Cycle 2D has no valid latest/previous checkpoint; refusing to restart")


def atomic_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    if not rows:
        return
    temp = path.with_name(path.name + f".{os.getpid()}.{time.time_ns()}.tmp")
    with temp.open("w", newline="", encoding="utf8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    temp.replace(path)


def read_csv(path):
    with Path(path).open(newline="", encoding="utf8") as handle:
        return list(csv.DictReader(handle))


def _save_rolling_checkpoint(path, payload):
    path = Path(path)
    temp = path.with_name(path.name + ".tmp")
    with temp.open("wb") as handle:
        torch.save(payload, handle)
        handle.flush()
        os.fsync(handle.fileno())
    previous = path.with_name("checkpoint_previous.pt")
    if path.exists():
        path.replace(previous)
    temp.replace(path)
    verified = load_valid_checkpoint(path)
    if verified is None:
        raise RuntimeError("Cycle 2D checkpoint failed read-back validation")
    return path


def _expected_config(source_config, stream_hash, git_sha, protocol_hash):
    return build_cycle2d_config(source_config, stream_hash, git_sha, protocol_hash)


def _assert_static_edges(model, initial_masks):
    if not isinstance(model, SparseMLP):
        raise RuntimeError("Cycle 2D model must be SparseMLP")
    counts = [int(layer.mask.sum()) for layer in model.layers]
    validate_active_counts(counts)
    if any(not torch.equal(layer.mask, start) for layer, start in zip(model.layers, initial_masks)):
        raise RuntimeError("Static Sparse mask changed during Cycle 2D")
    if any(bool((layer.weight[~layer.mask] != 0).any()) for layer in model.layers):
        raise RuntimeError("inactive sparse weights are nonzero")
    return counts


def _diagnostic_row(model, algorithm, stream, task_index, lifetime, phase, initial_masks, device):
    task = stream.task(task_index)
    diag_x, _ = stream.diagnostic_batch(task, 2000, device)
    result = {"task_index": task_index, "phase": phase,
              "lifetime_examples_seen": lifetime,
              **online_plasticity_diagnostics(model, diag_x),
              **_sparse_diagnostics(model, algorithm, initial_masks)}
    if not _finite_tree(result):
        raise FloatingPointError("non-finite Cycle 2D diagnostic")
    return result


def _finite(model, algorithm, optimizer=None):
    state = {"model": model.state_dict(), "algorithm": algorithm.state_dict()}
    if optimizer is not None:
        state["optimizer"] = optimizer.state_dict()
    return (_finite_tree(state)
            and all(bool(torch.isfinite(parameter).all()) for parameter in model.parameters()))


def _heartbeat(method, task_index, row, updates, counts, expected, next_horizon,
               latest_eval, checkpoint_path, finite_ok):
    edge_ok = counts == expected
    saved = load_valid_checkpoint(checkpoint_path)
    checkpoint_ok = bool(saved is not None
                         and saved.get("completed_task_index") == task_index
                         and saved.get("lifetime_examples_seen") == updates
                         and saved.get("method") == method)
    report = {
        "cycle": "2D", "method": method,
        "completed_task_index": task_index,
        "task_number": task_index + 1, "tasks_requested": 800,
        "online_accuracy": float(row["online_accuracy"]),
        "lifetime_optimizer_updates": int(updates),
        "expected_active_edges_by_layer": expected,
        "current_active_edges_by_layer": counts,
        "edge_budget_pass": edge_ok,
        "finite_state_pass": bool(finite_ok),
        "checkpoint_pass": checkpoint_ok,
        "next_candidate_horizon": next_horizon,
        "latest_candidate_gate_status": latest_eval.get("gate_passed") if latest_eval else None,
        "latest_candidate_horizon": latest_eval.get("horizon") if latest_eval else None,
        "checkpoint": str(checkpoint_path),
    }
    return report


def _task_start_diagnostics(index):
    return index == 0 or index % 5 == 0


def _prepare_resume(result_dir, config, protocol_hash, git_sha, full_stream_hash):
    checkpoint_path, checkpoint, recovered = select_resume_checkpoint(result_dir)
    expected = {"config": config, "protocol_hash": protocol_hash,
                "git_sha": git_sha, "method": "static_sparse",
                "task_sequence_sha256": full_stream_hash}
    mismatch = [key for key, value in expected.items() if checkpoint.get(key) != value]
    if mismatch:
        raise RuntimeError("Cycle 2D checkpoint identity/config mismatch: " + ", ".join(mismatch))
    if checkpoint.get("completed_task_index", -1) < 149:
        raise RuntimeError("Cycle 2D checkpoint regressed before inherited task-150 state")
    return checkpoint_path, checkpoint, recovered


def execute_calibration(result_dir, *, config, stream, stream_hashes,
                        protocol_hash, git_sha, source, status_path, log_path):
    result_dir = Path(result_dir)
    seed_everything(101, True)
    device = _device(config["experiment"].get("device", "auto"))
    if device.type == "cpu":
        torch.set_num_threads(int(config["experiment"].get("cpu_threads", 1)))
    model, algorithm, optimizer = _build_online_learner(config, stream, device)
    initial_masks = [layer.mask.detach().clone() for layer in model.layers]
    checkpoint_path, checkpoint, recovered = _prepare_resume(
        result_dir, config, protocol_hash, git_sha, stream_hashes[800])
    if checkpoint.get("source_checkpoint_sha256") != source["checkpoint_sha256"]:
        raise RuntimeError("Cycle 2D rolling checkpoint source checkpoint provenance mismatch")
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    algorithm.load_state_dict(checkpoint["algorithm"])
    restore_rng_state(checkpoint["rng_state"])
    _assert_static_edges(model, initial_masks)

    task_csv = result_dir / "task_accuracy.csv"
    diag_csv = result_dir / "diagnostics.csv"
    task_rows = read_csv(task_csv)
    diagnostic_rows = read_csv(diag_csv)
    first_task = int(checkpoint["completed_task_index"]) + 1
    lifetime = int(checkpoint["lifetime_examples_seen"])
    task_rows = [row for row in task_rows if int(row["task_index"]) < first_task]
    diagnostic_rows = [row for row in diagnostic_rows if int(row["lifetime_examples_seen"]) <= lifetime]
    if len(task_rows) != first_task or lifetime != first_task * 60000 or algorithm.global_step != lifetime:
        raise RuntimeError("Cycle 2D task CSV/checkpoint/optimizer progress is inconsistent")
    atomic_csv(task_csv, task_rows)
    atomic_csv(diag_csv, diagnostic_rows)
    evaluation_path = result_dir / "calibration_evaluations.json"
    evaluations = json.loads(evaluation_path.read_text()) if evaluation_path.exists() else []
    evaluations = [evaluation for evaluation in evaluations
                   if int(evaluation["horizon"]) <= first_task]
    evaluation_by_horizon = {int(row["horizon"]): row for row in evaluations}

    # Reconcile a crash after task checkpoint but before evaluation-file update.
    for horizon in HORIZONS:
        if horizon <= first_task and horizon not in evaluation_by_horizon:
            if not any(int(row["task_index"]) == horizon - 1 and row.get("phase") == "horizon_snapshot"
                       for row in diagnostic_rows):
                if horizon != first_task:
                    raise RuntimeError(f"missing persisted diagnostic snapshot at completed horizon {horizon}")
                diag = _diagnostic_row(model, algorithm, stream, horizon - 1, lifetime,
                                       "horizon_snapshot", initial_masks, device)
                diagnostic_rows.append(diag)
                atomic_csv(diag_csv, diagnostic_rows)
            evaluation = evaluate_calibration_horizon(task_rows, horizon)
            evaluation_by_horizon[horizon] = evaluation
            evaluations.append(evaluation)
    evaluations.sort(key=lambda item: int(item["horizon"]))
    atomic_json(evaluation_path, evaluations)
    outcome = calibration_outcome(evaluations)

    if outcome["state"] in {"CONFIRMED_FAILURE_HORIZON", "NO_CONFIRMED_FAILURE_THROUGH_800",
                             "UNCONFIRMED_PASS_AT_800"}:
        atomic_json(result_dir / "calibration_summary.json", {
            "status": outcome["state"], "outcome": outcome,
            "evaluations": evaluations, "completed_tasks": first_task,
            "optimizer_updates": lifetime,
            "runtime_seconds": checkpoint.get("runtime_seconds", 0.0),
        })
        atomic_json(status_path, {"state": outcome["state"], **outcome,
                                  "completed_tasks": first_task, "optimizer_updates": lifetime})
        return outcome

    started = time.perf_counter()
    last_eval = evaluations[-1] if evaluations else None
    for task_index in range(first_task, 800):
        if _task_start_diagnostics(task_index) and not any(
                int(row["task_index"]) == task_index and row.get("phase") == "task_start"
                for row in diagnostic_rows):
            diagnostic_rows.append(_diagnostic_row(model, algorithm, stream, task_index,
                                                   lifetime, "task_start", initial_masks, device))
            atomic_csv(diag_csv, diagnostic_rows)
        task = stream.task(task_index)
        correct = 0
        loss_sum = 0.0
        with task_watchdog():
            for position in range(60000):
                x, y = stream.sample(task, position, device)
                # Learner inputs remain exactly (x,y); evaluator-only indices stay here.
                prediction, loss = _learner_step(model, algorithm, optimizer, x, y)
                lifetime += 1
                if finite_check_due(algorithm.global_step) and not _finite(model, algorithm, optimizer):
                    raise FloatingPointError(f"non-finite state at update {algorithm.global_step}")
                correct += int(prediction == int(y.item()))
                loss_sum += loss
        finite_ok = _finite(model, algorithm, optimizer)
        if not finite_ok:
            raise FloatingPointError(f"non-finite state after task {task_index}")
        if algorithm.global_step != lifetime:
            raise RuntimeError("Cycle 2D optimizer update/lifetime counter mismatch")
        counts = _assert_static_edges(model, initial_masks)
        accuracy = correct / 60000
        task_row = {
            "task_index": task_index, "online_accuracy": accuracy,
            "correct_predictions": correct, "examples_seen_in_task": 60000,
            "lifetime_examples_seen": lifetime, "optimizer_updates": 60000,
            "mean_preupdate_loss": loss_sum / 60000,
            "active_edges_by_layer": json.dumps(counts),
            "algorithm_metrics": json.dumps(algorithm.metrics(), sort_keys=True),
        }
        task_rows.append(task_row)
        atomic_csv(task_csv, task_rows)

        completed_tasks = task_index + 1
        if completed_tasks in HORIZONS:
            diagnostic_rows.append(_diagnostic_row(model, algorithm, stream, task_index,
                                                   lifetime, "horizon_snapshot", initial_masks, device))
            atomic_csv(diag_csv, diagnostic_rows)
        payload = {
            "model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "algorithm": algorithm.state_dict(), "completed_task_index": task_index,
            "lifetime_examples_seen": lifetime, "rng_state": capture_rng_state(),
            "config": config, "protocol_hash": protocol_hash, "git_sha": git_sha,
            "method": "static_sparse", "task_sequence_sha256": stream_hashes[800],
            "source_execution_sha": CYCLE2C_EXECUTION_SHA,
            "source_protocol_hash": source["manifest"]["protocol_hash"],
            "source_run_id": source["manifest"]["run_id"],
            "source_checkpoint_sha256": source["checkpoint_sha256"],
            "runtime_seconds": float(checkpoint.get("runtime_seconds", 0.0)) + time.perf_counter() - started,
        }
        checkpoint_path = _save_rolling_checkpoint(result_dir / "checkpoint_latest.pt", payload)
        if completed_tasks in HORIZONS:
            last_eval = evaluate_calibration_horizon(task_rows, completed_tasks)
            evaluation_by_horizon[completed_tasks] = last_eval
            evaluations = [evaluation_by_horizon[h] for h in sorted(evaluation_by_horizon)]
            atomic_json(evaluation_path, evaluations)
            outcome = calibration_outcome(evaluations)
        next_horizon = next((h for h in HORIZONS if h > completed_tasks), None)
        heartbeat = _heartbeat("static_sparse", task_index, task_row, lifetime,
                               counts, EXPECTED_EDGES, next_horizon, last_eval,
                               checkpoint_path, finite_ok)
        if not (heartbeat["edge_budget_pass"] and heartbeat["finite_state_pass"]
                and heartbeat["checkpoint_pass"]):
            raise RuntimeError("Cycle 2D task-boundary health check failed")
        atomic_json(result_dir / "heartbeat.json", heartbeat)
        atomic_json(status_path, {"state": "RUNNING", **heartbeat,
                                  "calibration_state": outcome["state"]})
        line = (f"[CYCLE2D] static_sparse | Task {completed_tasks}/800 | "
                f"accuracy={accuracy:.4f} | updates={lifetime:,} | "
                f"expected_edges={EXPECTED_EDGES} current_edges={counts} edge_budget=PASS | "
                f"finite=PASS | checkpoint=PASS | "
                f"next_horizon={next_horizon} | "
                f"latest_gate={last_eval.get('horizon')}:{last_eval.get('gate_passed')}"
                if last_eval else
                f"[CYCLE2D] static_sparse | Task {completed_tasks}/800 | "
                f"accuracy={accuracy:.4f} | updates={lifetime:,} | "
                f"expected_edges={EXPECTED_EDGES} current_edges={counts} edge_budget=PASS | "
                f"finite=PASS | checkpoint=PASS | "
                f"next_horizon={next_horizon} | latest_gate=not_evaluated")
        print(line, flush=True)
        with (Path(log_path)).open("a", encoding="utf8") as log:
            log.write(line + "\n")
            log.flush()
            os.fsync(log.fileno())
        if outcome["state"] == "CONFIRMED_FAILURE_HORIZON":
            break
        if completed_tasks == 800:
            break

    outcome = calibration_outcome(evaluations)
    atomic_json(result_dir / "calibration_summary.json", {
        "status": outcome["state"], "outcome": outcome,
        "evaluations": evaluations, "completed_tasks": len(task_rows),
        "optimizer_updates": lifetime, "runtime_seconds": payload["runtime_seconds"],
    })
    atomic_json(status_path, {"state": outcome["state"], **outcome,
                              "completed_tasks": len(task_rows), "optimizer_updates": lifetime})
    return outcome


def _validate_environment(approved_sha):
    head, branch, clean, status = git_info()
    validate_approved_sha(approved_sha, head)
    validate_branch_clean(branch, clean, status)
    lock = verify_cycle2d_lock()
    return head, branch, lock


def _validate_real_stream(lock):
    protocol = yaml.safe_load((ROOT / "configs/cycle2d/protocol.yaml").read_text())
    config = yaml.safe_load((ROOT / "configs/cycle2d/static_sparse.yaml").read_text())
    stream = OnlinePermutedMNIST(seed=101, root=config["stream"]["dataset_root"], download=False)
    if stream.data_source != "mnist" or len(stream.labels) != 60000 or stream.input_size != 784:
        raise RuntimeError("Cycle 2D requires real 60,000-example MNIST")
    if not bool(torch.isfinite(stream.images).all()) or int(stream.labels.min()) != 0 or int(stream.labels.max()) != 9:
        raise RuntimeError("Cycle 2D MNIST failed finite/label checks")
    digests = task_sequence_digests(stream)
    digests[150] = stream.sequence_digest(150, 60000)
    expected = {int(key): value for key, value in protocol["stream"]["prefix_sha256"].items()}
    if digests != expected:
        raise RuntimeError(f"Cycle 2D frozen task stream mismatch: {digests}")
    if digests[150] != protocol["stream"]["prefix_150_sha256"]:
        raise RuntimeError("Cycle 2D first-150 prefix differs from Cycle 2C")
    if config["runtime"]["expected_task_sequence_sha256"] != digests[800]:
        raise RuntimeError("Cycle 2D config does not freeze the 800-task stream")
    return stream, config, protocol, digests


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("approved_sha", nargs="?", help="reviewer-approved full Git SHA")
    parser.add_argument("--source-run-root")
    args = parser.parse_args(argv)
    head = None
    lock = None
    run_lock_path = ROOT / "results/cycle2d/.run_lock"
    acquired = False
    try:
        head, branch, lock = _validate_environment(args.approved_sha)
        args.approved_sha = head
        acquire_run_lock(run_lock_path, head, lock["protocol_hash"])
        acquired = True
        stream, base_config, protocol, digests = _validate_real_stream(lock)
        source_root = Path(args.source_run_root) if args.source_run_root else discover_cycle2c_run(ROOT / "results/cycle2c")
        source = validate_source_run(source_root, stream)
        config = _expected_config(source["source_config"], digests[800], head, lock["protocol_hash"])
        next_task = stream.task(150)
        if next_task.task_index != 150:
            raise RuntimeError("Cycle 2D bootstrap does not continue with Task 151")

        run_id = f"seed101_{head[:8]}_{lock['protocol_hash'][:8]}"
        run_root = ROOT / "results/cycle2d" / run_id
        result_dir = run_root / "static_sparse" / "result"
        status_path = run_root / "status.json"
        log_path = run_root / "terminal.log"
        provenance = {
            "run_id": run_id, "timestamp": datetime.now(timezone.utc).isoformat(),
            "cycle": "2D", "git_commit": head, "git_branch": branch,
            "reviewer_approved_git_sha": args.approved_sha,
            "protocol_hash": lock["protocol_hash"], "protocol_source_sha256": lock["sha256"],
            "source_run_id": source["manifest"]["run_id"],
            "source_execution_sha": CYCLE2C_EXECUTION_SHA,
            "source_protocol_hash": source["manifest"]["protocol_hash"],
            "source_checkpoint_sha256": source["checkpoint_sha256"],
            "source_task_sequence_sha256": source["manifest"]["task_sequence_sha256"],
            "task_sequence_sha256_by_horizon": {str(k): v for k, v in sorted(digests.items())},
            "data_source": stream.data_source, "synthetic_fallback": False,
            "seed": 101, "architecture": protocol["learner"]["architecture"],
            "active_edges_by_layer": EXPECTED_EDGES,
            "logical_active_parameters": EXPECTED_ACTIVE_PARAMETERS,
            "candidate_horizons": HORIZONS,
            "updates_before_continuation": 9000000,
        }
        if run_root.exists():
            manifest_path = run_root / "run_manifest.json"
            if not manifest_path.exists():
                raise RuntimeError("existing Cycle 2D run directory has no identity manifest; refusing fresh run")
            existing = json.loads(manifest_path.read_text())
            identity_keys = ("git_commit", "protocol_hash", "source_checkpoint_sha256", "source_run_id")
            if any(existing.get(key) != provenance.get(key) for key in identity_keys):
                raise RuntimeError("existing Cycle 2D run identity differs from approved protocol/source")
            if not result_dir.exists():
                raise RuntimeError("existing Cycle 2D run has no result state; refusing fresh run")
            # Validate rolling state before allowing continuation.
            _prepare_resume(result_dir, config, lock["protocol_hash"], head, digests[800])
        else:
            run_root.mkdir(parents=True)
            atomic_json(status_path, {"state": "PREFLIGHT"})
            parity = bootstrap_inherited_state(
                source, result_dir, config=config, stream=stream,
                stream_hashes=digests, git_sha=head,
                protocol_hash=lock["protocol_hash"])
            atomic_json(run_root / "run_manifest.json", provenance)
            atomic_json(run_root / "inheritance_parity.json", parity)
        outcome = execute_calibration(
            result_dir, config=config, stream=stream, stream_hashes=digests,
            protocol_hash=lock["protocol_hash"], git_sha=head, source=source,
            status_path=status_path, log_path=log_path)
        print(json.dumps(outcome, indent=2), flush=True)
        return 0
    except KeyboardInterrupt:
        if 'status_path' in locals():
            atomic_json(status_path, {"state": "INTERRUPTED", "reason": "keyboard interrupt"})
        raise
    except TaskWatchdogTimeout as exc:
        if 'status_path' in locals():
            atomic_json(status_path, {"state": "INTERRUPTED", "reason": str(exc)})
        raise
    except FloatingPointError as exc:
        if 'status_path' in locals():
            atomic_json(status_path, {"state": "ABORTED_NONFINITE", "reason": str(exc)})
        raise
    except Exception as exc:
        status = {"state": "ABORTED_PROTOCOL_MISMATCH", "reason": str(exc),
                  "timestamp": datetime.now(timezone.utc).isoformat()}
        if 'run_root' in locals() and run_root.exists():
            atomic_json(run_root / "status.json", status)
        else:
            atomic_json(ROOT / "results/cycle2d/precheck_status.json", status)
        raise
    finally:
        if acquired:
            run_lock_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
