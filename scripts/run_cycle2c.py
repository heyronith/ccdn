#!/usr/bin/env python3
"""Self-monitoring launcher for the frozen Cycle 2C protocol.

This program is the Phase B terminal entry point. It refuses dirty or mismatched
protocols, verifies real MNIST/task ordering/paired initialization, then runs the
static-sparse gate before allowing any comparison method to launch.
"""
from __future__ import annotations

import hashlib
import argparse
import json
import os
import platform
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_STREAM = "1f2ddb0daa90715192118466b7088fd8d3ea78f44e8647af6bca0a686393f844"
METHODS = ("static_sparse", "selective_reset", "rigl_reference", "ccdn_0a")
LOCK_FILES = (
    "ccdn/algorithms/ccdn_0a.py", "ccdn/baselines/selective_reset.py",
    "ccdn/baselines/base.py", "ccdn/baselines/static_sparse.py",
    "ccdn/official_reference/rigl.py", "ccdn/models/sparse_mlp.py",
    "ccdn/streams/online_permuted_mnist.py", "ccdn/experiments/online_permuted_mnist.py",
    "scripts/run_cycle2c.py", "scripts/run_cycle2c.sh",
    "scripts/preflight_cycle2c.py", "scripts/collect_cycle2c_artifacts.py",
    "ccdn/metrics/plasticity.py", "ccdn/metrics/resources.py",
    "ccdn/utils/reproducibility.py", "pyproject.toml",
    "tests/test_cycle2c_protocol.py", "tests/test_cycle2c_orchestrator.py",
    "configs/cycle2c/static_sparse.yaml", "configs/cycle2c/selective_reset.yaml",
    "configs/cycle2c/rigl_reference.yaml", "configs/cycle2c/ccdn_0a.yaml",
    "configs/cycle2c/protocol.yaml",
)


def sha_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def file_hashes():
    return {name: sha_file(ROOT / name) for name in LOCK_FILES}


def protocol_digest(hashes=None):
    hashes = hashes or file_hashes()
    body = json.dumps(hashes, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(body).hexdigest()


def git_info():
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    branch = subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT, text=True).strip()
    status = subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=normal"], cwd=ROOT, text=True)
    return commit, branch, clean_status(status), status


def clean_status(status_text):
    return not bool(status_text.strip())


def tensor_digest(state, masks_only=False):
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        if masks_only and not name.endswith(".mask"):
            continue
        if torch.is_tensor(tensor):
            digest.update(name.encode())
            digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def acquire_lock(path: Path, commit: str, protocol_hash: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"pid": os.getpid(), "git_sha": commit, "protocol_hash": protocol_hash,
               "started_at": datetime.now(timezone.utc).isoformat()}
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    except FileExistsError:
        try:
            current = json.loads(path.read_text())
            os.kill(int(current["pid"]), 0)
        except ProcessLookupError:
            path.unlink(missing_ok=True)
            return acquire_lock(path, commit, protocol_hash)
        except (ValueError, KeyError, json.JSONDecodeError):
            raise RuntimeError(f"unreadable run lock; inspect before removing: {path}")
        raise RuntimeError(f"another Cycle 2C process owns run lock (PID {current['pid']})")
    with os.fdopen(fd, "w") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")


def verify_lock_file(hashes):
    path = ROOT / "configs/cycle2c/protocol_lock.json"
    stored = json.loads(path.read_text())
    validate_lock_hashes(stored.get("sha256", {}), hashes)
    if stored.get("protocol_hash") != protocol_digest(hashes):
        raise RuntimeError("protocol_lock.json protocol_hash does not match its source/config hashes")


def validate_lock_hashes(stored, actual):
    if stored != actual:
        changed = sorted(set(stored) | set(actual))
        changed = [name for name in changed if stored.get(name) != actual.get(name)]
        raise RuntimeError("protocol source/config hash mismatch: " + ", ".join(changed))


def check_mnist_and_sequence():
    from ccdn.streams.online_permuted_mnist import OnlinePermutedMNIST
    stream = OnlinePermutedMNIST(seed=101, root="./data", download=False)
    if stream.data_source != "mnist" or len(stream.labels) != 60000 or stream.input_size != 784:
        raise RuntimeError("Cycle 2C requires verified 60,000-example real MNIST with 784 inputs")
    if not bool(torch.isfinite(stream.images).all()) or int(stream.labels.min()) != 0 or int(stream.labels.max()) != 9:
        raise RuntimeError("MNIST labels/images failed validity checks")
    sequence_hash = stream.sequence_digest(150, 60000)
    if sequence_hash != EXPECTED_STREAM:
        raise RuntimeError(f"task stream digest mismatch: {sequence_hash}")
    return stream, sequence_hash


def verify_starting_states(stream):
    from ccdn.experiments.online_permuted_mnist import _build_online_learner, _device
    from ccdn.utils.reproducibility import seed_everything
    hashes, mask_hashes, counts = {}, {}, {}
    for method in METHODS:
        cfg = yaml.safe_load((ROOT / f"configs/cycle2c/{method}.yaml").read_text())
        seed_everything(101, True)
        model, _algorithm, _optimizer = _build_online_learner(cfg, stream, _device("cpu"))
        hashes[method] = tensor_digest(model.state_dict())
        mask_hashes[method] = tensor_digest(model.state_dict(), masks_only=True)
        counts[method] = [int(layer.mask.sum()) for layer in model.layers]
    if len(set(hashes.values())) != 1 or len(set(mask_hashes.values())) != 1:
        raise RuntimeError("paired sparse methods have nonidentical starting states")
    expected = [52685, 22579, 22579, 672]
    if any(value != expected for value in counts.values()):
        raise RuntimeError(f"active edge counts differ from frozen protocol: {counts}")
    return hashes, mask_hashes, counts[METHODS[0]]


def atomic_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".{os.getpid()}.{time.time_ns()}.tmp")
    with temp.open("w", encoding="utf8") as handle:
        handle.write(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    temp.replace(path)


def load_valid_checkpoint(path: Path):
    """Load an official rolling checkpoint, rejecting partial/invalid payloads."""
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        required = {"model", "optimizer", "algorithm", "completed_task_index",
                    "lifetime_examples_seen", "rng_state", "config"}
        if not isinstance(payload, dict) or not required.issubset(payload):
            raise ValueError("checkpoint payload is incomplete")
        return payload
    except Exception:
        return None


def select_resume_checkpoint(result_dir: Path):
    """Select latest, then previous; never treats an abandoned temp as state."""
    latest = result_dir / "checkpoint_latest.pt"
    previous = result_dir / "checkpoint_previous.pt"
    temp = result_dir / "checkpoint_latest.pt.tmp"
    if latest.exists():
        payload = load_valid_checkpoint(latest)
        if payload is not None:
            if temp.exists():
                temp.unlink()
                print(f"[CYCLE2C] removed abandoned temporary checkpoint: {temp}")
            return latest, payload, False
    else:
        payload = None
    previous_payload = load_valid_checkpoint(previous) if previous.exists() else None
    if previous_payload is not None:
        print(f"[CYCLE2C] RECOVERY: using checkpoint_previous.pt: {previous}")
        # Remove a corrupt/partial latest so the next rotation cannot overwrite
        # the valid previous checkpoint with that unusable file.
        latest.unlink(missing_ok=True)
        temp.unlink(missing_ok=True)
        return previous, previous_payload, True
    if temp.exists():
        temp.unlink()
        print(f"[CYCLE2C] removed abandoned temporary checkpoint: {temp}")
    if latest.exists() or previous.exists():
        raise RuntimeError("no valid latest or previous rolling checkpoint; refusing fresh run")
    raise RuntimeError("result directory has no authoritative rolling checkpoint; refusing fresh run")


def validate_approved_sha(approved_sha, actual_sha):
    if not approved_sha:
        raise RuntimeError("ABORTED_PROTOCOL_MISMATCH: reviewer-approved SHA is required")
    if len(approved_sha) != 40 or any(ch not in "0123456789abcdefABCDEF" for ch in approved_sha):
        raise RuntimeError("ABORTED_PROTOCOL_MISMATCH: reviewer-approved SHA must be a full 40-character SHA")
    if approved_sha.lower() != actual_sha.lower():
        raise RuntimeError(f"ABORTED_PROTOCOL_MISMATCH: approved SHA {approved_sha} != HEAD {actual_sha}")
    return True


def read_heartbeat(path: Path):
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def print_heartbeat(method, heartbeat):
    task = int(heartbeat["completed_task_index"]) + 1
    updates = int(heartbeat["optimizer_updates"])
    expected = heartbeat.get("expected_active_edges_by_layer", [])
    current = heartbeat.get("current_active_edges_by_layer", [])
    edge_health = current == expected
    print(f"[CYCLE2C] {method}\n"
          f"Task {task}/150\n"
          f"Accuracy: {float(heartbeat['online_accuracy']):.4f}\n"
          f"Updates: {updates:,} / 9,000,000\n"
          f"Expected active edges: {expected}\n"
          f"Current active edges:  {current}\n"
          f"Edge budget: {'PASS' if edge_health else 'FAIL'}\n"
          f"Finite state: {'PASS' if heartbeat.get('finite_state') else 'FAIL'}\n"
          f"Checkpoint: {'PASS' if heartbeat.get('checkpoint_valid') else 'FAIL'}",
          flush=True)
    return edge_health and bool(heartbeat.get("finite_state")) and bool(heartbeat.get("checkpoint_valid"))


def run_child(method, run_root, status_path, protocol_hash, commit):
    method_root = run_root / method
    method_root.mkdir(parents=True, exist_ok=True)
    result_dir = method_root / "result"
    config = yaml.safe_load((ROOT / f"configs/cycle2c/{method}.yaml").read_text())
    config["protocol_hash"] = protocol_hash
    config["git_sha"] = commit
    config.setdefault("runtime", {})["status_path"] = str(status_path)
    config_path = method_root / "config.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    checkpoint = result_dir / "checkpoint_latest.pt"
    completed_summary = None
    selected_checkpoint = None
    saved = None
    recovered_previous = False
    if result_dir.exists():
        if (result_dir / "summary.json").exists():
            candidate = json.loads((result_dir / "summary.json").read_text())
            if candidate.get("tasks_completed") == 150:
                completed_summary = candidate
        selected_checkpoint, saved, recovered_previous = select_resume_checkpoint(result_dir)
        if selected_checkpoint is None:
            raise RuntimeError(f"existing result directory has no resumable rolling checkpoint: {result_dir}")
        expected = {"git_sha": commit, "protocol_hash": protocol_hash, "method": method,
                    "config": config, "task_sequence_sha256": EXPECTED_STREAM}
        try:
            validate_resume_metadata(saved, expected)
        except RuntimeError as exc:
            raise RuntimeError(f"refusing to resume mismatched run: {exc}")
        if completed_summary is not None and not recovered_previous and saved.get("completed_task_index") == 149:
            return completed_summary
    cmd = [sys.executable, "-m", "ccdn.experiments.online_permuted_mnist",
           "--config", str(config_path), "--output-dir", str(result_dir)]
    if selected_checkpoint is not None: cmd += ["--resume-from", str(selected_checkpoint)]
    log_path = run_root / "terminal.log"
    with open(log_path, "a", encoding="utf8") as log:
        proc = subprocess.Popen(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        heartbeat = result_dir / "heartbeat.json"
        last_heartbeat = time.time()
        last_reported_task = -1
        while proc.poll() is None:
            time.sleep(1)
            current_heartbeat = read_heartbeat(heartbeat) if heartbeat.exists() else None
            if current_heartbeat is not None:
                task_index = int(current_heartbeat.get("completed_task_index", -1))
                if task_index > last_reported_task:
                    healthy = print_heartbeat(method, current_heartbeat)
                    last_reported_task = task_index
                    last_heartbeat = time.time()
                    if not healthy:
                        proc.terminate()
                        try: proc.wait(timeout=30)
                        except subprocess.TimeoutExpired: proc.kill()
                        raise RuntimeError("task-boundary heartbeat health check failed")
            # 30-minute timeout also covers a child that never writes its first heartbeat.
            if time.time() - last_heartbeat > 1800:
                proc.terminate()
                try: proc.wait(timeout=30)
                except subprocess.TimeoutExpired: proc.kill()
                atomic_json(status_path, {"state": "INTERRUPTED", "method": method,
                                          "reason": "30-minute task-boundary heartbeat timeout"})
                raise RuntimeError("task-boundary heartbeat watchdog expired")
        if proc.returncode != 0:
            current_status = json.loads(status_path.read_text()) if status_path.exists() else {}
            if current_status.get("state") not in {"ABORTED_NONFINITE", "ABORTED_INVARIANT", "ABORTED_PROTOCOL_MISMATCH"}:
                atomic_json(status_path, {"state": "ABORTED_INVARIANT", "method": method,
                                          "return_code": proc.returncode})
            raise RuntimeError(f"Cycle 2C learner child failed: {method}, rc={proc.returncode}")
    return json.loads((result_dir / "summary.json").read_text())


def static_gate(summary):
    return bool(summary.get("tasks_completed") == 150
                and summary.get("plasticity_drop", -1) >= 0.05
                and summary.get("peak_window_end", 149) < 130
                and summary.get("slope_from_peak_window_end_to_final", 0) < 0
                and summary.get("slope_tasks_100_149", 0) < 0)


def validate_resume_metadata(saved, expected):
    keys = ("git_sha", "protocol_hash", "method", "config", "task_sequence_sha256")
    mismatch = [key for key in keys if saved.get(key) != expected.get(key)]
    if mismatch:
        raise RuntimeError("refusing to resume mismatched run fields: " + ", ".join(mismatch))


def gated_sequence(run_method, cached=None):
    """Run the static gate first; dynamic methods are unreachable on failure."""
    cached = cached or {}
    summaries = {"static_sparse": cached.get("static_sparse") or run_method("static_sparse")}
    if not static_gate(summaries["static_sparse"]):
        return "STATIC_GATE_FAILED", summaries
    for method in METHODS[1:]:
        summaries[method] = cached.get(method) or run_method(method)
    return "COMPLETE", summaries


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("approved_sha", nargs="?", help="full reviewer-approved HEAD SHA")
    args = parser.parse_args(argv)
    try:
        commit, branch, clean, status = git_info()
        validate_approved_sha(args.approved_sha, commit)
        # Store the canonical Git spelling so the manifest field exactly equals
        # git_commit even if the caller supplied uppercase hexadecimal.
        args.approved_sha = commit
        hashes = file_hashes()
        verify_lock_file(hashes)
        protocol_hash = protocol_digest(hashes)
        if branch != "cycle-2c-ccdn-failure-regime" or not clean:
            raise RuntimeError(f"run requires clean frozen branch; branch={branch}, clean={clean}, status={status!r}")
    except Exception as exc:
        fail_root = ROOT / "results/cycle2c"
        fail_root.mkdir(parents=True, exist_ok=True)
        atomic_json(fail_root / "precheck_status.json", {"state": "ABORTED_PROTOCOL_MISMATCH",
                                                           "reason": str(exc),
                                                           "timestamp": datetime.now(timezone.utc).isoformat()})
        raise
    runtime_root = ROOT / "results/cycle2c"
    lock_path = runtime_root / ".run_lock"
    acquire_lock(lock_path, commit, protocol_hash)
    run_id = f"seed101_{commit[:8]}_{protocol_hash[:8]}"
    run_root = runtime_root / run_id
    run_root.mkdir(parents=True, exist_ok=True)
    status_path = run_root / "status.json"
    try:
        atomic_json(status_path, {"state": "PRECHECK"})
        stream, sequence_hash = check_mnist_and_sequence()
        model_hashes, mask_hashes, counts = verify_starting_states(stream)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        manifest = {
            "run_id": run_id, "timestamp": datetime.now(timezone.utc).isoformat(),
            "git_commit": commit, "git_branch": branch, "git_dirty": False,
            "reviewer_approved_git_sha": args.approved_sha,
            "python_version": platform.python_version(), "pytorch_version": torch.__version__,
            "device": device, "platform": platform.platform(), "protocol_hash": protocol_hash,
            "protocol_source_sha256": hashes, "task_sequence_sha256": sequence_hash,
            "data_source": stream.data_source, "synthetic_fallback": False,
            "seed": 101, "architecture": [784, 336, 336, 336, 10], "density": .2,
            "logical_active_parameters": sum(counts) + 1018, "active_edges_by_layer": counts,
            "optimizer": "SGD", "learning_rate": .003, "structural_interval": 8192,
            "turnover_fraction": .05, "utility_decay_formula": "0.99 ** (1 / 128)",
            "utility_decay_numeric": .99 ** (1 / 128),
            "gate_definition": "drop>=.05; peak before tasks 130-149; postpeak slope<0; tasks100-149 slope<0",
            "initial_model_state_sha256": model_hashes,
            "initial_mask_sha256": mask_hashes,
        }
        atomic_json(run_root / "run_manifest.json", manifest)
        print(f"[CYCLE2C] Git SHA: {commit}\nProtocol: VERIFIED\nMNIST: VERIFIED\nTask stream: VERIFIED\nInitial active parameters: {manifest['logical_active_parameters']:,}")
        def launch(method):
            state = "RUNNING_STATIC_GATE" if method == "static_sparse" else f"RUNNING_{method.upper()}"
            atomic_json(status_path, {"state": state, "method": method})
            summary = run_child(method, run_root, status_path, protocol_hash, commit)
            print(f"[CYCLE2C] {method}: final20={summary.get('final_20_task_accuracy')} drop={summary.get('plasticity_drop')}")
            if method == "static_sparse" and static_gate(summary):
                atomic_json(status_path, {"state": "STATIC_GATE_PASSED", "summary": summary})
            return summary
        run_state, summaries = gated_sequence(launch)
        if run_state == "STATIC_GATE_FAILED":
            atomic_json(status_path, {"state": run_state, "summary": summaries["static_sparse"]})
            print("[CYCLE2C] STATIC_GATE_FAILED; no comparison methods launched.")
            return 0
        atomic_json(run_root / "method_summaries.json", summaries)
        atomic_json(status_path, {"state": "COMPLETE", "methods": list(summaries)})
    except FloatingPointError as exc:
        atomic_json(status_path, {"state": "ABORTED_NONFINITE", "reason": str(exc)})
        raise
    except KeyboardInterrupt:
        atomic_json(status_path, {"state": "INTERRUPTED", "reason": "keyboard interrupt"})
        raise
    except Exception as exc:
        if not status_path.exists() or json.loads(status_path.read_text()).get("state") in {"PRECHECK", "RUNNING_STATIC_GATE", "RUNNING_SELECTIVE_RESET", "RUNNING_RIGL_REFERENCE", "RUNNING_CCDN_0A"}:
            atomic_json(status_path, {"state": "ABORTED_PROTOCOL_MISMATCH", "reason": str(exc)})
        raise
    finally:
        lock_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
