import json
import inspect
import time

import pytest
import torch

from ccdn.experiments.online_permuted_mnist import _learner_step
from scripts.bootstrap_cycle2d import CYCLE2C_EXECUTION_SHA
from scripts.run_cycle2d import (
    TaskWatchdogTimeout, _finite, acquire_run_lock, atomic_json,
    discover_cycle2c_run, load_valid_checkpoint, select_resume_checkpoint,
    task_watchdog, validate_active_counts, validate_approved_sha,
    validate_branch_clean, validate_lock_payload, _heartbeat,
)


def valid_checkpoint(step=150):
    return {"model": {}, "optimizer": {}, "algorithm": {}, "rng_state": {},
            "completed_task_index": step-1, "lifetime_examples_seen": step*60000,
            "config": {}, "protocol_hash": "p", "git_sha": "g",
            "method": "static_sparse", "task_sequence_sha256": "s",
            "source_checkpoint_sha256": "source"}


def save(path, payload):
    torch.save(payload, path)


@pytest.mark.parametrize("approved,head,good", [
    (None, "a"*40, False), ("abc", "a"*40, False),
    ("b"*40, "a"*40, False), ("a"*40, "a"*40, True),
])
def test_approved_sha_is_required_full_and_matching(approved, head, good):
    if good:
        assert validate_approved_sha(approved, head)
    else:
        with pytest.raises(RuntimeError, match="ABORTED_PROTOCOL_MISMATCH"):
            validate_approved_sha(approved, head)


def test_branch_and_clean_worktree_required():
    assert validate_branch_clean("cycle-2d-sparse-lifetime-calibration", True)
    with pytest.raises(RuntimeError, match="wrong branch"):
        validate_branch_clean("main", True)
    with pytest.raises(RuntimeError, match="dirty worktree"):
        validate_branch_clean("cycle-2d-sparse-lifetime-calibration", False, " M file")


def test_cycle2d_learner_step_never_receives_task_metadata():
    assert list(inspect.signature(_learner_step).parameters) == [
        "model", "algorithm", "optimizer", "x", "y"]


def test_protocol_lock_payload_accepts_exact_and_rejects_mismatch():
    hashes = {"code.py": "abc"}
    from scripts.run_cycle2d import cycle2d_protocol_digest
    lock = {"sha256": hashes, "protocol_hash": cycle2d_protocol_digest(hashes)}
    assert validate_lock_payload(lock, hashes) == lock["protocol_hash"]
    with pytest.raises(RuntimeError, match="protocol lock mismatch"):
        validate_lock_payload({**lock, "protocol_hash": "wrong"}, hashes)


def test_cycle2c_source_discovery_refuses_ambiguity(tmp_path):
    for name in ("one", "two"):
        run = tmp_path / name
        run.mkdir()
        (run / "run_manifest.json").write_text(json.dumps({"git_commit": CYCLE2C_EXECUTION_SHA}))
    with pytest.raises(RuntimeError, match="found 2"):
        discover_cycle2c_run(tmp_path)


def test_cycle2c_source_discovery_requires_a_unique_match(tmp_path):
    (tmp_path / "run").mkdir()
    (tmp_path / "run/run_manifest.json").write_text(json.dumps({"git_commit": CYCLE2C_EXECUTION_SHA}))
    assert discover_cycle2c_run(tmp_path) == tmp_path / "run"
    (tmp_path / "run/run_manifest.json").write_text("{")
    with pytest.raises(RuntimeError, match="found 0"):
        discover_cycle2c_run(tmp_path)


def test_checkpoint_latest_preferred_and_abandoned_temp_removed(tmp_path):
    latest, previous = tmp_path / "checkpoint_latest.pt", tmp_path / "checkpoint_previous.pt"
    save(previous, valid_checkpoint(150)); save(latest, valid_checkpoint(151))
    temp = tmp_path / "checkpoint_latest.pt.tmp"; temp.write_bytes(b"partial")
    path, payload, recovered = select_resume_checkpoint(tmp_path)
    assert path == latest and payload["completed_task_index"] == 150
    assert recovered is False and not temp.exists()


@pytest.mark.parametrize("bad_latest", [True, False])
def test_corrupt_or_missing_latest_recovers_previous(tmp_path, bad_latest):
    latest, previous = tmp_path / "checkpoint_latest.pt", tmp_path / "checkpoint_previous.pt"
    save(previous, valid_checkpoint(150))
    if bad_latest: latest.write_bytes(b"broken")
    path, payload, recovered = select_resume_checkpoint(tmp_path)
    assert path == previous and payload["completed_task_index"] == 149
    assert recovered is True and not latest.exists()


def test_two_bad_or_missing_checkpoints_never_restart(tmp_path):
    (tmp_path / "checkpoint_latest.pt").write_bytes(b"bad")
    (tmp_path / "checkpoint_previous.pt").write_bytes(b"bad")
    with pytest.raises(RuntimeError, match="refusing to restart"):
        select_resume_checkpoint(tmp_path)
    (tmp_path / "checkpoint_latest.pt").unlink()
    (tmp_path / "checkpoint_previous.pt").unlink()
    with pytest.raises(RuntimeError, match="refusing to restart"):
        select_resume_checkpoint(tmp_path)


def test_finite_state_failure_is_detected_and_active_edge_mismatch_aborts():
    model = torch.nn.Linear(2, 2)
    with torch.no_grad(): model.weight[0, 0] = float("nan")
    class Algorithm:
        def state_dict(self): return {"global_step": 0}
    assert not _finite(model, Algorithm())
    with pytest.raises(RuntimeError, match="active edge budget mismatch"):
        validate_active_counts([1, 2, 3, 4])


def test_heartbeat_reports_checkpoint_health_and_atomic_json(tmp_path):
    checkpoint_path = tmp_path / "checkpoint_latest.pt"
    save(checkpoint_path, valid_checkpoint())
    heartbeat = _heartbeat("static_sparse", 149,
                           {"online_accuracy": .8}, 9000000,
                           [52685, 22579, 22579, 672],
                           [52685, 22579, 22579, 672], 200, None,
                           checkpoint_path, True)
    assert heartbeat["checkpoint_pass"] and heartbeat["edge_budget_pass"]
    path = tmp_path / "heartbeat.json"
    atomic_json(path, heartbeat)
    assert json.loads(path.read_text())["task_number"] == 150
    assert not list(tmp_path.glob("heartbeat.json.*.tmp"))
    checkpoint_path.write_bytes(b"corrupt")
    failed = _heartbeat("static_sparse", 149, {"online_accuracy": .8}, 9000000,
                        [52685]*4, [52685]*4, 200, None, checkpoint_path, True)
    assert failed["checkpoint_pass"] is False


def test_stale_run_lock_only_recovers_when_pid_is_gone(tmp_path):
    lock = tmp_path / "lock.json"
    dead_pid = 99999999
    lock.write_text(json.dumps({"pid": dead_pid}))
    result = acquire_run_lock(lock, "g", "p")
    assert result["pid"] != dead_pid
    lock.unlink()


def test_task_watchdog_interrupts_a_silent_task():
    with pytest.raises(TaskWatchdogTimeout):
        with task_watchdog(.03): time.sleep(.08)
