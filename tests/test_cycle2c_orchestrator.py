import pytest
import torch

from scripts.run_cycle2c import (METHODS, clean_status, gated_sequence,
                                 static_gate, validate_lock_hashes,
                                 validate_resume_metadata, load_valid_checkpoint,
                                 select_resume_checkpoint, validate_approved_sha,
                                 atomic_json, print_heartbeat, read_heartbeat)


def passing_summary():
    return {"tasks_completed": 150, "plasticity_drop": .06,
            "peak_window_end": 70, "slope_from_peak_window_end_to_final": -.001,
            "slope_tasks_100_149": -.002}


def test_fake_static_sparse_gate_failure_launches_no_dynamic_methods():
    launched = []
    state, summaries = gated_sequence(lambda name: launched.append(name) or {"tasks_completed": 150})
    assert state == "STATIC_GATE_FAILED"
    assert launched == ["static_sparse"]
    assert list(summaries) == ["static_sparse"]


def test_fake_static_sparse_gate_pass_launches_frozen_order():
    launched = []
    def runner(name):
        launched.append(name)
        return passing_summary()
    state, _ = gated_sequence(runner)
    assert state == "COMPLETE"
    assert launched == list(METHODS)


def test_invariant_failure_aborts_without_later_methods():
    launched = []
    def runner(name):
        launched.append(name)
        if name == "rigl_reference":
            raise RuntimeError("active edge mismatch")
        return passing_summary()
    with pytest.raises(RuntimeError, match="active edge mismatch"):
        gated_sequence(runner)
    assert launched == ["static_sparse", "selective_reset", "rigl_reference"]


def test_resume_skips_completed_methods_and_refuses_mismatch():
    cached = {"static_sparse": passing_summary(), "selective_reset": {"tasks_completed": 150}}
    launched = []
    state, _ = gated_sequence(lambda name: launched.append(name) or {"tasks_completed": 150}, cached)
    assert state == "COMPLETE"
    assert launched == ["rigl_reference", "ccdn_0a"]
    identity = {"git_sha": "a", "protocol_hash": "b", "method": "rigl_reference",
                "config": {"seed": 101}, "task_sequence_sha256": "c"}
    validate_resume_metadata(identity, identity)
    with pytest.raises(RuntimeError, match="config"):
        validate_resume_metadata(identity, {**identity, "config": {"seed": 2}})


def test_protocol_hash_and_git_clean_checks():
    validate_lock_hashes({"a": "1"}, {"a": "1"})
    with pytest.raises(RuntimeError, match="a"):
        validate_lock_hashes({"a": "1"}, {"a": "2"})
    assert clean_status("")
    assert not clean_status(" M file.py\n")


def test_gate_requires_all_preregistered_conditions():
    assert static_gate(passing_summary())
    assert not static_gate({**passing_summary(), "slope_tasks_100_149": 0.0})


def checkpoint_payload(step=1):
    return {"model": {}, "optimizer": {}, "algorithm": {},
            "completed_task_index": step - 1, "lifetime_examples_seen": step,
            "rng_state": {}, "config": {}}


def write_checkpoint(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def test_resume_uses_valid_latest_and_removes_abandoned_temp(tmp_path):
    latest = tmp_path / "checkpoint_latest.pt"
    previous = tmp_path / "checkpoint_previous.pt"
    temp = tmp_path / "checkpoint_latest.pt.tmp"
    write_checkpoint(previous, checkpoint_payload(2))
    write_checkpoint(latest, checkpoint_payload(3))
    temp.write_bytes(b"partial")
    selected, payload, recovered = select_resume_checkpoint(tmp_path)
    assert selected == latest and payload["lifetime_examples_seen"] == 3
    assert recovered is False and not temp.exists()


@pytest.mark.parametrize("latest_state", ["corrupt", "missing"])
def test_resume_recovers_valid_previous_when_latest_unavailable(tmp_path, latest_state):
    latest = tmp_path / "checkpoint_latest.pt"
    previous = tmp_path / "checkpoint_previous.pt"
    write_checkpoint(previous, checkpoint_payload(2))
    if latest_state == "corrupt": latest.write_bytes(b"partial")
    selected, payload, recovered = select_resume_checkpoint(tmp_path)
    assert selected == previous and payload["lifetime_examples_seen"] == 2
    assert recovered is True and not latest.exists()


def test_resume_refuses_two_corrupt_checkpoints(tmp_path):
    (tmp_path / "checkpoint_latest.pt").write_bytes(b"bad")
    (tmp_path / "checkpoint_previous.pt").write_bytes(b"bad")
    with pytest.raises(RuntimeError, match="no valid latest or previous"):
        select_resume_checkpoint(tmp_path)


def test_resume_refuses_no_checkpoint_even_with_abandoned_temp(tmp_path):
    (tmp_path / "checkpoint_latest.pt.tmp").write_bytes(b"partial")
    with pytest.raises(RuntimeError, match="no authoritative"):
        select_resume_checkpoint(tmp_path)
    assert not (tmp_path / "checkpoint_latest.pt.tmp").exists()


def test_resume_refuses_latest_and_previous_both_missing(tmp_path):
    with pytest.raises(RuntimeError, match="no authoritative"):
        select_resume_checkpoint(tmp_path)


def test_checkpoint_payload_validation_rejects_incomplete_state(tmp_path):
    path = tmp_path / "checkpoint_latest.pt"
    torch.save({"model": {}}, path)
    assert load_valid_checkpoint(path) is None


def test_reviewer_approved_sha_is_required_and_must_match():
    sha = "a" * 40
    assert validate_approved_sha(sha, sha)
    with pytest.raises(RuntimeError, match="!= HEAD"):
        validate_approved_sha("b" * 40, sha)
    with pytest.raises(RuntimeError, match="required"):
        validate_approved_sha(None, sha)


def test_parent_atomic_status_and_transient_heartbeat_read(tmp_path, capsys):
    status = tmp_path / "status.json"
    atomic_json(status, {"state": "RUNNING"})
    assert status.read_text().startswith("{")
    assert list(tmp_path.iterdir()) == [status]
    heartbeat = tmp_path / "heartbeat.json"
    heartbeat.write_text("{")
    assert read_heartbeat(heartbeat) is None
    heartbeat_value = {"completed_task_index": 36, "optimizer_updates": 2220000,
                       "online_accuracy": .8912,
                       "expected_active_edges_by_layer": [52685, 22579, 22579, 672],
                       "current_active_edges_by_layer": [52685, 22579, 22579, 672],
                       "finite_state": True, "checkpoint_valid": True}
    assert print_heartbeat("static_sparse", heartbeat_value)
    text = capsys.readouterr().out
    assert "Task 37/150" in text and "Updates: 2,220,000 / 9,000,000" in text
    assert "Edge budget: PASS" in text and "Checkpoint: PASS" in text
