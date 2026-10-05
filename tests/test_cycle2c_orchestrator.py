import pytest

from scripts.run_cycle2c import (METHODS, clean_status, gated_sequence,
                                 static_gate, validate_lock_hashes,
                                 validate_resume_metadata)


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
