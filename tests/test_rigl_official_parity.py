from pathlib import Path
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]


def test_rigl_reference_matches_pinned_sparse_optimizer_base():
    if not (ROOT / ".external/official_baselines/rigl/rigl/sparse_optimizers_base.py").exists():
        pytest.skip("pinned external source not fetched")
    python = ROOT / ".external/envs/rigl/bin/python"
    if not python.exists():
        pytest.skip("isolated official RigL environment not installed")
    import os
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{ROOT}:{ROOT / '.external/runtime'}"
    env["TF_USE_LEGACY_KERAS"] = "1"
    env["TF_CPP_MIN_LOG_LEVEL"] = "2"
    subprocess.run([str(python), "scripts/validate_rigl_parity.py"],
                   cwd=ROOT, env=env, check=True)
    import json
    result = json.loads((ROOT / "results/official_validation/rigl_parity.json").read_text())
    assert result["weights_match"]
    assert result["exact_edge_conservation"]
    assert result["new_edge_optimizer_state_reset"]
