from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]


def test_cbp_reference_parity_against_pinned_source():
    if not (ROOT / ".external/official_baselines/loss-of-plasticity/lop/algos/gnt.py").exists():
        pytest.skip("pinned external source not fetched")
    env_python = ROOT / ".external/envs/cbp/bin/python"
    if not env_python.exists():
        pytest.skip("isolated official CBP environment not installed")
    subprocess.run([str(env_python), "scripts/validate_official_cbp.py"],
                   cwd=ROOT, check=True)
    import json
    result = json.loads((ROOT / "review_artifacts/cycle_2v/cbp_parity.json").read_text())
    assert result["numerical_tensor_comparisons"] is True
    assert result["replacement_count"] > 0
