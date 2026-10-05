from pathlib import Path
import json
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_online_runner_sgd_matches_pinned_official_backprop():
    if not (ROOT / ".external/official_baselines/loss-of-plasticity/lop/algos/bp.py").exists():
        pytest.skip("pinned external source not fetched")
    python = ROOT / ".external/envs/cbp/bin/python"
    if not python.exists():
        pytest.skip("isolated official CBP environment not installed")
    subprocess.run([str(python), "scripts/validate_online_backprop.py"],
                   cwd=ROOT, check=True)
    result = json.loads((ROOT / "review_artifacts/cycle_2b/backprop_runner_parity.json").read_text())
    assert result["predictions_before_update_match"]
    assert result["losses_match"]
    assert result["weight_updates_match"]
    assert result["bias_updates_match"]
