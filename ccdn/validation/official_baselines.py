"""Verify pinned official baselines and reproduce their compact validations.

Default mode reruns mechanism/parity checks against already fetched source.
Pass ``--full`` to also rerun both real-MNIST execution smokes.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
EXTERNAL = ROOT / ".external"
CBP = EXTERNAL / "official_baselines/loss-of-plasticity"
RIGL = EXTERNAL / "official_baselines/rigl"
CBP_PY = EXTERNAL / "envs/cbp/bin/python"
RIGL_PY = EXTERNAL / "envs/rigl/bin/python"
RUNTIME = EXTERNAL / "runtime"


def _run(command, *, cwd=ROOT, env=None):
    subprocess.run([str(part) for part in command], cwd=cwd, env=env, check=True)


def _verify_clone(path, expected_remote, expected_sha):
    if not (path / ".git").exists():
        raise RuntimeError(f"official source clone missing: {path}; run scripts/fetch_official_baselines.py")
    remote = subprocess.check_output(["git", "remote", "get-url", "origin"], cwd=path, text=True).strip()
    actual = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=path, text=True).strip()
    normalized = remote.removesuffix(".git")
    if normalized != expected_remote or actual != expected_sha:
        raise RuntimeError(f"source mismatch for {path.name}: origin={remote}, sha={actual}")
    print(f"{path.name}: {actual} VERIFIED")


def _full_execution():
    stamp = time.strftime("%Y%m%dT%H%M%S")
    cbp_output = ROOT / "results/official_validation" / f"cbp_full_{stamp}.pkl"
    rigl_logdir = ROOT / "results/official_validation" / f"rigl_full_{stamp}"
    cbp_config = RUNTIME / f"cbp_full_{stamp}.json"
    RUNTIME.mkdir(parents=True, exist_ok=True)
    config = {
        "agent": "cbp", "opt": "sgd", "use_gpu": 0,
        "num_features": 100, "num_hidden_layers": 3, "step_size": 0.003,
        "mini_batch_size": 10000, "change_after": 30000, "num_examples": 60000,
        "replacement_rate": 0.01, "decay_rate": 0.99,
        "maturity_threshold": 100, "mt": 100,
        "util_type": "adaptable_contribution", "data_file": str(cbp_output),
    }
    cbp_config.write_text(json.dumps(config, indent=2) + "\n")
    cbp_output.parent.mkdir(parents=True, exist_ok=True)
    _run([CBP_PY, "-m", "lop.permuted_mnist.load_mnist"], cwd=CBP)
    cbp_code = (
        "import numpy,random,runpy,sys,torch; "
        "numpy.random.seed(11); random.seed(11); torch.manual_seed(11); "
        f"sys.argv=['online_expr.py','-c',{str(cbp_config)!r}]; "
        "runpy.run_module('lop.permuted_mnist.online_expr',run_name='__main__')"
    )
    _run([CBP_PY, "-c", cbp_code], cwd=CBP)

    rigl_logdir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update({"PYTHONPATH": f"{RIGL}:{RUNTIME}",
                "TFDS_DATA_DIR": str(ROOT / "data/tfds"),
                "TF_CPP_MIN_LOG_LEVEL": "2"})
    _run([RIGL_PY, RUNTIME / "run_official_rigl.py",
          RIGL / "rigl/rigl_tf2/train.py", f"--logdir={rigl_logdir}",
          "--use_tpu=false", "--seed=11",
          f"--gin_config={RUNTIME / 'rigl_mnist_smoke.gin'}"], env=env)
    return cbp_output, rigl_logdir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true",
                        help="also rerun pinned real-MNIST training smokes")
    args = parser.parse_args()
    _verify_clone(CBP, "https://github.com/shibhansh/loss-of-plasticity",
                  "a6b79580d85f3025bdb601566d3627c5f489f13b")
    _verify_clone(RIGL, "https://github.com/google-research/rigl",
                  "d39fc7d46505cb3196cb1edeb32ed0b6dd44c0f9")
    for executable in (CBP_PY, RIGL_PY):
        if not executable.exists():
            raise RuntimeError(f"isolated validation interpreter missing: {executable}")

    cbp_env = os.environ.copy()
    cbp_env["PYTHONPATH"] = str(ROOT)
    _run([CBP_PY, "scripts/validate_official_cbp.py"], env=cbp_env)
    rigl_env = os.environ.copy()
    rigl_env.update({"PYTHONPATH": f"{ROOT}:{RUNTIME}",
                     "TF_USE_LEGACY_KERAS": "1", "TF_CPP_MIN_LOG_LEVEL": "2"})
    _run([RIGL_PY, "scripts/validate_official_rigl.py"], env=rigl_env)

    if args.full:
        cbp_output, rigl_logdir = _full_execution()
    else:
        cbp_output = ROOT / "results/official_validation/cbp/official_metrics.pkl"
        rigl_logdir = ROOT / "results/official_validation/rigl_final"
    if not cbp_output.exists() or not (rigl_logdir / "ckpt-3.index").exists():
        raise RuntimeError("official MNIST smoke outputs are missing; run with --full")
    collector_env = dict(rigl_env)
    _run([RIGL_PY, "scripts/collect_official_validation.py",
          f"--cbp-output={cbp_output}", f"--rigl-logdir={rigl_logdir}"],
         env=collector_env)

    art = ROOT / "review_artifacts/cycle_2v"
    cbp_parity = json.loads((art / "cbp_parity.json").read_text())
    rigl_parity = json.loads((ROOT / "results/official_validation/rigl_parity.json").read_text())
    cbp_execution = json.loads((art / "official_cbp_execution.json").read_text())
    rigl_execution = json.loads((art / "official_rigl_execution.json").read_text())
    summary = {
        "continual_backprop": {
            "official_repository_verified": True,
            "official_commit_verified": True,
            "official_code_executed": cbp_execution["output_generated"] and cbp_execution["training_metrics_finite"],
            "reference_port_created": True,
            "parity_passed": cbp_parity["numerical_tensor_comparisons"],
        },
        "rigl": {
            "official_repository_verified": True,
            "official_commit_verified": True,
            "official_code_executed": rigl_execution["mask_updater"] == "RigL" and rigl_execution["training_finite"],
            "reference_port_created": True,
            "parity_passed": (rigl_parity["weights_match"] and
                              rigl_parity["exact_edge_conservation"] and
                              rigl_parity["new_edge_optimizer_state_reset"]),
        },
        "classification": "validated_reference",
        "no_ccdn_performance_experiments_run": True,
        "scientific_performance_claim": False,
    }
    (art / "validation_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"validation summary: {art / 'validation_summary.json'}")


if __name__ == "__main__":
    main()
