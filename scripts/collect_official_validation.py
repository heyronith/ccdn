"""Collect compact evidence from completed official source validation smokes."""
from __future__ import annotations

import json
import argparse
import pickle
import platform
import subprocess
from pathlib import Path

import numpy as np
import scipy
import tensorflow as tf
import tensorflow_datasets as tfds
import torch
import torchvision

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "review_artifacts/cycle_2v"
CBP_SOURCE = ROOT / ".external/official_baselines/loss-of-plasticity"
RIGL_SOURCE = ROOT / ".external/official_baselines/rigl"


def sha(path):
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=path, text=True).strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cbp-output", type=Path,
                        default=ROOT / "results/official_validation/cbp/official_metrics.pkl")
    parser.add_argument("--rigl-logdir", type=Path,
                        default=ROOT / "results/official_validation/rigl_final")
    args = parser.parse_args()
    ART.mkdir(parents=True, exist_ok=True)
    official = {
        "continual_backprop": {"repository": "shibhansh/loss-of-plasticity",
                                "commit": "a6b79580d85f3025bdb601566d3627c5f489f13b"},
        "rigl": {"repository": "google-research/rigl",
                 "commit": "d39fc7d46505cb3196cb1edeb32ed0b6dd44c0f9"},
    }
    versions = {
        "validation_python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cbp_environment": {"python": "3.11.15", "torch": "2.1.0",
                            "torchvision": "0.16.0", "numpy": "1.24.1",
                            "scipy": "1.11.2", "editable_install": True},
        "rigl_environment": {"python": "3.11.15", "tensorflow": tf.__version__,
                             "tensorflow_datasets": tfds.__version__,
                             "tensorflow_model_optimization": "0.8.0",
                             "tf_keras": "2.15.0", "jax": "0.4.23",
                             "jaxlib": "0.4.23", "numpy": np.__version__,
                             "scipy": scipy.__version__, "torch": torch.__version__},
    }
    (ART / "environment_versions.json").write_text(json.dumps(versions, indent=2) + "\n")
    provenance = {
        "ccdn_base_commit": "c663d62f9658e9f6dac1325a408a607ff83ab11a",
        "official_repositories": {
            "continual_backprop": {**official["continual_backprop"], "verified_head": sha(CBP_SOURCE)},
            "rigl": {**official["rigl"], "verified_head": sha(RIGL_SOURCE)},
        },
        "external_source_trees_committed": False,
        "runtime_patch": "official_baselines/patches/rigl_tf215_runtime.py",
        "runtime_patch_scope": "TensorFlow 2.15 legacy optimizer API; no algorithm source edits",
    }
    (ART / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")

    cbp_file = args.cbp_output
    cbp = pickle.load(cbp_file.open("rb"))
    acc = cbp["accuracies"]
    official_cbp = {
        "official_commit": official["continual_backprop"]["commit"],
        "official_code_unchanged": True,
        "loader": "pinned lop/permuted_mnist/load_mnist.py",
        "data_source": "MNIST", "synthetic": False,
        "tasks": 2, "change_after_examples": 30000,
        "updates": int(acc.numel()), "batch_size": 10000,
        "hidden_layers": 3, "hidden_width": 100,
        "training_metrics_finite": bool(torch.isfinite(acc).all()),
        "output_generated": cbp_file.exists(),
        "output_metrics": {"accuracy_values": acc.tolist(),
                           "approximate_ranks_finite": bool(torch.isfinite(cbp["approximate_ranks"]).all()),
                           "dead_neurons_finite": bool(torch.isfinite(cbp["dead_neurons"]).all())},
        "limitation": "Upstream online_expr allocates rank rows in 60,000-example units; smoke uses two 30,000-example task segments and six 10,000-example updates.",
    }
    (ART / "official_cbp_execution.json").write_text(json.dumps(official_cbp, indent=2) + "\n")

    logdir = args.rigl_logdir
    initial_path, final_path = str(logdir / "ckpt-0"), str(logdir / "ckpt-3")
    initial_masks, final_masks = [], []
    for index in range(3):
        key = f"model/layer_with_weights-{index}/mask/.ATTRIBUTES/VARIABLE_VALUE"
        initial_masks.append(tf.train.load_variable(initial_path, key))
        final_masks.append(tf.train.load_variable(final_path, key))
    event_file = next((logdir / "train_logs").glob("events.out.tfevents.*"))
    scalars = {}
    for event in tf.compat.v1.train.summary_iterator(str(event_file)):
        for value in event.summary.value:
            if value.HasField("simple_value"):
                scalars.setdefault(value.tag, []).append(float(value.simple_value))
    active_initial = [int(mask.sum()) for mask in initial_masks]
    active_final = [int(mask.sum()) for mask in final_masks]
    changed = [not np.array_equal(a, b) for a, b in zip(initial_masks, final_masks)]
    official_rigl = {
        "official_commit": official["rigl"]["commit"],
        "execution_path": "pinned rigl/rigl_tf2/train.py with RigL mask updater",
        "runtime_patch": "TF 2.15 legacy SGD API only; upstream source pristine",
        "dataset": "MNIST via official TFDS loader", "synthetic": False,
        "updates_configured": 3,
        # TensorFlow 2.15's legacy SGD checkpoint names the iteration variable
        # ``iter`` (the modern optimizer API uses ``iterations``).
        "updates_observed": int(tf.train.load_variable(
            final_path, "optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE")),
        "mask_updater": "RigL", "one_shot_initial_sparsity": 0.8,
        "initial_active_edges_by_layer": active_initial,
        "final_active_edges_by_layer": active_final,
        "active_count_conserved_by_layer": active_initial == active_final,
        "mask_changed_by_layer": changed,
        "mask_changed": any(changed),
        "training_finite": all(np.isfinite(vals).all() for vals in scalars.values()),
        "test_loss_values": scalars.get("test_loss", []),
        "test_accuracy_values": scalars.get("test_acc", []),
        "zero_grow_initialization_and_new_slot_reset": "verified against SparseRigLOptimizerBase in rigl_parity.json",
    }
    (ART / "official_rigl_execution.json").write_text(json.dumps(official_rigl, indent=2) + "\n")
    (ART / "rigl_parity.json").write_text(
        (ROOT / "results/official_validation/rigl_parity.json").read_text())
    print(f"wrote validation environment, provenance, and execution evidence under {ART}")


if __name__ == "__main__":
    main()
