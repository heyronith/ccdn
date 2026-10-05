"""Compare a fixed RigL edge update with the pinned TensorFlow implementation.

Run with the isolated RigL environment created in the Cycle 2V review package:
``.external/envs/rigl/bin/python scripts/validate_rigl_parity.py``.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import torch
import tensorflow as tf
from tensorflow.python.ops import control_flow_ops

# TensorFlow 2.15 removed this TF1 internal alias used by the pinned source.
if not hasattr(control_flow_ops, "Assert"):
    control_flow_ops.Assert = tf.debugging.Assert

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / ".external/official_baselines/rigl"))
from rigl.sparse_optimizers_base import SparseRigLOptimizerBase  # noqa: E402
sys.path.insert(0, str(ROOT))
from ccdn.models.sparse_mlp import SparseMLP  # noqa: E402
from ccdn.official_reference.rigl import RigLReference  # noqa: E402


def main():
    tf.compat.v1.disable_eager_execution()
    initial_mask = np.array([[1, 1, 1, 1, 0, 0]], dtype=np.float32)
    initial_weight = np.array([[0.9, 0.1, 0.6, 0.4, 0.0, 0.0]], dtype=np.float32)
    grow_score = np.array([[0.1, 0.5, 0.2, 0.3, 0.8, 0.7]], dtype=np.float32)
    tf1 = tf.compat.v1
    tf1.reset_default_graph()
    tf_mask = tf1.Variable(initial_mask)
    tf_weight = tf1.Variable(initial_weight)
    optimizer = tf1.train.MomentumOptimizer(0.1, momentum=0.9)
    optimizer._create_slots([tf_weight])
    official = SparseRigLOptimizerBase(
        optimizer, begin_step=0, end_step=0, frequency=1,
        drop_fraction=0.25, grow_init="zeros")
    official._use_stateless = False
    official.drop_fraction = tf.constant(0.25, dtype=tf.float32)
    official._global_step = tf1.Variable(0, trainable=False, dtype=tf.int64)
    official._weight2masked_grads = {tf_weight.name: tf.constant(grow_score)}
    momentum_slot = optimizer.get_slot(tf_weight, "momentum")
    update = official._get_update_op(
        tf.abs(tf_mask * tf_weight), tf.abs(tf.constant(grow_score)),
        tf_mask, tf_weight)
    with tf1.Session() as session:
        session.run(tf1.global_variables_initializer())
        session.run(tf1.assign(momentum_slot, tf.ones_like(momentum_slot)))
        session.run(update)
        official_mask, official_weight, official_momentum = session.run(
            [tf_mask, tf_weight, momentum_slot])

    model = SparseMLP(input_size=6, hidden_sizes=(), output_size=1,
                      density=2 / 3, seed=1)
    layer = model.layers[0]
    with torch.no_grad():
        layer.mask.copy_(torch.from_numpy(initial_mask.astype(bool)))
        layer.weight.copy_(torch.from_numpy(initial_weight))
    port = RigLReference(model, update_freq=1, init_drop_fraction=0.25)
    port.update_layer(0, torch.from_numpy(grow_score), 0.25,
                      torch.optim.SGD(model.parameters(), lr=0.1))
    np.testing.assert_array_equal(layer.mask.cpu().numpy(), official_mask.astype(bool))
    np.testing.assert_allclose(layer.weight.detach().cpu().numpy(), official_weight)
    newly_connected = (official_mask.astype(bool) & ~initial_mask.astype(bool))
    active_mask = initial_mask.astype(bool)
    lowest_active_index = int(np.argmin(np.where(active_mask, np.abs(initial_weight), np.inf)))
    highest_candidate_index = int(np.argmax(np.where(~active_mask, grow_score, -np.inf)))
    assert np.all(official_weight[newly_connected] == 0)
    assert np.all(official_momentum[newly_connected] == 0)
    result = {
        "source_commit": "d39fc7d46505cb3196cb1edeb32ed0b6dd44c0f9",
        "drop_fraction": 0.25,
        "official_mask": official_mask.astype(int).tolist(),
        "pytorch_mask": layer.mask.cpu().numpy().astype(int).tolist(),
        "weights_match": True,
        "exact_edge_conservation": int(official_mask.sum()) == int(initial_mask.sum()),
        "lowest_magnitude_edge_pruned": bool(not official_mask.flatten()[lowest_active_index]),
        "growth_used_gradient_ranking": bool(official_mask.flatten()[highest_candidate_index]),
        "new_edges_zero_initialized": bool(np.all(official_weight[newly_connected] == 0)),
        "new_edge_optimizer_state_reset": bool(np.all(official_momentum[newly_connected] == 0)),
    }
    out = ROOT / "results/official_validation/rigl_parity.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
