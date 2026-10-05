"""Check runner SGD semantics against the pinned official Backprop class."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / ".external/official_baselines/loss-of-plasticity"))
sys.path.insert(0, str(ROOT))

from lop.algos.bp import Backprop  # noqa: E402
from lop.nets.deep_ffnn import DeepFFNN  # noqa: E402
from ccdn.baselines import StaticDense  # noqa: E402
from ccdn.experiments.online_permuted_mnist import _learner_step  # noqa: E402
from ccdn.models.dense_mlp import DenseMLP  # noqa: E402


def main():
    torch.set_num_threads(1)
    torch.manual_seed(551)
    local = DenseMLP(input_size=12, hidden_sizes=(7, 7, 7), output_size=10,
                     initialization="published_kaiming")
    official_net = DeepFFNN(input_size=12, num_features=7, num_outputs=10,
                            num_hidden_layers=3, act_type="relu")
    upstream_layers = ([official_net.in_layer.fc]
                       + [layer.fc for layer in official_net.hidden_layers]
                       + [official_net.out_layer.fc])
    with torch.no_grad():
        for local_layer, upstream_layer in zip(local.layers, upstream_layers):
            upstream_layer.weight.copy_(local_layer.weight)
            upstream_layer.bias.copy_(local_layer.bias)

    lr = 0.003
    local_optimizer = torch.optim.SGD(local.parameters(), lr=lr, momentum=0,
                                      weight_decay=0)
    learner = Backprop(official_net, step_size=lr, opt="sgd", loss="nll",
                       weight_decay=0, momentum=0, device="cpu")
    policy = StaticDense(local)

    data_gen = torch.Generator().manual_seed(82)
    inputs = torch.randn(9, 12, generator=data_gen)
    labels = torch.randint(0, 10, (9,), generator=data_gen)
    predictions_match = losses_match = weights_match = biases_match = True
    max_loss_error = max_weight_error = max_bias_error = 0.0
    for i in range(len(labels)):
        x, y = inputs[i:i + 1], labels[i:i + 1]
        official_loss, official_logits = learner.learn(x, y)
        local_prediction, local_loss = _learner_step(
            local, policy, local_optimizer, x, y)
        predictions_match &= int(official_logits.argmax(1).item()) == local_prediction
        loss_error = abs(float(official_loss) - local_loss)
        max_loss_error = max(max_loss_error, loss_error)
        losses_match &= loss_error <= 1e-7
        for ours, upstream in zip(local.layers, upstream_layers):
            weight_error = float((ours.weight - upstream.weight).abs().max())
            bias_error = float((ours.bias - upstream.bias).abs().max())
            max_weight_error = max(max_weight_error, weight_error)
            max_bias_error = max(max_bias_error, bias_error)
            weights_match &= torch.allclose(ours.weight, upstream.weight,
                                            rtol=1e-6, atol=1e-7)
            biases_match &= torch.allclose(ours.bias, upstream.bias,
                                           rtol=1e-6, atol=1e-7)

    result = {
        "official_repository": "shibhansh/loss-of-plasticity",
        "official_commit": "a6b79580d85f3025bdb601566d3627c5f489f13b",
        "official_classes": ["lop.algos.bp.Backprop", "lop.nets.deep_ffnn.DeepFFNN"],
        "updates_compared": len(labels),
        "learning_rate": lr,
        "batch_size": 1,
        "same_initial_weights": True,
        "same_inputs_labels_order": True,
        "predictions_before_update_match": bool(predictions_match),
        "losses_match": bool(losses_match),
        "weight_updates_match": bool(weights_match),
        "bias_updates_match": bool(biases_match),
        "max_loss_abs_error": max_loss_error,
        "max_weight_abs_error": max_weight_error,
        "max_bias_abs_error": max_bias_error,
        "classification": "runner_semantics_parity; not a benchmark result",
    }
    out = ROOT / "review_artifacts/cycle_2b/backprop_runner_parity.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if not all((predictions_match, losses_match, weights_match, biases_match)):
        raise SystemExit("online runner semantics differ from official Backprop")


if __name__ == "__main__":
    main()
