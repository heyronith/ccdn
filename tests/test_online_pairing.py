import torch

from ccdn.official_reference import ContinualBackpropReference
from ccdn.models.dense_mlp import DenseMLP
from ccdn.streams.online_permuted_mnist import OnlinePermutedMNIST
from ccdn.utils.reproducibility import seed_everything


def test_backprop_and_reference_cbp_can_share_exact_initial_state_and_stream():
    seed_everything(101)
    backprop = DenseMLP(hidden_sizes=(8, 8, 8), initialization="published_kaiming")
    seed_everything(101)
    cbp = DenseMLP(hidden_sizes=(8, 8, 8), initialization="published_kaiming")
    ContinualBackpropReference(cbp, replacement_rate=1e-5,
                               decay_rate=0.99, maturity_threshold=100,
                               accumulate=True, util_type="adaptable_contribution")
    for left, right in zip(backprop.parameters(), cbp.parameters()):
        assert torch.equal(left, right)

    images = torch.arange(60 * 28 * 28, dtype=torch.int32).reshape(60, 28, 28).to(torch.uint8)
    labels = torch.arange(60) % 10
    stream_bp = OnlinePermutedMNIST(seed=101, images=images, labels=labels)
    stream_cbp = OnlinePermutedMNIST(seed=101, images=images, labels=labels)
    assert stream_bp.sequence_digest(3, 60) == stream_cbp.sequence_digest(3, 60)
