import torch

from ccdn.streams.online_permuted_mnist import OnlinePermutedMNIST
from scripts.bootstrap_cycle2d import task_sequence_digests


def synthetic_stream(seed=101):
    generator = torch.Generator().manual_seed(11)
    images = torch.randint(0, 256, (32, 28, 28), generator=generator, dtype=torch.uint8)
    labels = torch.arange(32) % 10
    return OnlinePermutedMNIST(seed=seed, images=images, labels=labels)


def test_incremental_prefix_hash_matches_stream_digest_and_is_deterministic():
    stream = synthetic_stream()
    left = task_sequence_digests(stream, horizons=[1, 3], examples_per_task=32)
    right = task_sequence_digests(stream, horizons=[1, 3], examples_per_task=32)
    assert left == right
    assert left[1] == stream.sequence_digest(1, 32)
    assert left[3] == stream.sequence_digest(3, 32)


def test_task_150_is_deterministic_and_does_not_advance_global_rng():
    stream = synthetic_stream()
    torch.manual_seed(808)
    before = torch.get_rng_state().clone()
    a = stream.task(150)
    after = torch.get_rng_state().clone()
    b = stream.task(150)
    assert a.task_index == b.task_index == 150
    assert torch.equal(a.permutation, b.permutation)
    assert torch.equal(a.order, b.order)
    assert torch.equal(before, after)
