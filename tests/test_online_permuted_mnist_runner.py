import inspect
import numpy as np
import torch

from ccdn.experiments.online_permuted_mnist import run_online_experiment
from ccdn.streams.online_permuted_mnist import OnlinePermutedMNIST


def _small_stream():
    gen = torch.Generator().manual_seed(91)
    images = torch.rand(36, 2, 4, generator=gen)
    labels = torch.arange(36) % 3
    return OnlinePermutedMNIST(seed=101, images=images, labels=labels)


def _config():
    return {
        "experiment": {"name": "resume_test", "seed": 101,
                       "device": "cpu", "deterministic": True, "cpu_threads": 1},
        "stream": {"tasks": 3, "examples_per_task": 12},
        "model": {"type": "continual_backprop_reference",
                  "hidden_sizes": [4, 4, 4], "output_size": 3,
                  "initialization": "published_kaiming"},
        "optimizer": {"type": "sgd", "learning_rate": 0.003,
                      "momentum": 0.0, "weight_decay": 0.0},
        "algorithm": {"replacement_rate": 0.05, "decay_rate": 0.99,
                      "maturity_threshold": 1, "accumulate": True,
                      "util_type": "adaptable_contribution"},
        "diagnostics": {"examples": 8},
        "output": {"root": "results"},
    }


def test_task_sequence_is_reconstructible_and_has_no_learner_metadata():
    first, second = _small_stream(), _small_stream()
    for task_index in range(3):
        task_a, task_b = first.task(task_index), second.task(task_index)
        assert torch.equal(task_a.permutation, task_b.permutation)
        assert torch.equal(task_a.order, task_b.order)
        xa, ya = first.sample(task_a, 0)
        xb, yb = second.sample(task_b, 0)
        assert torch.equal(xa, xb)
        assert torch.equal(ya, yb)
    parameters = inspect.signature(
        __import__("ccdn.experiments.online_permuted_mnist",
                   fromlist=["_learner_step"])._learner_step).parameters
    assert "task_index" not in parameters
    assert "task_boundary" not in parameters


def test_task_boundary_resume_matches_continuous_run(tmp_path):
    config = _config()
    continuous_dir = tmp_path / "continuous"
    interrupted_dir = tmp_path / "interrupted"
    run_online_experiment(config, stream=_small_stream(), output_dir=continuous_dir,
                          max_tasks_this_invocation=3)
    run_online_experiment(config, stream=_small_stream(), output_dir=interrupted_dir,
                          max_tasks_this_invocation=1)
    checkpoint = interrupted_dir / "checkpoint_task_001.pt"
    run_online_experiment(config, stream=_small_stream(), resume_from=checkpoint)

    continuous = torch.load(continuous_dir / "checkpoint_task_003.pt",
                            map_location="cpu", weights_only=False)
    resumed = torch.load(interrupted_dir / "checkpoint_task_003.pt",
                         map_location="cpu", weights_only=False)
    assert continuous["completed_task_index"] == resumed["completed_task_index"] == 2
    assert continuous["lifetime_examples_seen"] == resumed["lifetime_examples_seen"] == 36
    assert continuous["optimizer"] == resumed["optimizer"]
    assert continuous["rng_state"]["python"] == resumed["rng_state"]["python"]
    left_numpy, right_numpy = (continuous["rng_state"]["numpy"],
                               resumed["rng_state"]["numpy"])
    assert left_numpy[0] == right_numpy[0]
    assert np.array_equal(left_numpy[1], right_numpy[1])
    assert left_numpy[2:] == right_numpy[2:]
    assert torch.equal(continuous["rng_state"]["torch"], resumed["rng_state"]["torch"])
    for key, value in continuous["model"].items():
        assert torch.equal(value, resumed["model"][key])
    for key, value in continuous["algorithm"].items():
        if isinstance(value, list) and value and torch.is_tensor(value[0]):
            for a, b in zip(value, resumed["algorithm"][key]):
                assert torch.equal(a, b)
        else:
            assert value == resumed["algorithm"][key]
    with open(continuous_dir / "task_accuracy.csv") as a, \
            open(interrupted_dir / "task_accuracy.csv") as b:
        assert a.read() == b.read()
