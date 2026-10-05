import math

import torch

from ccdn.models.sparse_mlp import SparseMLP
from ccdn.official_reference.rigl import RigLReference


def test_begin_end_frequency_and_cosine_schedule():
    model = SparseMLP(8, (4,), 2, density=.5, seed=1)
    optimizer = torch.optim.SGD(model.parameters(), lr=.01)
    port = RigLReference(model, begin_step=2, end_step=6, update_freq=2,
                         init_drop_fraction=.2, schedule="cosine")
    counts = []
    for _ in range(8):
        port.after_optimizer_step(optimizer)
        counts.append(port.rewire_event_count)
    assert counts == [0, 0, 1, 1, 1, 1, 1, 1]
    assert port.last_update_step == 6
    assert port.drop_fraction(2) == pytest_approx(.1)
    assert math.isclose(port.drop_fraction(6), 0.0, abs_tol=1e-8)


def pytest_approx(value):
    # Keep the test dependency surface to pytest itself while being explicit.
    import pytest
    return pytest.approx(value)
