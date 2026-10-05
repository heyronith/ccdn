"""Initialization helpers matching ``torch.nn.Linear.reset_parameters``."""
import math
import torch


def linear_init_bound(in_features: int) -> float:
    """Return Linear's default uniform bound for a layer's full fan-in."""
    return 1.0 / math.sqrt(in_features)


def initialize_linear_uniform_(tensor: torch.Tensor, in_features: int) -> torch.Tensor:
    """Initialize values with the default ``nn.Linear`` uniform distribution.

    ``in_features`` is supplied by the full layer, so initializing a row or
    column slice cannot accidentally change the inferred fan-in.
    """
    bound = linear_init_bound(in_features)
    with torch.no_grad():
        return tensor.uniform_(-bound, bound)
