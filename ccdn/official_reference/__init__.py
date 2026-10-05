"""Local PyTorch ports used only to validate upstream baseline semantics."""

from .continual_backprop import ContinualBackpropReference
from .rigl import RigLReference

__all__ = ["ContinualBackpropReference", "RigLReference"]
