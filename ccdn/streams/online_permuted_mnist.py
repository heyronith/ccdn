"""Deterministic single-example task sequence for online Permuted MNIST.

Task metadata stays on the evaluator side. A learner is called only with the
current ``(x, y)`` pair; it is never passed a task index or boundary callback.
Permutation and order RNGs are recreated from ``seed`` and ``task_index`` so a
task can be replayed exactly without storing its permuted image tensor.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib

import torch


@dataclass
class OnlineTask:
    task_index: int
    permutation: torch.Tensor
    order: torch.Tensor


class OnlinePermutedMNIST:
    def __init__(self, seed=101, root="./data", images=None, labels=None,
                 download=True):
        self.seed = int(seed)
        self.data_source = "mnist" if images is None and labels is None else "injected_test_data"
        if images is None or labels is None:
            from torchvision.datasets import MNIST
            from torchvision.transforms import ToTensor

            dataset = MNIST(root, train=True, download=download,
                            transform=ToTensor())
            images = dataset.data
            labels = dataset.targets
        x = torch.as_tensor(images)
        if x.ndim == 3:
            x = x.unsqueeze(1)
        if x.ndim != 4:
            raise ValueError("images must be N×H×W or N×C×H×W")
        if x.dtype == torch.uint8:
            x = x.float().div_(255.0)
        else:
            x = x.float()
        self.images = x.reshape(x.shape[0], -1).contiguous()
        self.labels = torch.as_tensor(labels, dtype=torch.long).reshape(-1).contiguous()
        if self.images.shape[0] != self.labels.shape[0]:
            raise ValueError("image and label counts do not match")
        self.input_size = int(self.images.shape[1])

    def task(self, task_index: int) -> OnlineTask:
        task_index = int(task_index)
        # Distinct reproducible streams for pixel permutation and image order.
        perm_gen = torch.Generator(device="cpu").manual_seed(
            self.seed + 1_000_003 * (task_index + 1))
        order_gen = torch.Generator(device="cpu").manual_seed(
            self.seed + 2_000_033 * (task_index + 1))
        return OnlineTask(task_index,
                          torch.randperm(self.input_size, generator=perm_gen),
                          torch.randperm(len(self.labels), generator=order_gen))

    def sample(self, task: OnlineTask, position: int, device="cpu"):
        dataset_index = int(task.order[position])
        x = self.images[dataset_index, task.permutation].to(device).unsqueeze(0)
        y = self.labels[dataset_index].to(device).reshape(1)
        return x, y

    def diagnostic_batch(self, task: OnlineTask, count=2000, device="cpu"):
        count = min(int(count), len(task.order))
        indices = task.order[:count]
        x = self.images[indices][:, task.permutation].to(device)
        y = self.labels[indices].to(device)
        return x, y

    def sequence_digest(self, tasks: int, examples_per_task: int) -> str:
        """Hash task permutations/orders without retaining duplicated tensors."""
        digest = hashlib.sha256()
        for task_index in range(int(tasks)):
            current = self.task(task_index)
            digest.update(task_index.to_bytes(8, "little", signed=False))
            digest.update(current.permutation.numpy().tobytes())
            digest.update(current.order[:int(examples_per_task)].numpy().tobytes())
        return digest.hexdigest()

    @property
    def metadata(self):
        return {
            "stream": "online_permuted_mnist",
            "data_source": self.data_source,
            "task_permutation_seed": "seed + 1000003 * (task_index + 1)",
            "task_order_seed": "seed + 2000033 * (task_index + 1)",
            "task_generation": "torch.randperm on CPU",
            "images_stored_once": True,
        }
