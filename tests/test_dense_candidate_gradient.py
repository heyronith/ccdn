import torch
import torch.nn.functional as F

from ccdn.models.sparse_mlp import SparseMLP, dense_candidate_gradient_scores


def test_candidate_gradient_reconstruction_matches_full_effective_weight_gradient():
    torch.manual_seed(41)
    model = SparseMLP(input_size=7, hidden_sizes=(5, 4), output_size=3,
                      density=0.47, seed=13).double()
    x = torch.randn(6, 7, dtype=torch.float64)
    labels = torch.tensor([0, 2, 1, 0, 1, 2])

    logits = model(x)
    F.cross_entropy(logits, labels).backward()

    reconstructed = [dense_candidate_gradient_scores(model, i).clone()
                     for i in range(len(model.layers))]

    # Rebuild the same forward pass while making every effective dense weight
    # independently differentiable, including positions that are inactive in
    # the sparse model. This is the gradient RigL needs to rank candidates.
    dense_weights = [(layer.weight.detach() * layer.mask).clone().requires_grad_(True)
                     for layer in model.layers]
    dense_biases = [layer.bias.detach().clone() for layer in model.layers]
    activations = x
    for i, (weight, bias) in enumerate(zip(dense_weights, dense_biases)):
        activations = F.linear(activations, weight, bias)
        if i < len(dense_weights) - 1:
            activations = F.relu(activations)
    F.cross_entropy(activations, labels).backward()

    for observed, weight in zip(reconstructed, dense_weights):
        torch.testing.assert_close(observed, weight.grad.abs(), rtol=1e-12, atol=1e-12)

    # Ensure the fixture exercises inactive positions in every tested layer.
    assert len(reconstructed) == 3
    assert all((~layer.mask).any() for layer in model.layers)
