import torch

def diagnostics(model):
    acts=[a.detach() for a in getattr(model,"last_activations",[]) if a is not None]
    if not acts: return {"dormant_relu_fraction":0.0,"activation_mean":0.0,"activation_std":0.0,"weight_l2_norm":float(torch.sqrt(sum((p.detach()**2).sum() for p in model.parameters())).item())}
    total=sum(a.numel() for a in acts); dead=sum((a<=0).sum().item() for a in acts)
    flat=torch.cat([a.flatten() for a in acts])
    return {"dormant_relu_fraction":dead/total,"activation_mean":float(flat.mean()),"activation_std":float(flat.std(unbiased=False)),"weight_l2_norm":float(torch.sqrt(sum((p.detach()**2).sum() for p in model.parameters())).item())}


@torch.inference_mode()
def online_plasticity_diagnostics(model, images, batch_size=256):
    """Dead-unit, mean-|weight|, and representation effective-rank checks.

    ``images`` is a deterministic diagnostic set of task-permuted examples.
    No optimizer or learner state is touched. Dead means a ReLU unit emitted
    exactly zero on every diagnostic example.
    """
    was_training = model.training
    model.eval()
    layer_acts = [[] for _ in model.hidden_sizes]
    for start in range(0, len(images), batch_size):
        model(images[start:start + batch_size])
        for i, activations in enumerate(model.last_activations):
            layer_acts[i].append(activations.detach())
    model.train(was_training)

    result = {}
    total_dead = 0
    total_units = 0
    for i, chunks in enumerate(layer_acts):
        acts = torch.cat(chunks, dim=0)
        dead = acts.eq(0).all(dim=0)
        count = int(dead.sum())
        total_dead += count
        total_units += dead.numel()
        result[f"dead_unit_fraction_layer_{i}"] = count / max(dead.numel(), 1)
        singular_values = torch.linalg.svdvals(acts.to(torch.float64))
        mass = singular_values.sum()
        if float(mass) == 0.0:
            rank = 0.0
        else:
            probabilities = singular_values / mass
            probabilities = probabilities[probabilities > 0]
            rank = float(torch.exp(-(probabilities * probabilities.log()).sum()))
        result[f"effective_rank_layer_{i}"] = rank
    result["dead_unit_fraction_overall"] = total_dead / max(total_units, 1)
    weights = [layer.weight.detach().abs().mean() for layer in model.layers]
    names = [f"mean_absolute_weight_layer_{i}" for i in range(len(model.hidden_sizes))]
    names.append("mean_absolute_weight_output")
    for name, value in zip(names, weights):
        result[name] = float(value)
    total_weight_count = sum(layer.weight.numel() for layer in model.layers)
    result["mean_absolute_weight"] = float(
        sum(layer.weight.detach().abs().sum() for layer in model.layers)
        / max(total_weight_count, 1))
    return result
