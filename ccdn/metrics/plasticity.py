import torch

def diagnostics(model):
    acts=[a.detach() for a in getattr(model,"last_activations",[]) if a is not None]
    if not acts: return {"dormant_relu_fraction":0.0,"activation_mean":0.0,"activation_std":0.0,"weight_l2_norm":float(torch.sqrt(sum((p.detach()**2).sum() for p in model.parameters())).item())}
    total=sum(a.numel() for a in acts); dead=sum((a<=0).sum().item() for a in acts)
    flat=torch.cat([a.flatten() for a in acts])
    return {"dormant_relu_fraction":dead/total,"activation_mean":float(flat.mean()),"activation_std":float(flat.std(unbiased=False)),"weight_l2_norm":float(torch.sqrt(sum((p.detach()**2).sum() for p in model.parameters())).item())}
