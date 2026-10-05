import torch

def _bytes(obj):
    if torch.is_tensor(obj): return obj.numel()*obj.element_size()
    if isinstance(obj,dict): return sum(_bytes(k)+_bytes(v) for k,v in obj.items())
    if isinstance(obj,(list,tuple)): return sum(_bytes(v) for v in obj)
    return 0

def account(model,algorithm,optimizer):
    params=list(model.parameters()); total=sum(p.numel() for p in params)
    sparse_layers=[l for l in getattr(model,"layers",[]) if hasattr(l,"mask")]
    active=(sum(int(l.mask.sum()) for l in sparse_layers)+sum(l.bias.numel() for l in sparse_layers if l.bias is not None)) if sparse_layers else total
    parameter_bytes=sum(p.numel()*p.element_size() for p in params)
    buffer_bytes=sum(b.numel()*b.element_size() for b in model.buffers())
    mask_bytes=sum(l.mask.numel()*l.mask.element_size() for l in sparse_layers)
    model_bytes=parameter_bytes+buffer_bytes
    aux=_bytes(algorithm.state_dict())
    opt=_bytes(optimizer.state)
    dense_macs=sum(a.in_features*a.out_features for a in getattr(model,"layers",[]))
    return {"total_possible_parameters":total,"actual_tensor_parameters":total,
            "logical_active_parameters":active,"active_parameters":active,
            "trainable_active_parameters":active,"parameter_tensor_memory_bytes":parameter_bytes,
            "mask_memory_bytes":mask_bytes,"model_tensor_memory_bytes":model_bytes,
            "auxiliary_algorithm_state_memory_bytes":aux,"optimizer_state_memory_bytes":opt,
            "estimated_total_training_state_memory_bytes":model_bytes+aux+opt,
            "estimated_dense_macs_per_example":dense_macs,
            "dense_execution":bool(sparse_layers),"sparse_compute_claim":False}
