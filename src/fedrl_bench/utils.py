import torch
from torch import nn

def flatten_params(model: nn.Module) -> torch.Tensor:
    flattened = torch.cat([p.flatten() for p in model.parameters()]).detach()
    return flattened

def flatten_grads(model: nn.Module) -> torch.Tensor:
    params = list(model.parameters())
    grads = [p.grad if p.grad is not None else torch.zeros_like(p) for p in params]
    flattened = torch.cat([g.flatten() for g in grads])
    return flattened

def unflatten_into(model: nn.Module, vec: torch.Tensor) -> None:
    r"""Assumes single device"""
    size = sum(p.numel() for p in model.parameters())
    if size != vec.numel():
        raise ValueError(size, vec.numel(), "Tensor does not match the size of model's parameters")
    with torch.no_grad():
        for p in model.parameters():
            vec_numel = vec.numel()
            param_size = p.size()
            param_numel = param_size.numel()
            v = vec.split([param_numel, vec_numel-param_numel])
            new_params = v[0].unflatten(0, param_size)
            
            p.copy_(new_params)
            vec = v[1]
        
        
