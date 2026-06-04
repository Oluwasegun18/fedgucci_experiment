import copy
import random
from typing import Dict, List, Optional
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

try:
    from torch.func import functional_call
except Exception:
    from torch.nn.utils.stateless import functional_call

from model_utils import clone_state_dict, state_to_device


def _functional_logits(model, params, buffers, x):
    merged = {**params, **buffers}
    return functional_call(model, merged, (x,))


def connectivity_loss(model, anchor_states: List[Dict[str, torch.Tensor]], x, y, device, samples: int = 1):
    """Differentiable FedGuCci connectivity loss.

    Computes E_alpha L_i(alpha*w_i + (1-alpha)*w_anchor) using stateless functional_call,
    so gradients flow to the current model parameters.
    """
    if not anchor_states:
        return torch.tensor(0.0, device=device)

    current_params = dict(model.named_parameters())
    current_buffers = dict(model.named_buffers())
    total = torch.tensor(0.0, device=device)

    for _ in range(samples):
        anchor = random.choice(anchor_states)
        alpha = random.random()
        interp_params = {}
        for name, p in current_params.items():
            a = anchor[name].to(device)
            interp_params[name] = alpha * p + (1.0 - alpha) * a
        # Keep current buffers. Models in this project avoid BatchNorm for stable stateless calls.
        logits = _functional_logits(model, interp_params, current_buffers, x)
        total = total + F.cross_entropy(logits, y)
    return total / max(samples, 1)


def _grad_norm(model):
    norms = []
    for p in model.parameters():
        if p.grad is not None:
            norms.append(torch.norm(p.grad.detach(), p=2))
    if not norms:
        return torch.tensor(0.0, device=next(model.parameters()).device)
    return torch.norm(torch.stack(norms), p=2)


def _apply_sam_perturbation(model, rho: float):
    norm = _grad_norm(model)
    scale = rho / (norm + 1e-12)
    e_ws = []
    with torch.no_grad():
        for p in model.parameters():
            if p.grad is None:
                e_ws.append(None)
                continue
            e_w = p.grad * scale.to(p)
            p.add_(e_w)
            e_ws.append(e_w)
    return e_ws


def _remove_sam_perturbation(model, e_ws):
    with torch.no_grad():
        for p, e_w in zip(model.parameters(), e_ws):
            if e_w is not None:
                p.sub_(e_w)


def compute_local_objective(model, x, y, method, config, global_params=None, anchor_states=None, beta_i=None):
    logits = model(x)
    loss = F.cross_entropy(logits, y)

    if method == "fedprox":
        prox = torch.tensor(0.0, device=x.device)
        for name, p in model.named_parameters():
            prox = prox + torch.sum((p - global_params[name]) ** 2)
        loss = loss + 0.5 * config.fedprox_mu * prox

    if method.startswith("fedgucci"):
        beta = config.beta if beta_i is None else beta_i
        conn = connectivity_loss(
            model=model,
            anchor_states=anchor_states or [],
            x=x,
            y=y,
            device=x.device,
            samples=config.alpha_interp_samples,
        )
        loss = loss + beta * conn
    return loss


def train_one_client(
    model,
    global_state: Dict[str, torch.Tensor],
    train_dataset,
    indices: List[int],
    method: str,
    config,
    device,
    anchor_states: Optional[List[Dict[str, torch.Tensor]]] = None,
    beta_i: Optional[float] = None,
):
    model.load_state_dict(global_state, strict=True)
    model.to(device)
    model.train()

    loader = DataLoader(
        Subset(train_dataset, indices),
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )

    optimizer = torch.optim.SGD(
        model.parameters(), lr=config.lr, momentum=config.momentum, weight_decay=config.weight_decay
    )
    global_params = {name: p.detach().clone() for name, p in model.named_parameters()}
    anchors_on_device = [state_to_device(a, device) for a in (anchor_states or [])]

    for _ in range(config.local_epochs):
        for x, y in loader:
            x, y = x.to(device), y.to(device)

            if method == "fedsam":
                optimizer.zero_grad(set_to_none=True)
                loss = compute_local_objective(model, x, y, method="fedavg", config=config)
                loss.backward()
                e_ws = _apply_sam_perturbation(model, config.sam_rho)

                optimizer.zero_grad(set_to_none=True)
                loss_perturbed = compute_local_objective(model, x, y, method="fedavg", config=config)
                loss_perturbed.backward()
                _remove_sam_perturbation(model, e_ws)
                optimizer.step()
                continue

            optimizer.zero_grad(set_to_none=True)
            loss = compute_local_objective(
                model=model,
                x=x,
                y=y,
                method=method,
                config=config,
                global_params=global_params,
                anchor_states=anchors_on_device,
                beta_i=beta_i,
            )
            loss.backward()
            optimizer.step()

    return clone_state_dict(model.state_dict())
