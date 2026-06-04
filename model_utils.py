import copy
from typing import Dict, List, Optional
import torch


StateDict = Dict[str, torch.Tensor]


def clone_state_dict(state: StateDict) -> StateDict:
    return {k: v.detach().cpu().clone() for k, v in state.items()}


def average_state_dicts(states: List[StateDict], weights: Optional[List[float]] = None) -> StateDict:
    if not states:
        raise ValueError("Cannot average an empty state list.")
    if weights is None:
        weights = [1.0 / len(states)] * len(states)
    total = sum(weights)
    weights = [w / total for w in weights]

    avg = {}
    for k in states[0].keys():
        if torch.is_floating_point(states[0][k]):
            avg[k] = sum(weights[i] * states[i][k].float() for i in range(len(states))).clone()
        else:
            avg[k] = states[0][k].clone()
    return avg


def weighted_client_average(client_states: List[StateDict], client_sizes: List[int]) -> StateDict:
    weights = [s / max(sum(client_sizes), 1) for s in client_sizes]
    return average_state_dicts(client_states, weights)


def state_to_device(state: StateDict, device: torch.device) -> StateDict:
    return {k: v.to(device) for k, v in state.items()}
