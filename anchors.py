from typing import Dict, List, Optional
import copy
import torch
from model_utils import average_state_dicts, clone_state_dict


class AnchorManager:
    """Handles FedGuCci anchor selection and update variants."""

    def __init__(self, method: str, config):
        self.method = method
        self.config = config
        self.anchor_states: List[Dict[str, torch.Tensor]] = []
        self.ema_anchor: Optional[Dict[str, torch.Tensor]] = None
        self.initialized = False

    def needs_best_client_initialization(self):
        return self.method in {"fedgucci_best_anchor", "fedgucci_topq_anchor", "fedgucci_topq_ema", "fedgucci_hsa"}

    def uses_ema(self):
        return self.method in {"fedgucci_ema_anchor", "fedgucci_topq_ema", "fedgucci_hsa"}

    def get_anchors(self):
        if not self.method.startswith("fedgucci"):
            return []
        if self.uses_ema():
            return [] if self.ema_anchor is None else [copy.deepcopy(self.ema_anchor)]
        return copy.deepcopy(self.anchor_states)

    def initialize_with_global(self, global_state):
        state = clone_state_dict(global_state)
        self.anchor_states = [copy.deepcopy(state)]
        self.ema_anchor = copy.deepcopy(state)
        self.initialized = True

    def initialize_with_clients(self, client_states, client_scores):
        if not client_states:
            return
        if self.method in {"fedgucci_best_anchor"}:
            best = max(range(len(client_states)), key=lambda i: client_scores[i])
            anchor = clone_state_dict(client_states[best])
        else:
            q = min(self.config.top_q, len(client_states))
            top_ids = sorted(range(len(client_states)), key=lambda i: client_scores[i], reverse=True)[:q]
            anchor = average_state_dicts([client_states[i] for i in top_ids])

        self.anchor_states = [copy.deepcopy(anchor)]
        self.ema_anchor = copy.deepcopy(anchor)
        self.initialized = True

    def update_after_round(self, global_state):
        if not self.method.startswith("fedgucci"):
            return
        if not self.initialized:
            self.initialize_with_global(global_state)
            return

        global_cpu = clone_state_dict(global_state)
        if self.uses_ema():
            tau = self.config.ema_tau
            for k in self.ema_anchor.keys():
                if torch.is_floating_point(self.ema_anchor[k]):
                    self.ema_anchor[k] = tau * self.ema_anchor[k].float() + (1.0 - tau) * global_cpu[k].float()
                else:
                    self.ema_anchor[k] = global_cpu[k].clone()
        else:
            self.anchor_states.append(global_cpu)
            self.anchor_states = self.anchor_states[-self.config.num_anchors:]
