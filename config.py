from dataclasses import dataclass, asdict
from typing import List
import json
import os


@dataclass
class ExperimentConfig:
    # Experiment
    experiment_name: str = "fedgucci_hsa_cifar10"
    seed: int = 42
    output_dir: str = "./results"
    device: str = "cuda"

    # Data
    dataset: str = "cifar10"  # cifar10 or fashionmnist
    data_dir: str = "./data"
    num_classes: int = 10
    num_clients: int = 50
    dirichlet_alpha: float = 0.5
    val_fraction: float = 0.15
    min_client_size: int = 20

    # FL
    rounds: int = 100
    participation_rate: float = 1.0
    local_epochs: int = 3
    batch_size: int = 64
    lr: float = 0.01
    momentum: float = 0.9
    weight_decay: float = 1e-4
    eval_every: int = 5
    client_val_eval_every: int = 10
    client_val_num_clients: int = 0  # 0 means all clients when client validation runs.

    # Methods to run in a single comparison
    methods: List[str] = None

    # FedProx
    fedprox_mu: float = 0.01

    # FedSAM
    sam_rho: float = 0.05

    # FedGuCci base
    beta: float = 0.1
    num_anchors: int = 3
    alpha_interp_samples: int = 1

    # Heterogeneity-aware beta
    adaptive_beta: bool = True
    adaptive_beta_scale: float = 1.0

    # Anchor variants
    top_q: int = 5
    ema_tau: float = 0.9

    # Metric options
    compute_connectivity_barrier: bool = True #False
    barrier_num_clients: int = 5

    def __post_init__(self):
        if self.methods is None:
            self.methods = [
                "fedavg",
                "fedprox",
                "fedsam",
                "fedgucci",
                "fedgucci_adaptive_beta",
                "fedgucci_best_anchor",
                "fedgucci_topq_anchor",
                "fedgucci_ema_anchor",
                "fedgucci_topq_ema",
                "fedgucci_hsa",
            ]

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)
