from collections import Counter
from typing import Dict, List, Tuple
import numpy as np


def dirichlet_partition(
    labels,
    num_clients: int,
    alpha: float,
    num_classes: int,
    min_size: int = 20,
    seed: int = 42,
) -> Tuple[Dict[int, List[int]], Dict[int, np.ndarray]]:
    """Dirichlet label-skew split."""
    rng = np.random.default_rng(seed)
    labels = np.asarray(labels)

    while True:
        client_indices = {i: [] for i in range(num_clients)}

        for c in range(num_classes):
            idx_c = np.where(labels == c)[0]
            rng.shuffle(idx_c)
            proportions = rng.dirichlet(alpha * np.ones(num_clients))
            proportions = proportions / proportions.sum()
            split_points = (np.cumsum(proportions) * len(idx_c)).astype(int)[:-1]
            splits = np.split(idx_c, split_points)
            for i, idx in enumerate(splits):
                client_indices[i].extend(idx.tolist())

        sizes = [len(client_indices[i]) for i in range(num_clients)]
        if min(sizes) >= min_size:
            break

    class_counts = {}
    for i in range(num_clients):
        rng.shuffle(client_indices[i])
        counts = np.zeros(num_classes, dtype=np.int64)
        counter = Counter(labels[client_indices[i]])
        for cls, val in counter.items():
            counts[int(cls)] = int(val)
        class_counts[i] = counts
    return client_indices, class_counts


def split_client_train_val(client_indices, val_fraction: float = 0.15, seed: int = 42):
    rng = np.random.default_rng(seed)
    train_indices = {}
    val_indices = {}
    for cid, idxs in client_indices.items():
        idxs = np.asarray(idxs)
        rng.shuffle(idxs)
        n_val = max(1, int(len(idxs) * val_fraction))
        val_indices[cid] = idxs[:n_val].tolist()
        train_indices[cid] = idxs[n_val:].tolist()
    return train_indices, val_indices


def compute_kl_heterogeneity_scores(class_counts: Dict[int, np.ndarray]) -> Dict[int, float]:
    all_counts = np.sum(np.stack(list(class_counts.values())), axis=0)
    global_p = all_counts / max(all_counts.sum(), 1)
    eps = 1e-12
    scores = {}
    for cid, counts in class_counts.items():
        p = counts / max(counts.sum(), 1)
        scores[cid] = float(np.sum(p * (np.log(p + eps) - np.log(global_p + eps))))
    return scores


def normalize_scores(scores: Dict[int, float]) -> Dict[int, float]:
    vals = np.asarray(list(scores.values()), dtype=float)
    lo, hi = float(vals.min()), float(vals.max())
    if hi - lo < 1e-12:
        return {k: 0.0 for k in scores}
    return {k: float((v - lo) / (hi - lo)) for k, v in scores.items()}
