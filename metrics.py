from typing import Dict, List
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from model_utils import average_state_dicts


def evaluate_model(model, state, dataset, device, batch_size=256, indices=None):
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()
    if indices is None:
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    else:
        loader = DataLoader(Subset(dataset, indices), batch_size=batch_size, shuffle=False, num_workers=2)

    total, correct, loss_sum = 0, 0, 0.0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            pred = logits.argmax(1)
            total += y.numel()
            correct += (pred == y).sum().item()
            loss_sum += loss.item() * y.numel()
    return {"loss": loss_sum / max(total, 1), "acc": correct / max(total, 1)}


def evaluate_client_val_stats(model, state, train_dataset, val_indices: Dict[int, List[int]], device, batch_size=256):
    accs = []
    losses = []
    for cid in sorted(val_indices.keys()):
        if len(val_indices[cid]) == 0:
            continue
        res = evaluate_model(model, state, train_dataset, device, batch_size=batch_size, indices=val_indices[cid])
        accs.append(res["acc"])
        losses.append(res["loss"])
    accs_np = np.asarray(accs, dtype=float)
    if accs_np.size == 0:
        return {"client_val_acc_mean": 0.0, "client_val_acc_worst10": 0.0, "client_val_acc_std": 0.0, "client_val_acc_gap": 0.0}
    q = max(1, int(np.ceil(0.1 * len(accs_np))))
    worst10 = np.sort(accs_np)[:q].mean()
    return {
        "client_val_acc_mean": float(accs_np.mean()),
        "client_val_acc_worst10": float(worst10),
        "client_val_acc_std": float(accs_np.std()),
        "client_val_acc_gap": float(accs_np.max() - accs_np.min()),
    }


def compute_connectivity_barrier(model, local_states, test_dataset, device, batch_size=256):
    """Accuracy barrier: 1 - Acc(avg local model) / mean_i Acc(local_i). Lower is better."""
    if len(local_states) < 2:
        return 0.0
    accs = []
    for sd in local_states:
        res = evaluate_model(model, sd, test_dataset, device, batch_size=batch_size)
        accs.append(res["acc"])
    avg_state = average_state_dicts(local_states)
    avg_res = evaluate_model(model, avg_state, test_dataset, device, batch_size=batch_size)
    mean_ind = float(np.mean(accs))
    return float(1.0 - avg_res["acc"] / max(mean_ind, 1e-12))
