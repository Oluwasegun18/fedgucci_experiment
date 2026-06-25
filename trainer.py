import os
import random
from typing import Dict, List
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from anchors import AnchorManager
from local_train import train_one_client
from metrics import evaluate_model, evaluate_client_val_stats, compute_connectivity_barrier
from model_utils import clone_state_dict, weighted_client_average
from models import build_model


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def method_is_fedgucci(method: str) -> bool:
    return method.startswith("fedgucci")


def beta_for_client(method: str, config, cid: int, hetero_scores: Dict[int, float]):
    if method in {"fedgucci_adaptive_beta", "fedgucci_hsa"}:
        s_i = hetero_scores.get(cid, 0.0)
        return config.beta / (1.0 + config.adaptive_beta_scale * s_i)
    return config.beta


def evaluate_local_client_scores(model, client_states, train_dataset, val_indices, selected_clients, device, batch_size):
    scores = []
    for state, cid in zip(client_states, selected_clients):
        if len(val_indices[cid]) == 0:
            scores.append(0.0)
        else:
            res = evaluate_model(model, state, train_dataset, device, batch_size=batch_size, indices=val_indices[cid])
            scores.append(res["acc"])
    return scores


def select_client_val_indices(val_indices, config):
    max_clients = getattr(config, "client_val_num_clients", 0)
    if max_clients <= 0:
        return val_indices

    eligible = [cid for cid in sorted(val_indices.keys()) if len(val_indices[cid]) > 0]
    if max_clients >= len(eligible):
        return {cid: val_indices[cid] for cid in eligible}

    sampled = sorted(random.sample(eligible, max_clients))
    return {cid: val_indices[cid] for cid in sampled}


def run_single_method(method: str, config, fed_data, run_dir: str):
    train_set, test_set, train_indices, val_indices, class_counts, hetero_scores = fed_data
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    seed_everything(config.seed)

    model = build_model(config.dataset, config.num_classes)
    global_state = clone_state_dict(model.state_dict())
    anchor_manager = AnchorManager(method, config)

    if method_is_fedgucci(method) and not anchor_manager.needs_best_client_initialization():
        anchor_manager.initialize_with_global(global_state)

    rows = []
    num_selected = max(1, int(config.participation_rate * config.num_clients))
    client_ids = list(range(config.num_clients))

    for rnd in range(1, config.rounds + 1):
        selected = random.sample(client_ids, num_selected)
        anchors = anchor_manager.get_anchors()
        local_states = []
        local_sizes = []

        for cid in selected:
            local_model = build_model(config.dataset, config.num_classes)
            beta_i = beta_for_client(method, config, cid, hetero_scores)
            local_state = train_one_client(
                model=local_model,
                global_state=global_state,
                train_dataset=train_set,
                indices=train_indices[cid],
                method=method,
                config=config,
                device=device,
                anchor_states=anchors,
                beta_i=beta_i,
            )
            local_states.append(local_state)
            local_sizes.append(len(train_indices[cid]))

        # First-round best/top-q anchor initialization happens after local training, before normal anchor update.
        if method_is_fedgucci(method) and anchor_manager.needs_best_client_initialization() and not anchor_manager.initialized:
            scores = evaluate_local_client_scores(
                model=model,
                client_states=local_states,
                train_dataset=train_set,
                val_indices=val_indices,
                selected_clients=selected,
                device=device,
                batch_size=max(config.batch_size, 128),
            )
            anchor_manager.initialize_with_clients(local_states, scores)

        global_state = weighted_client_average(local_states, local_sizes)
        anchor_manager.update_after_round(global_state)

        if rnd % config.eval_every == 0 or rnd == 1 or rnd == config.rounds:
            test_res = evaluate_model(model, global_state, test_set, device, batch_size=256)
            should_eval_client_val = (
                rnd == 1
                or rnd == config.rounds
                or rnd % config.client_val_eval_every == 0
            )
            if should_eval_client_val:
                eval_val_indices = select_client_val_indices(val_indices, config)
                client_stats = evaluate_client_val_stats(
                    model, global_state, train_set, eval_val_indices, device, batch_size=256
                )
                client_val_clients = sum(1 for idxs in eval_val_indices.values() if len(idxs) > 0)
            else:
                client_stats = {
                    "client_val_acc_mean": np.nan,
                    "client_val_acc_worst10": np.nan,
                    "client_val_acc_std": np.nan,
                    "client_val_acc_gap": np.nan,
                }
                client_val_clients = 0
            row = {
                "method": method,
                "round": rnd,
                "test_loss": test_res["loss"],
                "test_acc": test_res["acc"],
                "client_val_clients": client_val_clients,
                **client_stats,
            }
            print('Method: ', method, ' | Round: ', rnd, ' | Test Loss: ', test_res["loss"], ' | Test Accuracy: ', test_res["acc"])
            if config.compute_connectivity_barrier:
                k = min(config.barrier_num_clients, len(local_states))
                row["connectivity_barrier"] = compute_connectivity_barrier(
                    model, local_states[:k], test_set, device, batch_size=256
                )
            rows.append(row)
            client_mean = (
                f"{row['client_val_acc_mean']:.4f}"
                if pd.notna(row["client_val_acc_mean"])
                else "skipped"
            )
            worst10 = (
                f"{row['client_val_acc_worst10']:.4f}"
                if pd.notna(row["client_val_acc_worst10"])
                else "skipped"
            )
            print(
                f"[{method}] round {rnd:03d}/{config.rounds} | "
                f"test_acc={row['test_acc']:.4f} | client_mean={client_mean} | "
                f"worst10={worst10}"
            )

    df = pd.DataFrame(rows)
    os.makedirs(run_dir, exist_ok=True)
    df.to_csv(os.path.join(run_dir, f"metrics_{method}.csv"), index=False)
    torch.save(global_state, os.path.join(run_dir, f"final_model_{method}.pt"))
    return df


def run_all_methods(config, fed_data, run_dir: str):
    all_dfs = []
    for method in config.methods:
        print("\n" + "=" * 90)
        print(f"Running method: {method}")
        print("=" * 90)
        df = run_single_method(method, config, fed_data, run_dir)
        all_dfs.append(df)
    combined = pd.concat(all_dfs, ignore_index=True)
    combined.to_csv(os.path.join(run_dir, "metrics_all_methods.csv"), index=False)
    return combined
