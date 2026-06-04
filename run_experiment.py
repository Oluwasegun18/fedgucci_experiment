import argparse
import os
from dataclasses import asdict

from config import ExperimentConfig
from data import prepare_federated_data
from plots import make_plots
from trainer import run_all_methods


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=str, default="cifar10", choices=["cifar10", "fashionmnist"])
    p.add_argument("--data_dir", type=str, default="./data")
    p.add_argument("--output_dir", type=str, default="./results")
    p.add_argument("--experiment_name", type=str, default="fedgucci_compare")
    p.add_argument("--num_clients", type=int, default=50)
    p.add_argument("--dirichlet_alpha", type=float, default=0.01)
    p.add_argument("--rounds", type=int, default=100)
    p.add_argument("--local_epochs", type=int, default=3)
    p.add_argument("--participation_rate", type=float, default=1.0)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--beta", type=float, default=0.1)
    p.add_argument("--num_anchors", type=int, default=4)
    p.add_argument("--top_q", type=int, default=5)
    p.add_argument("--ema_tau", type=float, default=0.9)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="gpu")
    p.add_argument("--methods", type=str, default="fedavg,fedprox,fedsam,fedgucci,fedgucci_adaptive_beta,fedgucci_best_anchor,fedgucci_topq_anchor,fedgucci_ema_anchor,fedgucci_topq_ema,fedgucci_hsa") #fedavg,fedprox,fedsam,fedgucci,fedgucci_adaptive_beta,fedgucci_topq_ema,fedgucci_hsa
    p.add_argument("--compute_connectivity_barrier", action="store_true") 
    return p.parse_args()


def main():
    args = parse_args()
    cfg = ExperimentConfig()
    for k, v in vars(args).items():
        if k == "methods":
            cfg.methods = [m.strip() for m in v.split(",") if m.strip()]
        else:
            setattr(cfg, k, v)

    run_dir = os.path.join(
        cfg.output_dir,
        f"{cfg.experiment_name}_{cfg.dataset}_alpha{cfg.dirichlet_alpha}_seed{cfg.seed}",
    )
    os.makedirs(run_dir, exist_ok=True)
    cfg.save(os.path.join(run_dir, "config.json"))

    print("Preparing federated data...")
    fed_data = prepare_federated_data(cfg)

    print("Running all requested methods...")
    combined = run_all_methods(cfg, fed_data, run_dir)

    print("Generating plots...")
    make_plots(os.path.join(run_dir, "metrics_all_methods.csv"), os.path.join(run_dir, "plots"))

    print("\nDone.")
    print(f"Results saved to: {run_dir}")


if __name__ == "__main__":
    main()
