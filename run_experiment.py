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
    p.add_argument("--run_tag", type=str, default="")
    p.add_argument("--num_clients", type=int, default=50)
    p.add_argument("--dirichlet_alpha", type=float, default=0.01)
    p.add_argument("--min_client_size", type=int, default=20)
    p.add_argument("--rounds", type=int, default=100)
    p.add_argument("--local_epochs", type=int, default=3)
    p.add_argument("--participation_rate", type=float, default=1.0)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--eval_every", type=int, default=1)
    p.add_argument("--client_val_eval_every", type=int, default=1)
    p.add_argument("--client_val_num_clients", type=int, default=0)
    p.add_argument("--beta", type=float, default=0.1)
    p.add_argument("--num_anchors", type=int, default=4)
    p.add_argument("--top_q", type=int, default=5)
    p.add_argument("--ema_tau", type=float, default=0.9)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--methods", type=str, default="fedavg,fedprox,fedsam,fedgucci,fedgucci_adaptive_beta,fedgucci_best_anchor,fedgucci_topq_anchor,fedgucci_ema_anchor,fedgucci_topq_ema,fedgucci_hsa") #fedavg,fedprox,fedsam,fedgucci,fedgucci_adaptive_beta,fedgucci_topq_ema,fedgucci_hsa
    p.add_argument("--compute_connectivity_barrier", action="store_true") 
    p.add_argument("--skip_combined_outputs", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = ExperimentConfig()
    for k, v in vars(args).items():
        if k == "methods":
            cfg.methods = [m.strip() for m in v.split(",") if m.strip()]
        else:
            setattr(cfg, k, v)

    if cfg.run_tag and not all(ch.isalnum() or ch in "_.-" for ch in cfg.run_tag):
        raise ValueError("--run_tag may only contain letters, numbers, underscore, dash, or dot")
    if cfg.min_client_size < 0:
        raise ValueError("--min_client_size must be >= 0")
    if cfg.eval_every <= 0:
        raise ValueError("--eval_every must be positive")
    if cfg.client_val_eval_every <= 0:
        raise ValueError("--client_val_eval_every must be positive")
    if cfg.client_val_num_clients < 0:
        raise ValueError("--client_val_num_clients must be >= 0")

    run_dir = os.path.join(
        cfg.output_dir,
        f"{cfg.experiment_name}_{cfg.dataset}_alpha{cfg.dirichlet_alpha}_seed{cfg.seed}",
    )
    os.makedirs(run_dir, exist_ok=True)
    config_filename = "config.json"
    if args.skip_combined_outputs and len(cfg.methods) == 1:
        config_stem = cfg.methods[0]
        if cfg.run_tag:
            config_stem = f"{config_stem}_{cfg.run_tag}"
        config_filename = f"config_{config_stem}.json"
    cfg.save(os.path.join(run_dir, config_filename))

    print("Preparing federated data...")
    fed_data = prepare_federated_data(cfg)

    write_combined_outputs = not args.skip_combined_outputs

    print("Running all requested methods...")
    combined = run_all_methods(cfg, fed_data, run_dir, write_combined=write_combined_outputs)

    if write_combined_outputs:
        print("Generating plots...")
        make_plots(os.path.join(run_dir, "metrics_all_methods.csv"), os.path.join(run_dir, "plots"))
    else:
        print("Skipping combined metrics and plots for this single-method job.")

    print("\nDone.")
    print(f"Results saved to: {run_dir}")


if __name__ == "__main__":
    main()
