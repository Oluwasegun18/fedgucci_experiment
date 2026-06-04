import os
import pandas as pd
import matplotlib.pyplot as plt


def _plot_metric(df, metric, ylabel, out_path):
    plt.figure(figsize=(8, 5))
    for method, sub in df.groupby("method"):
        sub = sub.sort_values("round")
        plt.plot(sub["round"], sub[metric], label=method)
    plt.xlabel("Communication round")
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def make_plots(metrics_csv: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(metrics_csv)

    plot_specs = [
        ("test_acc", "Global test accuracy", "compare_test_accuracy.png"),
        ("client_val_acc_mean", "Mean client validation accuracy", "compare_client_mean_accuracy.png"),
        ("client_val_acc_worst10", "Worst 10% client validation accuracy", "compare_worst10_accuracy.png"),
        ("client_val_acc_std", "Std. of client validation accuracy", "compare_client_accuracy_std.png"),
        ("client_val_acc_gap", "Client accuracy gap", "compare_client_accuracy_gap.png"),
    ]

    for metric, ylabel, filename in plot_specs:
        if metric in df.columns:
            _plot_metric(df, metric, ylabel, os.path.join(output_dir, filename))

    if "connectivity_barrier" in df.columns:
        _plot_metric(df, "connectivity_barrier", "Connectivity barrier", os.path.join(output_dir, "compare_connectivity_barrier.png"))

    # Final-round summary bar chart
    final = df.sort_values("round").groupby("method").tail(1).sort_values("test_acc", ascending=False)
    plt.figure(figsize=(9, 5))
    plt.bar(final["method"], final["test_acc"])
    plt.ylabel("Final global test accuracy")
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "final_test_accuracy_bar.png"), dpi=300)
    plt.close()

    final.to_csv(os.path.join(output_dir, "final_round_summary.csv"), index=False)
