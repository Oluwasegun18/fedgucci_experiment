# FedGuCci Baseline and Variant Experiment Framework

This project compares standard FL baselines and FedGuCci variants under Dirichlet non-IID label skew.

## Methods implemented

- `fedavg`
- `fedprox`
- `fedsam`
- `fedgucci`
- `fedgucci_adaptive_beta`
- `fedgucci_best_anchor`
- `fedgucci_topq_anchor`
- `fedgucci_ema_anchor`
- `fedgucci_topq_ema`
- `fedgucci_hsa`

`fedgucci_hsa` means heterogeneity-aware stable-anchor FedGuCci. It uses top-q anchor initialization, EMA anchor update, and adaptive client-specific beta.

## Run a quick experiment

```bash
python run_experiment.py \
  --dataset cifar10 \
  --num_clients 20 \
  --dirichlet_alpha 0.5 \
  --rounds 20 \
  --local_epochs 1 \
  --methods fedavg,fedprox,fedgucci,fedgucci_topq_ema,fedgucci_hsa
```

## Run the fuller comparison

```bash
python run_experiment.py \
  --dataset cifar10 \
  --num_clients 50 \
  --dirichlet_alpha 0.5 \
  --rounds 100 \
  --local_epochs 3 \
  --participation_rate 1.0 \
  --methods fedavg,fedprox,fedsam,fedgucci,fedgucci_adaptive_beta,fedgucci_best_anchor,fedgucci_topq_anchor,fedgucci_ema_anchor,fedgucci_topq_ema,fedgucci_hsa \
  --compute_connectivity_barrier
```

## Outputs

The script creates:

- `metrics_<method>.csv`
- `metrics_all_methods.csv`
- `final_model_<method>.pt`
- `plots/compare_test_accuracy.png`
- `plots/compare_client_mean_accuracy.png`
- `plots/compare_worst10_accuracy.png`
- `plots/final_test_accuracy_bar.png`
- `plots/final_round_summary.csv`

## Notes

- The code uses a small CNN without BatchNorm so that the FedGuCci connectivity loss can be computed through differentiable stateless parameter interpolation.
- The connectivity loss is computed as:

  `E_alpha L_i(alpha*w_i + (1-alpha)*w_anchor)`

- Adaptive beta uses normalized KL divergence between client and global class distributions:

  `beta_i = beta / (1 + adaptive_beta_scale * s_i)`

