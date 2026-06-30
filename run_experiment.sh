#!/bin/bash -l
#SBATCH --job-name=fedgucci_gpu
#SBATCH --output=logs/fedgucci_%A_%a.out
#SBATCH --error=logs/fedgucci_%A_%a.err
#SBATCH --array=0-56
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1

set -euo pipefail

module purge
module load python/3.12.0/default
module load cuda/12.8/default

mkdir -p logs

cd /speed-scratch/ol_tal/simulation/local_global/fedgucci_experiment

echo "Node: $(hostname)"
echo "Python path:"
which python
python -V

echo "NVIDIA driver/GPU info:"
nvidia-smi || true

echo "Checking PyTorch/CUDA..."
python - <<'PY'
import sys
import torch

print("torch:", torch.__version__)
print("torch file:", torch.__file__)
print("torch CUDA build:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
if not torch.cuda.is_available():
    sys.exit(
        "ERROR: CUDA was requested, but PyTorch cannot use the GPU. "
        "This is usually a node driver vs PyTorch CUDA-build mismatch."
    )
print("GPU:", torch.cuda.get_device_name(0))
PY
python -c "import torchvision; print('torchvision:', torchvision.__version__, 'file:', torchvision.__file__)"

# Format: "method run_tag beta num_anchors top_q ema_tau"
# Each row is one Slurm array task. Add only combinations that actually matter.
# Defaults are beta=0.1, num_anchors=4, top_q=5, ema_tau=0.9.
# After editing this list, set #SBATCH --array to 0-(number of JOBS - 1).
JOBS=(
  # Base run for every method.
  "fedavg base 0.1 4 5 0.9"
  "fedprox base 0.1 4 5 0.9"
  "fedsam base 0.1 4 5 0.9"
  "fedgucci base 0.1 4 5 0.9"
  "fedgucci_adaptive_beta base 0.1 4 5 0.9"
  "fedgucci_best_anchor base 0.1 4 5 0.9"
  "fedgucci_topq_anchor base 0.1 4 5 0.9"
  "fedgucci_ema_anchor base 0.1 4 5 0.9"
  "fedgucci_topq_ema base 0.1 4 5 0.9"
  "fedgucci_hsa base 0.1 4 5 0.9"

  # beta = [0.05, 0.1, 0.2]. Base rows already cover beta=0.1.
  "fedgucci b0p05 0.05 4 5 0.9"
  "fedgucci b0p2 0.2 4 5 0.9"
  "fedgucci_adaptive_beta b0p05 0.05 4 5 0.9"
  "fedgucci_adaptive_beta b0p2 0.2 4 5 0.9"
  "fedgucci_best_anchor b0p05 0.05 4 5 0.9"
  "fedgucci_best_anchor b0p2 0.2 4 5 0.9"
  "fedgucci_topq_anchor b0p05 0.05 4 5 0.9"
  "fedgucci_topq_anchor b0p2 0.2 4 5 0.9"
  "fedgucci_ema_anchor b0p05 0.05 4 5 0.9"
  "fedgucci_ema_anchor b0p2 0.2 4 5 0.9"
  "fedgucci_topq_ema b0p05 0.05 4 5 0.9"
  "fedgucci_topq_ema b0p2 0.2 4 5 0.9"
  "fedgucci_hsa b0p05 0.05 4 5 0.9"
  "fedgucci_hsa b0p2 0.2 4 5 0.9"

  # num_anchors = [3, 4, 5, 6]. Base rows already cover num_anchors=4.
  "fedgucci a3 0.1 3 5 0.9"
  "fedgucci a5 0.1 5 5 0.9"
  "fedgucci a6 0.1 6 5 0.9"
  "fedgucci_adaptive_beta a3 0.1 3 5 0.9"
  "fedgucci_adaptive_beta a5 0.1 5 5 0.9"
  "fedgucci_adaptive_beta a6 0.1 6 5 0.9"
  "fedgucci_best_anchor a3 0.1 3 5 0.9"
  "fedgucci_best_anchor a5 0.1 5 5 0.9"
  "fedgucci_best_anchor a6 0.1 6 5 0.9"
  "fedgucci_topq_anchor a3 0.1 3 5 0.9"
  "fedgucci_topq_anchor a5 0.1 5 5 0.9"
  "fedgucci_topq_anchor a6 0.1 6 5 0.9"

  # top_q = [3, 4, 5, 6]. Base rows already cover top_q=5.
  "fedgucci_topq_anchor q3 0.1 4 3 0.9"
  "fedgucci_topq_anchor q4 0.1 4 4 0.9"
  "fedgucci_topq_anchor q6 0.1 4 6 0.9"
  "fedgucci_topq_ema q3 0.1 4 3 0.9"
  "fedgucci_topq_ema q4 0.1 4 4 0.9"
  "fedgucci_topq_ema q6 0.1 4 6 0.9"
  "fedgucci_hsa q3 0.1 4 3 0.9"
  "fedgucci_hsa q4 0.1 4 4 0.9"
  "fedgucci_hsa q6 0.1 4 6 0.9"

  # ema_tau = [0.4, 0.5, 0.6, 0.8, 0.9]. Base rows already cover ema_tau=0.9.
  "fedgucci_ema_anchor ema0p4 0.1 4 5 0.4"
  "fedgucci_ema_anchor ema0p5 0.1 4 5 0.5"
  "fedgucci_ema_anchor ema0p6 0.1 4 5 0.6"
  "fedgucci_ema_anchor ema0p8 0.1 4 5 0.8"
  "fedgucci_topq_ema ema0p4 0.1 4 5 0.4"
  "fedgucci_topq_ema ema0p5 0.1 4 5 0.5"
  "fedgucci_topq_ema ema0p6 0.1 4 5 0.6"
  "fedgucci_topq_ema ema0p8 0.1 4 5 0.8"
  "fedgucci_hsa ema0p4 0.1 4 5 0.4"
  "fedgucci_hsa ema0p5 0.1 4 5 0.5"
  "fedgucci_hsa ema0p6 0.1 4 5 0.6"
  "fedgucci_hsa ema0p8 0.1 4 5 0.8"
)

if (( SLURM_ARRAY_TASK_ID >= ${#JOBS[@]} )); then
  echo "Invalid SLURM_ARRAY_TASK_ID: ${SLURM_ARRAY_TASK_ID}; only ${#JOBS[@]} jobs are configured."
  exit 1
fi

read -r METHOD RUN_TAG BETA NUM_ANCHORS TOP_Q EMA_TAU <<< "${JOBS[$SLURM_ARRAY_TASK_ID]}"

if [[ -z "${METHOD}" || -z "${RUN_TAG}" ]]; then
  echo "Invalid job row for SLURM_ARRAY_TASK_ID: ${SLURM_ARRAY_TASK_ID}"
  exit 1
fi

echo "Selected method: ${METHOD}"
echo "Selected parameters: tag=${RUN_TAG} beta=${BETA} num_anchors=${NUM_ANCHORS} top_q=${TOP_Q} ema_tau=${EMA_TAU}"
echo "Starting experiment..."
python run_experiment.py \
  --device cuda \
  --experiment_name "fedgucci_compare" \
  --run_tag "${RUN_TAG}" \
  --methods "${METHOD}" \
  --beta "${BETA}" \
  --num_anchors "${NUM_ANCHORS}" \
  --top_q "${TOP_Q}" \
  --ema_tau "${EMA_TAU}" \
  --skip_combined_outputs \
  --eval_every 1 \
  --client_val_eval_every 1 \
  --client_val_num_clients 0
