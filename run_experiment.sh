#!/bin/bash -l
#SBATCH --job-name=fedgucci_gpu
#SBATCH --output=logs/fedgucci_%A_%a.out
#SBATCH --error=logs/fedgucci_%A_%a.err
#SBATCH --array=0-9
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1

module purge
module load python/3.12.0/default
module load cuda/12.8/default

mkdir -p logs

cd /speed-scratch/ol_tal/simulation/local_global/fedgucci_experiment

echo "Python path:"
which python
python -V

echo "Checking PyTorch..."
python -c "import torch; print('torch:', torch.__version__)"
python -c "import torchvision; print('torchvision:', torchvision.__version__)"
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"

METHODS=(
  fedavg
  fedprox
  fedsam
  fedgucci
  fedgucci_adaptive_beta
  fedgucci_best_anchor
  fedgucci_topq_anchor
  fedgucci_ema_anchor
  fedgucci_topq_ema
  fedgucci_hsa
)

METHOD="${METHODS[$SLURM_ARRAY_TASK_ID]}"
if [[ -z "${METHOD}" ]]; then
  echo "Invalid SLURM_ARRAY_TASK_ID: ${SLURM_ARRAY_TASK_ID}"
  exit 1
fi

echo "Selected method: ${METHOD}"

echo "Starting experiment..."
python run_experiment.py \
  --device cuda \
  --experiment_name "fedgucci_compare_${METHOD}" \
  --methods "${METHOD}" \
  --eval_every 1 \
  --client_val_eval_every 10 \
  --client_val_num_clients 5