#!/bin/bash
#SBATCH --job-name=fedgucci_gpu
#SBATCH --output=logs/fedgucci_%j.out
#SBATCH --error=logs/fedgucci_%j.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1

mkdir -p logs

cd /speed-scratch/ol_tal/simulation/local_global/fedgucci_experiment


python run_experiment.py