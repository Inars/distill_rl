#!/bin/bash
# Run a single experiment (optionally with extra Hydra overrides). Examples:
#     sbatch slurm/run_one.slurm grpo_distill
#     sbatch slurm/run_one.slurm grpo grpo.beta=0.02 grpo.group_size=16
#     sbatch slurm/run_one.slurm kd distill.alpha=0.7 epochs=5
#
#SBATCH --job-name=distill_rl_one
#SBATCH --output=slurm/%x-%j.out
#SBATCH --error=slurm/%x-%j.err
#SBATCH --partition=gpu-best
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00

set -uo pipefail
cd /home/tau/elsaad/distill_rl

set -a; [ -f .env ] && . ./.env; set +a

METHOD=${1:?usage: sbatch slurm/run_one.slurm <method> [hydra overrides...]}
shift || true

echo "[run_one] method=$METHOD overrides=$* node=$(hostname)"
nvidia-smi -L || true

uv run python hydra_script/train.py method="$METHOD" "$@"

echo "[run_one] method=$METHOD DONE (exit=$?)"
