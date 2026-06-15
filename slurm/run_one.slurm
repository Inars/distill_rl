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
#SBATCH --time=12:00:00
# A100/A40 (ampere) + H100 (hopper) only. V100 (margpu004) excluded too: torch
# 2.12+cu130 dropped Volta sm_70 kernels, so V100 nodes crash.
#SBATCH --exclude=margpu002,margpu003,margpu004,margpu017,margpu018,margpu019,margpu020,margpu021,margpu022,margpu023,margpu024,margpu025,margpu026,margpu027,margpu028,margpu029

set -uo pipefail
cd /home/tau/elsaad/distill_rl

set -a; [ -f .env ] && . ./.env; set +a

METHOD=${1:?usage: sbatch slurm/run_one.slurm <method> [hydra overrides...]}
shift || true

echo "[run_one] method=$METHOD overrides=$* node=$(hostname)"
nvidia-smi -L || true

uv run python hydra_script/train.py method="$METHOD" "$@"
rc=$?
echo "[run_one] method=$METHOD DONE (exit=$rc)"
exit $rc
