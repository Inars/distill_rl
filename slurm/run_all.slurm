#!/bin/bash
# The four experiments as one Slurm array, one GPU node each. Submit with:
#     sbatch slurm/run_all.slurm
#
#   idx  method         what
#    0   sft            supervised CE on gold labels
#    1   kd             Hinton logit KD from roberta-large-mnli
#    2   grpo           single-step bandit GRPO, reference = frozen student snapshot
#    3   grpo_distill   altered GRPO (reference = roberta-large-mnli) + soft KD aux
#
#SBATCH --job-name=distill_rl
#SBATCH --output=slurm/%x-%A_%a.out
#SBATCH --error=slurm/%x-%A_%a.err
#SBATCH --partition=gpu-best
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --array=0-3
# Same node set as experimental_grow/distill_nli's run_nli_grid.slurm: exclude the
# old rtx/turing/pascal cards, leaving V100 + A100 (ampere) + H100 (hopper).
#SBATCH --exclude=margpu002,margpu003,margpu017,margpu018,margpu019,margpu020,margpu021,margpu022,margpu023,margpu024,margpu025,margpu026,margpu027,margpu028,margpu029

set -uo pipefail
cd /home/tau/elsaad/distill_rl

# W&B credentials (gitignored .env). Never hardcode the key in tracked files.
set -a; [ -f .env ] && . ./.env; set +a

METHODS=(sft kd grpo grpo_distill)
M=${METHODS[$SLURM_ARRAY_TASK_ID]}

echo "[array $SLURM_ARRAY_TASK_ID] method=$M node=$(hostname)"
nvidia-smi -L || true

uv run python hydra_script/train.py method="$M"

echo "[array $SLURM_ARRAY_TASK_ID] method=$M DONE (exit=$?)"
