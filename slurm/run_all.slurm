#!/bin/bash
# Run a set of experiments as a Slurm array, one GPU node each.
#
# The method set is selectable via the METHODS env var (space-separated); the array
# size must match it. By DEFAULT this runs only the two NEW methods (kdrl, trrd) --
# the focus of current testing -- so a bare submit does not re-run the existing four:
#
#     sbatch slurm/run_all.slurm                       # -> kdrl, trrd   (array 0-1)
#
# To reproduce every method, override both METHODS and the array range:
#
#     sbatch --array=0-6 \
#            --export=ALL,METHODS="sft kd grpo grpo_distill kdrl kdrl_anneal trrd" \
#            slurm/run_all.slurm
#
# Any subset works the same way, e.g. just the KDRL variants:
#     sbatch --array=0-1 --export=ALL,METHODS="kdrl kdrl_anneal" slurm/run_all.slurm
#
#   method         what
#   sft            supervised CE on gold labels
#   kd             Hinton logit KD from roberta-large-mnli
#   grpo           single-step bandit GRPO, reference = frozen student snapshot
#   grpo_distill   altered GRPO (reference = roberta-large-mnli) + soft KD aux
#   kdrl           GRPO + on-policy reverse-KL (k2) distillation (joint loss; ref-KL off)
#   kdrl_anneal    kdrl with a linearly annealed KD coefficient (5e-3 -> 1e-3)
#   trrd           GRPO with the teacher/old-policy mixture-anchored ratio (RLAD)
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
#SBATCH --array=0-1
# Restrict to A100/A40 (ampere) + H100 (hopper). We exclude rtx/turing/pascal AND
# V100 (margpu004): torch 2.12+cu130 has no Volta sm_70 kernels (CUDA 13 dropped
# Volta), so V100 nodes crash with "no kernel image available for execution".
#SBATCH --exclude=margpu002,margpu003,margpu004,margpu017,margpu018,margpu019,margpu020,margpu021,margpu022,margpu023,margpu024,margpu025,margpu026,margpu027,margpu028,margpu029

set -uo pipefail
cd /home/tau/elsaad/distill_rl

# W&B credentials (gitignored .env). Never hardcode the key in tracked files.
set -a; [ -f .env ] && . ./.env; set +a

# Default to the two new methods; override with --export=ALL,METHODS="..."
read -r -a METHODS <<< "${METHODS:-kdrl trrd}"

if (( SLURM_ARRAY_TASK_ID >= ${#METHODS[@]} )); then
  echo "[array $SLURM_ARRAY_TASK_ID] out of range for METHODS=(${METHODS[*]}); " \
       "set --array=0-$(( ${#METHODS[@]} - 1 ))" >&2
  exit 2
fi

M=${METHODS[$SLURM_ARRAY_TASK_ID]}

echo "[array $SLURM_ARRAY_TASK_ID] method=$M node=$(hostname) (set: ${METHODS[*]})"
nvidia-smi -L || true

uv run python hydra_script/train.py method="$M"
rc=$?
echo "[array $SLURM_ARRAY_TASK_ID] method=$M DONE (exit=$rc)"
exit $rc
