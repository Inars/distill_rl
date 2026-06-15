# distill_rl

Experiments on **how distillation fits into the GRPO algorithm**, on MNLI.

A `roberta-base` student is trained four ways and compared:

| `method=`      | algorithm | teacher role |
|----------------|-----------|--------------|
| `sft`          | supervised cross-entropy on gold labels | — |
| `kd`           | supervised Hinton **logit KD** | `roberta-large-mnli` (soft targets) |
| `grpo`         | single-step **bandit GRPO** | reference = frozen student snapshot |
| `grpo_distill` | **altered GRPO** + soft KD aux | `roberta-large-mnli` as KL/penalty **reference** *and* KD teacher |

- **Student:** `FacebookAI/roberta-base` + a fresh 3-way head.
- **Teacher:** `FacebookAI/roberta-large-mnli` (frozen, logits permuted to canonical order).
- **Data:** `SetFit/mnli` (label order `0=entailment, 1=neutral, 2=contradiction`).
- **Tracking:** Weights & Biases → `TAU-Frugal/distillation-rl`.

See [PLAN.md](PLAN.md) for the design rationale and the exact GRPO math.

## GRPO, in one paragraph

Each MNLI example is a one-step episode. The policy `pi_theta(a|x)=softmax(logits)`
is a categorical over the 3 labels. We sample `G` actions per example, reward
correctness, form **group-relative advantages** `(r-mean)/(std+eps)`, and update
with the PPO-clipped objective plus `beta * KL(pi_theta || pi_ref)`. The loss math
mirrors TRL's `GRPOTrainer`; the only difference is the "sequence" is a single
label token. In `grpo` the reference is a frozen snapshot of the student; in
`grpo_distill` it is the teacher (`roberta-large-mnli`), and a soft KD term from
the teacher is added on top.

## Setup

```bash
uv sync                      # builds .venv (torch cu130, transformers, trl, wandb, ...)
cp .env.example .env         # then put your WANDB_API_KEY in .env  (gitignored)
# or: wandb login
```

The W&B key is read from the environment / `wandb login` cache only — it is never
stored in a tracked file.

## Run

Locally (single GPU or CPU):
```bash
uv run python hydra_script/train.py method=sft
uv run python hydra_script/train.py method=kd
uv run python hydra_script/train.py method=grpo
uv run python hydra_script/train.py method=grpo_distill
```

Quick smoke (no GPU, no W&B):
```bash
uv run python hydra_script/train.py method=grpo_distill \
  epochs=1 data.train_limit=64 data.val_limit=64 data.num_workers=0 \
  student.device=cpu teacher.device=cpu wandb.mode=disabled
```

On the cluster (SLURM):
```bash
sbatch slurm/run_all.slurm                 # all 4 experiments, one GPU node each (array 0-3)
sbatch slurm/run_one.slurm grpo_distill    # a single experiment
sbatch slurm/run_one.slurm grpo grpo.beta=0.02 grpo.group_size=16   # with overrides
```

## Common overrides

| Knob | Default | Meaning |
|------|---------|---------|
| `epochs` | 3 | training epochs |
| `optimizer.lr` | 2e-5 | learning rate |
| `data.train_batch_size` | 16 | train batch |
| `grpo.group_size` | 8 | GRPO group size `G` |
| `grpo.beta` | 0.04 | KL penalty weight |
| `grpo.clip_eps` | 0.2 | PPO ratio clip |
| `grpo.kl_estimator` | `exact` | `exact` (closed-form) or `k3` (TRL estimator) |
| `distill.temperature` / `distill.alpha` | 2.0 / 0.5 | KD temperature / soft-vs-hard weight |
| `distill.lambda_kd` | 1.0 | weight of the soft KD aux in `grpo_distill` |
| `wandb.mode` | `online` | `online` / `offline` / `disabled` |

Model selection and the final reported number use the MNLI **validation** split
(the `test` split is the unlabeled GLUE set).

## Tests

```bash
uv run python -m pytest -q     # 30 CPU tests: data, models (incl. teacher permutation), KD, GRPO
```

## Layout

```
distill_rl/        package: data/ models/ distillation/ grpo/ training/ utils/
hydra_script/      train.py entrypoint + configs/ (method/ data/ student/ teacher/ grpo/ ...)
slurm/             run_all.slurm (array) + run_one.slurm
tests/             pytest suite
```
