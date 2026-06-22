# distill_rl

Experiments on **how distillation fits into the GRPO algorithm**, on MNLI.

A `roberta-base` student is trained several ways and compared:

| `method=`      | algorithm | teacher role |
|----------------|-----------|--------------|
| `sft`          | supervised cross-entropy on gold labels | — |
| `kd`           | supervised Hinton **logit KD** | `roberta-large-mnli` (soft targets) |
| `grpo`         | single-step **bandit GRPO** | reference = frozen student snapshot |
| `grpo_distill` | **altered GRPO** + soft KD aux | `roberta-large-mnli` as KL/penalty **reference** *and* KD teacher |
| `kdrl`         | **GRPO + reverse-KL (k2) distillation** (joint loss) | `roberta-large-mnli` as on-policy KD-RKL teacher (ref-KL off) |
| `kdrl_anneal`  | `kdrl` with an annealed KD coefficient (β: 5e-3 → 1e-3) | same as `kdrl` |
| `trrd`         | **GRPO with a mixture-anchored ratio** (RLAD) | `roberta-large-mnli` folded into the trust-region ratio |

- **Student:** `FacebookAI/roberta-base` + a fresh 3-way head.
- **Teacher:** `FacebookAI/roberta-large-mnli` (frozen, logits permuted to canonical order).
- **Data:** `SetFit/mnli` (label order `0=entailment, 1=neutral, 2=contradiction`).
- **Tracking:** Weights & Biases → `TAU-Frugal/distillation-rl`.

The base methods follow the GRPO math below; `kdrl` and `trrd` implement
[KDRL](https://arxiv.org/abs/2506.02208) and [RLAD / TRRD](https://arxiv.org/abs/2602.22495)
respectively (one-paragraph summaries follow).

## GRPO, in one paragraph

Each MNLI example is a one-step episode. The policy `pi_theta(a|x)=softmax(logits)`
is a categorical over the 3 labels. We sample `G` actions per example, reward
correctness, form **group-relative advantages** `(r-mean)/(std+eps)`, and update
with the PPO-clipped objective plus `beta * KL(pi_theta || pi_ref)`. The loss math
mirrors TRL's `GRPOTrainer`; the only difference is the "sequence" is a single
label token. In `grpo` the reference is a frozen snapshot of the student; in
`grpo_distill` it is the teacher (`roberta-large-mnli`), and a soft KD term from
the teacher is added on top.

## KDRL and TRRD, in one paragraph each

**`kdrl`** ([Xu et al.](https://arxiv.org/abs/2506.02208)) keeps the GRPO policy update
but adds an auxiliary **reverse-KL distillation** term toward the teacher, estimated on
the on-policy sampled actions with the unbiased **k2** estimator (`½·R²`,
`R = log π_T − log π_θ`), combined via a joint loss `L = L_GRPO + β·D_KL(π_θ‖π_T)`.
Faithful to the paper, the GRPO policy-anchoring KL is off (`grpo.beta=0`).
`kdrl_anneal` linearly decays `β` (5e-3 → 1e-3); optional reward-guided masking
(`kdrl.masking`) drops the KD term on already-correct samples. This is a *reverse* KL,
unlike `grpo_distill`'s forward/Hinton KD.

**`trrd`** ([Zhang et al.](https://arxiv.org/abs/2602.22495)) instead leaves the loss
shape alone and replaces the GRPO importance **ratio** with one anchored on a geometric
mixture of the old student policy and the teacher,
`r = π_θ / (π_θ_old^α · π_T^{1−α})`, then applies the usual PPO clip (plus an Appendix-B
clamp on `log(π_θ/π_T)`). There is no separate KD term — distillation is folded into the
trust region, active only when the advantage supports it. `α=1` recovers plain GRPO,
`α=0` is DPO-like; the Eq. 4 reference-KL to a frozen student snapshot is kept (`grpo.beta`).

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
uv run python hydra_script/train.py method=kdrl
uv run python hydra_script/train.py method=kdrl_anneal
uv run python hydra_script/train.py method=trrd
```

Quick smoke (no GPU, no W&B):
```bash
uv run python hydra_script/train.py method=grpo_distill \
  epochs=1 data.train_limit=64 data.val_limit=64 data.num_workers=0 \
  student.device=cpu teacher.device=cpu wandb.mode=disabled
```

On the cluster (SLURM):
```bash
# Default: the two new methods (kdrl, trrd), one GPU node each (array 0-1).
sbatch slurm/run_all.slurm

# Reproduce every method (override METHODS + the array range):
sbatch --array=0-6 \
       --export=ALL,METHODS="sft kd grpo grpo_distill kdrl kdrl_anneal trrd" \
       slurm/run_all.slurm

# A single experiment, optionally with Hydra overrides:
sbatch slurm/run_one.slurm trrd
sbatch slurm/run_one.slurm kdrl kdrl.estimator=k3 kdrl.masking=response
sbatch slurm/run_one.slurm grpo grpo.beta=0.02 grpo.group_size=16
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
| `kdrl.estimator` | `k2` | KD-RKL estimator: `k2` / `k3` / `exact` (`kdrl`) |
| `kdrl.beta` | 2e-3 | KD-RKL coefficient (constant schedule) |
| `kdrl.schedule` | `constant` | `constant` or `anneal` (β: `beta_init`→`beta_min`) |
| `kdrl.masking` | `none` | reward-guided masking: `none` / `response` / `group` |
| `trrd.alpha` | 0.5 | mixture coefficient (1 = GRPO, 0 = teacher-anchored) (`trrd`) |
| `trrd.distill_clip` | 1.0 | clamp on `log(π_θ/π_T)` |
| `wandb.mode` | `online` | `online` / `offline` / `disabled` |

Model selection and the final reported number use the MNLI **validation** split
(the `test` split is the unlabeled GLUE set).

## Tests

```bash
uv run python -m pytest -q     # 43 CPU tests: data, models (incl. teacher permutation), KD, GRPO, KDRL, TRRD
```

## Layout

```
distill_rl/        package: data/ models/ distillation/ grpo/ training/ utils/
hydra_script/      train.py entrypoint + configs/ (method/ data/ student/ teacher/ grpo/ ...)
slurm/             run_all.slurm (array) + run_one.slurm
tests/             pytest suite
```
