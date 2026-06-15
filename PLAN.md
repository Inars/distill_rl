# distill_rl — Project Plan / Workflow

> **Status:** DRAFT for review. Nothing in `distill_rl/` has been built yet except
> this file. Please leave comments inline (e.g. `<!-- ELIE: ... -->`) or edit any
> section directly, then tell me to proceed. I will **not** start writing code
> until you approve this plan.

---

## 0. Research question

How does **distillation** fit into the **GRPO** reinforcement-learning algorithm?
We compare four ways of training a small student (`roberta-base`) on MNLI:

1. **SFT** — classic supervised fine-tuning (cross-entropy on gold labels).
2. **GRPO** — single-step (bandit) GRPO with a frozen *student snapshot* as the
   KL reference.
3. **KD** — supervised distillation (Hinton logit KD) from `roberta-large-mnli`.
4. **GRPO+distill** — the *altered* GRPO: same algorithm, but (a) the KL /
   reward-penalty reference model is `roberta-large-mnli`, and (b) an auxiliary
   logit-KD loss from `roberta-large-mnli` is added.

All runs log to **W&B** `TAU-Frugal/distillation-rl`.

---

## 1. Decisions locked in (from our Q&A)

| Topic | Decision |
|---|---|
| Student | `FacebookAI/roberta-base` + fresh 3-way NLI head |
| KD teacher | `FacebookAI/roberta-large-mnli` (frozen), logits permuted to canonical order |
| GRPO reference (altered) | `FacebookAI/roberta-large-mnli` (your revised choice) |
| Dataset | `SetFit/mnli` |
| GRPO formulation | **Single-step bandit** (custom loop mirroring TRL's GRPO loss math) |
| Distillation type | **Logit/response KD** (Hinton KL over the 3 NLI classes) — no hidden-state projection |
| Experiment tracking | **W&B** only (no MLflow), entity `TAU-Frugal`, project `distillation-rl` |
| Config framework | **Hydra** (config groups + a single entrypoint) |
| Env / runner | **uv** project + **SLURM** (`gpu-best` / `tau` partitions) |

### Canonical label order
`0 = entailment, 1 = neutral, 2 = contradiction` (matches `SetFit/mnli`).
`roberta-large-mnli` is natively `contradiction/neutral/entailment`, so its logits
are permuted to canonical order (same trick as the team's `distill_nli`).

### `roberta-large` (non-MNLI) — DROPPED (resolved)
Per your decision, plain `FacebookAI/roberta-large` is **not used anywhere**. Both
the KD teacher and the GRPO reference are `roberta-large-mnli`. Only two
checkpoints are ever loaded: `roberta-base` (student) and `roberta-large-mnli`
(teacher / reference).

### How the student head works (verified on the real checkpoints)
`roberta-base` ships as `encoder + lm_head`, where `lm_head` is the
masked-LM head mapping each 768-dim token state to **vocab** logits
(`Linear 768→768 → LayerNorm → Linear 768→50265`, ≈39.2M params).
Loading via `AutoModelForSequenceClassification.from_pretrained(..., num_labels=3)`
**drops the MLM head** and attaches a fresh, randomly-initialized
`RobertaClassificationHead` on the `<s>`/CLS token:
`Dropout → Linear 768→768 → tanh → Dropout → Linear 768→3` (≈0.59M params); the
124M-param encoder is loaded intact. This is the **same recipe** that produced
`roberta-large-mnli` from `roberta-large` (just 1024-dim and already trained).
Both student and teacher therefore output exactly **3 logits → a 3-way softmax**;
the 768-vs-1024 gap is internal to each head and never surfaces, so logit-KD and
the GRPO KL-reference need **no projection**.

---

## 2. The four experiments — precise spec

Notation: input `x = (premise, hypothesis)`, gold label `y ∈ {0,1,2}`.
Student policy `π_θ(a|x) = softmax(f_θ(x))` over the 3 labels.

### (a) SFT
```
L = CE(f_θ(x), y)
```

### (c) KD (supervised distillation) — teacher `T = roberta-large-mnli`
Hinton KD (same as team's `distillation_loss`):
```
L = α · τ² · KL( softmax(f_θ(x)/τ) ‖ softmax(T(x)/τ) )  +  (1−α) · CE(f_θ(x), y)
```
Defaults: `τ = 2.0`, `α = 0.5` (configurable).

### (b) GRPO (single-step bandit) — reference `π_ref = frozen student snapshot`
For each `x`, sample a group of `G` actions `a_1..a_G ~ π_θ_old(·|x)`:
```
r_i  = 1 if a_i == y else 0                       # reward = correctness (configurable shaping)
Â_i  = (r_i − mean_j r_j) / (std_j r_j + ε)        # group-relative advantage
ρ_i  = π_θ(a_i|x) / π_θ_old(a_i|x)                 # importance ratio
L_pg = −(1/G) Σ_i min( ρ_i Â_i, clip(ρ_i, 1±ε_clip) Â_i )
L    = L_pg + β · D_KL( π_θ(·|x) ‖ π_ref(·|x) )
```
- `π_θ_old` = policy snapshot at rollout time (enables `μ>1` inner epochs; with
  `μ=1`, `ρ≡1` and this reduces to REINFORCE-with-group-baseline + KL — exactly
  TRL's default behaviour).
- KL is computed **exactly** over the 3-way categorical (lower variance than
  TRL's k3 sample estimator, which only makes sense for large vocabularies). A
  config flag `grpo.kl_estimator=exact|k3` lets us match TRL's estimator if you
  prefer. Note `r̃_i = r_i − β·log(π_θ/π_ref)` (reward-penalty form) is
  mathematically equivalent to the KL-in-loss form; we implement KL-in-loss
  (TRL default) and document the equivalence.
- Defaults: `G = 8`, `ε_clip = 0.2`, `β = 0.04`, `μ = 1`, `scale_rewards=true`.

### (d) GRPO + distill (the *altered* GRPO)
Same as (b) **except**:
- `π_ref = roberta-large-mnli` (the teacher), permuted to canonical order — used
  in **both** the KL term and (equivalently) the reward penalty.
- Add the Hinton **KD auxiliary loss** from the same teacher:
```
L = L_pg + β · D_KL( π_θ ‖ π_ref^{large-mnli} ) + λ_kd · L_KD
```
Defaults: `λ_kd = 1.0`, KD aux = soft-KL term (hard-CE optional, off by default
since reward already supplies the correctness signal).

> Because the reference and the KD teacher are the same model, `roberta-large-mnli`
> is loaded **once** and reused for both roles in experiment (d).

---

## 3. Repository layout (everything under `distill_rl/`, nothing outside)

```
distill_rl/
├── PLAN.md                      # this file
├── README.md                    # how to set up + run
├── pyproject.toml               # uv project metadata + deps
├── uv.lock                      # generated by `uv lock`
├── .python-version              # 3.13 (matches the team's stack)
├── .gitignore                   # .env, wandb/, outputs/, multirun/, .venv, __pycache__, *.pt …
├── .env.example                 # template: WANDB_API_KEY=...  (real .env is gitignored)
│
├── distill_rl/                  # importable package
│   ├── __init__.py
│   ├── data/
│   │   └── mnli.py              # SetFit/mnli load, tokenize, collate, dataloaders, CANONICAL_LABEL_ORDER
│   ├── models/
│   │   ├── student.py           # roberta-base + 3-way head (canonical order)
│   │   ├── teacher.py           # roberta-large-mnli, frozen, label-permuted (KD teacher + GRPO ref)
│   │   └── reference.py         # GRPO reference: frozen student snapshot OR teacher wrapper
│   ├── distillation/
│   │   └── losses.py            # Hinton logit KD (soft KL + optional hard CE)
│   ├── grpo/
│   │   ├── sampling.py          # sample G actions, gather logprobs (current/old/ref)
│   │   ├── advantage.py         # group-relative advantage
│   │   └── loss.py              # GRPO loss: PPO-clip + exact/k3 KL  (mirrors TRL math)
│   ├── training/
│   │   ├── supervised.py        # SFT + KD loop (experiments a, c)
│   │   ├── grpo_loop.py         # GRPO loop (experiments b, d)
│   │   └── evaluate.py          # accuracy / per-class metrics on val (+ optional test)
│   └── utils/
│       ├── seed.py              # seed_everything
│       ├── device.py            # get_device (cuda→cpu)
│       ├── wandb_logger.py      # init/log/finish helpers (reads entity+project from cfg)
│       └── config.py            # OmegaConf → dict helpers, flatten for logging
│
├── hydra_script/
│   ├── train.py                 # SINGLE entrypoint; dispatches on cfg.method
│   └── configs/
│       ├── config.yaml          # top-level defaults
│       ├── method/
│       │   ├── sft.yaml
│       │   ├── kd.yaml
│       │   ├── grpo.yaml
│       │   └── grpo_distill.yaml
│       ├── data/mnli.yaml
│       ├── student/roberta_base.yaml
│       ├── teacher/roberta_large_mnli.yaml
│       ├── grpo/default.yaml
│       ├── distill/hinton.yaml
│       ├── optimizer/adamw.yaml
│       ├── wandb/default.yaml
│       └── hydra/launcher/margaret_submitit.yaml
│
├── slurm/
│   ├── run_one.slurm            # run a single method (arg = method name)
│   └── run_all.slurm            # SLURM array 0–3 → the four experiments, one GPU each
│
└── tests/                       # lightweight CPU tests (label perm, KD loss, GRPO advantage/KL, sampling)
```

**Single entrypoint, method dispatch.** `hydra_script/train.py` reads
`cfg.method.algo` (`supervised` | `grpo`) plus toggles (`distill.enabled`,
`grpo.reference`) and calls the right loop. The four `method/*.yaml` presets are
thin files that compose the toggles, so the four experiments are just:
```
uv run python hydra_script/train.py method=sft
uv run python hydra_script/train.py method=kd
uv run python hydra_script/train.py method=grpo
uv run python hydra_script/train.py method=grpo_distill
```

---

## 4. Config sketch (Hydra)

`configs/config.yaml` (defaults list + shared knobs):
```yaml
defaults:
  - _self_
  - method: sft               # sft | kd | grpo | grpo_distill
  - data: mnli
  - student: roberta_base
  - teacher: roberta_large_mnli
  - optimizer: adamw
  - distill: hinton
  - grpo: default
  - wandb: default

seed: 42
epochs: 3
eval:
  every_n_steps: 500
run_name: ${method.name}_seed${seed}
hydra:
  run:
    dir: ${logging.out_dir}/${run_name}/${now:%Y-%m-%d}_${now:%H-%M-%S}
logging:
  out_dir: /scratch/elsaad/distill_rl_runs   # off the home quota, like the team's setup
```

`configs/method/grpo_distill.yaml`:
```yaml
name: grpo_distill
algo: grpo
distill: {enabled: true}
grpo:  {reference: teacher}     # roberta-large-mnli as KL/penalty reference
```

`configs/grpo/default.yaml`:
```yaml
group_size: 8
beta: 0.04
clip_eps: 0.2
inner_epochs: 1            # μ
kl_estimator: exact        # exact | k3
scale_rewards: true
reward: correctness        # +1 correct / 0 wrong (shaping configurable later)
reference: student         # student (frozen snapshot) | teacher (roberta-large-mnli)
```

`configs/wandb/default.yaml`:
```yaml
enabled: true
entity: TAU-Frugal
project: distillation-rl
mode: online               # online | offline | disabled
group: ${method.name}
tags: [mnli, roberta-base, ${method.name}]
```

(Hyperparameter defaults above are sensible starting points — flag any you want
changed.)

---

## 5. W&B logging & secret handling

- **Auth:** I will `wandb login` with the API key you gave me. The key is written
  only to `~/.netrc` (outside the repo) and/or a **gitignored `.env`**. It will
  **never** be committed. The repo ships a `.env.example` placeholder instead.
  SLURM scripts will `export WANDB_API_KEY` from your environment/`.env`.
- **What gets logged:** per-step train losses (total, policy, KL, KD, CE),
  reward mean/std, advantage stats, mean correctness, LR, grad-norm; per-eval
  val accuracy (+ per-class), best checkpoint metric; full resolved Hydra config
  as run config; run grouped by method so the 4 experiments overlay cleanly.
- One W&B run per training run; `run_name = <method>_seed<seed>`.

---

## 6. SLURM / GPU

From `sinfo`: `gpu-best` (26 nodes, rtx/volta/ampere), `tau` (team partition,
ampere), `gpu` (volta/ampere). These experiments need **no fp64**, so any GPU is
fine and a single 16 GB card comfortably holds `roberta-base` student +
`roberta-large-mnli` teacher (teacher is frozen / eval-only).

- `slurm/run_all.slurm`: `--array=0-3`, `--partition=gpu-best`, `--gres=gpu:1`,
  `--mem=24G`, maps array index → `{sft, kd, grpo, grpo_distill}`.
- `slurm/run_one.slurm`: single method via `$1`.
- Optional `--exclude` knob for the oldest cards; not required here.
- Hydra submitit launcher config provided for `--multirun` sweeps too.

---

## 7. Dependencies (uv project)

`pyproject.toml` deps: `torch`, `transformers`, `datasets`, `accelerate`,
`evaluate`, `hydra-core>=1.3.2`, `hydra-submitit-launcher`, `wandb`,
`trl` (kept as the reference for GRPO loss math; our loop is self-contained),
`numpy`, `tqdm`, `python-dotenv`. Dev group: `pytest`, `ruff`.
Python 3.13, CUDA build of torch (matches the team's `experimental_grow` stack).
I'll generate `uv.lock` with `uv lock` and verify `uv sync` works before any run.

---

## 8. Data pipeline

Reuse the team's proven approach (adapted, written fresh in `distill_rl`):
load `SetFit/mnli` (`text1`=premise, `text2`=hypothesis, `label`), tokenize
pairs (`max_seq_len=128`), dynamic-pad collator, train/val DataLoaders.
`train_limit` / `val_limit` knobs for fast smoke runs. Eval on the `validation`
split; `test` split optional at the end.

---

## 9. Build order (once you approve)

1. uv project scaffold (`pyproject.toml`, `.python-version`, `.gitignore`,
   `.env.example`) → `uv lock && uv sync`; `wandb login`.
2. `data/mnli.py` + a tiny CPU test (label order, collate).
3. `models/student.py`, `models/teacher.py` (+ permutation test).
4. `distillation/losses.py` (+ KD loss test).
5. `grpo/` (sampling, advantage, loss) (+ advantage/KL unit tests).
6. `training/` loops + `evaluate.py`.
7. `hydra_script/train.py` + all configs.
8. Smoke-run all 4 methods locally with `train_limit` small (CPU/1 GPU), confirm
   W&B logging works.
9. `slurm/` scripts; launch the 4-experiment array.
10. `README.md`.

I'll go module-by-module and pause for review at logical checkpoints rather than
dumping everything at once, if you prefer.

---

## 10. Constraints I will respect

- Write **only** inside `distill_rl/`; read-only everywhere else.
- **No git commits / pushes / branch changes** here. If anything needs
  committing or pushing, I'll ask you to do it.
- The W&B API key never enters a tracked file.

---

## 11. Things to confirm / your call

1. ~~`roberta-large` (non-MNLI)~~ — **RESOLVED: dropped entirely.**
2. ~~Hyperparameters~~ — **RESOLVED: approved as listed** (epochs 3, GRPO `G=8`,
   `β=0.04`, `clip_eps=0.2`, KD `τ=2.0`/`α=0.5`, `λ_kd=1.0`, `lr=2e-5`, `batch=16`).
3. **GRPO reward:** plain correctness (+1/0). Want any shaping (e.g. −1 for
   wrong, or confidence-margin reward)? *(I'll default to +1/0 unless told.)*
4. **KL estimator:** `exact` categorical (my default) vs TRL's `k3` sample
   estimator — any preference? *(I'll default to `exact`.)*
5. **Eval split:** `SetFit/mnli` `validation` for model selection, `test` at the
   end — good? *(I'll default to this.)*
6. **Entrypoint style:** single `train.py` + `method=` (my plan) vs four separate
   scripts — preference? *(I'll default to the single entrypoint.)*
```
