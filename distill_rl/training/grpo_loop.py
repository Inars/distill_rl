"""GRPO-family training loop: plain GRPO, GRPO+distill, KDRL, and TRRD.

Per batch (mu=1, TRL's default): one student forward, sample a group of actions from
the (detached) policy, reward correctness, form group-relative advantages, and take a
PPO-clipped step. Two default-off switches extend the base GRPO objective:

* ``objective="trrd"`` swaps the importance ratio for the teacher (+) old-policy
  mixture-anchored TRRD ratio (Zhang et al., 2602.22495). ``objective="grpo"`` (default)
  is the original behaviour.
* ``kd_rkl_cfg`` adds the KDRL on-policy reverse-KL distillation term toward the teacher
  (Xu et al., 2506.02208), with its own (optionally annealed) coefficient and reward-guided
  masking.

The existing Hinton soft-KD aux (``kd_aux``) path is unchanged. The teacher is forwarded
at most once per step and reused for every teacher-based role (ratio / KD-RKL / Hinton /
the KL reference when ``reference`` is the teacher).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch

from distill_rl.distillation.losses import soft_kd_loss
from distill_rl.grpo.advantage import correctness_reward, group_relative_advantage
from distill_rl.grpo.kdrl import anneal_beta, kd_rkl_loss
from distill_rl.grpo.loss import grpo_loss
from distill_rl.grpo.sampling import gather_logprobs, policy_entropy, sample_actions
from distill_rl.grpo.trrd import trrd_loss
from distill_rl.models.reference import Reference, TeacherReference
from distill_rl.models.teacher import FrozenTeacher
from distill_rl.training.supervised import _eval_and_track
from distill_rl.utils.wandb_logger import WandbLogger


def _kd_rkl_beta(cfg: dict[str, Any], step: int) -> float:
    """Current KD-RKL coefficient: constant, or linearly annealed by optimizer step."""
    if str(cfg.get("schedule", "constant")) == "anneal":
        return anneal_beta(
            step,
            beta_init=float(cfg["beta_init"]),
            beta_min=float(cfg["beta_min"]),
            delta=float(cfg["delta"]),
        )
    return float(cfg.get("beta", 2e-3))


def train_grpo(
    *,
    student: torch.nn.Module,
    reference: Reference | None,
    train_loader,
    val_loader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    device: torch.device | str,
    epochs: int,
    grad_accum_steps: int,
    max_grad_norm: float,
    eval_every_optim_steps: int,
    log_every_optim_steps: int,
    grpo_cfg: dict[str, Any],
    logger: WandbLogger,
    log: logging.Logger,
    run_dir: Path,
    teacher: FrozenTeacher | None = None,
    kd_aux: bool = False,
    kd_cfg: dict[str, Any] | None = None,
    objective: str = "grpo",
    trrd_cfg: dict[str, Any] | None = None,
    kd_rkl_cfg: dict[str, Any] | None = None,
    generator: torch.Generator | None = None,
    save_best: bool = True,
) -> dict[str, float]:
    G = int(grpo_cfg["group_size"])
    beta = float(grpo_cfg["beta"])
    clip_eps = float(grpo_cfg["clip_eps"])
    kl_estimator = str(grpo_cfg.get("kl_estimator", "exact"))
    scale_rewards = bool(grpo_cfg.get("scale_rewards", True))
    r_correct = float(grpo_cfg.get("reward_correct", 1.0))
    r_wrong = float(grpo_cfg.get("reward_wrong", 0.0))

    if objective not in ("grpo", "trrd"):
        raise ValueError(f"unknown objective {objective!r} (expected 'grpo' or 'trrd')")

    kd_aux_on = kd_aux and teacher is not None
    kd_temperature = float((kd_cfg or {}).get("temperature", 2.0))
    lambda_kd = float((kd_cfg or {}).get("lambda_kd", 1.0))

    kdrl_on = kd_rkl_cfg is not None
    if kdrl_on:
        if teacher is None:
            raise ValueError("kd_rkl_cfg set but no teacher model provided")
        kd_estimator = str(kd_rkl_cfg.get("estimator", "k2"))
        kd_masking = str(kd_rkl_cfg.get("masking", "none"))

    if objective == "trrd":
        if teacher is None:
            raise ValueError("objective='trrd' requires a teacher model")
        alpha = float((trrd_cfg or {}).get("alpha", 0.5))
        distill_clip = float((trrd_cfg or {}).get("distill_clip", 1.0))

    ref_is_teacher = isinstance(reference, TeacherReference)
    need_teacher = teacher is not None and (
        objective == "trrd" or kdrl_on or kd_aux_on or ref_is_teacher
    )

    student.train()
    optim_step = 0
    best_acc = -1.0
    best_metrics: dict[str, float] = {}

    for epoch in range(epochs):
        optimizer.zero_grad()
        for i, batch in enumerate(train_loader):
            batch = batch.to(device)
            ids, mask, labels = batch.input_ids, batch.attention_mask, batch.labels

            logits = student(input_ids=ids, attention_mask=mask).logits  # (B, 3), grad

            with torch.no_grad():
                old_logits = logits.detach()
                actions = sample_actions(old_logits, G, generator)        # (B, G)
                old_logprobs = gather_logprobs(old_logits, actions)       # (B, G)
                rewards = correctness_reward(
                    actions, labels, correct=r_correct, wrong=r_wrong
                )                                                         # (B, G)
                advantages = group_relative_advantage(
                    rewards, scale_rewards=scale_rewards
                )                                                         # (B, G)
                correct = actions == labels.unsqueeze(1)                  # (B, G) bool

                # Teacher forwarded at most once; reused for ratio / KD / reference roles.
                teacher_logits = teacher(ids, mask) if need_teacher else None
                if beta > 0.0 and reference is not None:
                    ref_logits = teacher_logits if ref_is_teacher else reference.logits(ids, mask)
                else:
                    ref_logits = old_logits  # dummy: the ref-KL weight beta is 0

            # ----- policy objective -----
            if objective == "grpo":
                loss, pm = grpo_loss(
                    logits, old_logprobs, actions, advantages, ref_logits,
                    beta=beta, clip_eps=clip_eps, kl_estimator=kl_estimator,
                )
            else:  # trrd
                teacher_logprobs = gather_logprobs(teacher_logits, actions)  # (B, G)
                loss, pm = trrd_loss(
                    logits, old_logprobs, teacher_logprobs, actions, advantages, ref_logits,
                    alpha=alpha, distill_clip=distill_clip,
                    beta=beta, clip_eps=clip_eps, kl_estimator=kl_estimator,
                )
            metrics = pm.as_dict()

            # ----- distillation terms (mutually exclusive across methods in practice) -----
            if kd_aux_on:
                # In grpo_distill the KD teacher IS the reference, so teacher_logits already
                # are its canonical logits (reused above for ref_logits) -- no extra forward.
                kd = soft_kd_loss(logits, teacher_logits, temperature=kd_temperature)
                loss = loss + lambda_kd * kd
                metrics["kd"] = float(kd.detach())

            if kdrl_on:
                beta_kd = _kd_rkl_beta(kd_rkl_cfg, optim_step)
                kd_rkl, km = kd_rkl_loss(
                    logits, teacher_logits, actions, correct,
                    estimator=kd_estimator, masking=kd_masking,
                )
                loss = loss + beta_kd * kd_rkl
                metrics.update(km.as_dict())
                metrics["beta_kd"] = beta_kd

            if kd_aux_on or kdrl_on:
                metrics["loss"] = float(loss.detach())

            with torch.no_grad():
                metrics["reward_mean"] = float(rewards.mean())
                metrics["reward_std"] = float(rewards.std())
                metrics["advantage_abs_mean"] = float(advantages.abs().mean())
                metrics["entropy"] = float(policy_entropy(logits).mean())
                metrics["policy_acc"] = float((logits.argmax(-1) == labels).float().mean())

            (loss / grad_accum_steps).backward()

            if (i + 1) % grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(student.parameters(), max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                optim_step += 1

                if optim_step % log_every_optim_steps == 0:
                    payload = {f"train/{k}": v for k, v in metrics.items()}
                    payload["train/lr"] = scheduler.get_last_lr()[0]
                    payload["train/epoch"] = epoch
                    logger.log(payload, step=optim_step)

                if eval_every_optim_steps > 0 and optim_step % eval_every_optim_steps == 0:
                    best_acc, best_metrics = _eval_and_track(
                        student, val_loader, device, optim_step, logger, log,
                        run_dir, best_acc, best_metrics, save_best,
                    )

        best_acc, best_metrics = _eval_and_track(
            student, val_loader, device, optim_step, logger, log,
            run_dir, best_acc, best_metrics, save_best,
        )

    logger.set_summary({f"best/{k.split('/')[-1]}": v for k, v in best_metrics.items()})
    return best_metrics
