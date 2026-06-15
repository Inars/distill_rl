"""GRPO training loop (single-step bandit), for plain GRPO and altered GRPO+distill.

Per batch (mu=1, TRL's default): one student forward, sample a group of actions
from the (detached) policy, reward correctness, form group-relative advantages,
and take the PPO-clipped + KL-penalized GRPO step. The reference for the KL is
passed in (frozen student snapshot for plain GRPO; teacher for altered GRPO). When
``kd_teacher`` is given, the soft KD term is added as an auxiliary loss.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch

from distill_rl.distillation.losses import soft_kd_loss
from distill_rl.grpo.advantage import correctness_reward, group_relative_advantage
from distill_rl.grpo.loss import grpo_loss
from distill_rl.grpo.sampling import gather_logprobs, policy_entropy, sample_actions
from distill_rl.models.reference import Reference
from distill_rl.models.teacher import FrozenTeacher
from distill_rl.training.supervised import _eval_and_track
from distill_rl.utils.wandb_logger import WandbLogger


def train_grpo(
    *,
    student: torch.nn.Module,
    reference: Reference,
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
    kd_teacher: FrozenTeacher | None = None,
    kd_cfg: dict[str, Any] | None = None,
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

    kd_on = kd_teacher is not None
    kd_temperature = float((kd_cfg or {}).get("temperature", 2.0))
    lambda_kd = float((kd_cfg or {}).get("lambda_kd", 1.0))

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
                ref_logits = reference.logits(ids, mask)                  # (B, 3)
                rewards = correctness_reward(
                    actions, labels, correct=r_correct, wrong=r_wrong
                )                                                         # (B, G)
                advantages = group_relative_advantage(
                    rewards, scale_rewards=scale_rewards
                )                                                         # (B, G)

            loss, gm = grpo_loss(
                logits, old_logprobs, actions, advantages, ref_logits,
                beta=beta, clip_eps=clip_eps, kl_estimator=kl_estimator,
            )
            metrics = gm.as_dict()

            if kd_on:
                t_logits = kd_teacher(ids, mask)                          # (B, 3), no grad
                kd = soft_kd_loss(logits, t_logits, temperature=kd_temperature)
                loss = loss + lambda_kd * kd
                metrics["kd"] = float(kd)
                metrics["loss"] = float(loss)

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
