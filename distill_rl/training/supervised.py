"""Supervised training loop, shared by SFT and KD.

``compute_loss(model, batch) -> (loss, metrics)`` is the only thing that differs
between the two: plain cross-entropy for SFT, Hinton KD for KD. Everything else
(grad accumulation, clipping, scheduling, periodic eval, best-checkpoint, W&B
logging) is identical.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable

import torch

from distill_rl.data.mnli import MNLIBatch
from distill_rl.training.evaluate import evaluate
from distill_rl.utils.wandb_logger import WandbLogger

LossFn = Callable[[torch.nn.Module, MNLIBatch], tuple[torch.Tensor, dict[str, float]]]


def train_supervised(
    *,
    model: torch.nn.Module,
    train_loader,
    val_loader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    compute_loss: LossFn,
    device: torch.device | str,
    epochs: int,
    grad_accum_steps: int,
    max_grad_norm: float,
    eval_every_optim_steps: int,
    log_every_optim_steps: int,
    logger: WandbLogger,
    log: logging.Logger,
    run_dir: Path,
    save_best: bool = True,
) -> dict[str, float]:
    model.train()
    optim_step = 0
    best_acc = -1.0
    best_metrics: dict[str, float] = {}

    for epoch in range(epochs):
        optimizer.zero_grad()
        for i, batch in enumerate(train_loader):
            batch = batch.to(device)
            loss, metrics = compute_loss(model, batch)
            (loss / grad_accum_steps).backward()

            if (i + 1) % grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
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
                        model, val_loader, device, optim_step, logger, log,
                        run_dir, best_acc, best_metrics, save_best,
                    )

        # end-of-epoch eval (guarantees at least one eval even with large eval_every)
        best_acc, best_metrics = _eval_and_track(
            model, val_loader, device, optim_step, logger, log,
            run_dir, best_acc, best_metrics, save_best,
        )

    logger.set_summary({f"best/{k.split('/')[-1]}": v for k, v in best_metrics.items()})
    return best_metrics


def _eval_and_track(model, val_loader, device, step, logger, log, run_dir,
                    best_acc, best_metrics, save_best):
    val = evaluate(model, val_loader, device)
    logger.log(val, step=step)
    log.info(f"[eval @ step {step}] " + " ".join(f"{k}={v:.4f}" for k, v in val.items()))
    if val["val/accuracy"] > best_acc:
        best_acc = val["val/accuracy"]
        best_metrics = dict(val)
        if save_best:
            torch.save(model.state_dict(), Path(run_dir) / "best.pt")
    return best_acc, best_metrics
