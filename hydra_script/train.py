"""Hydra entrypoint for all four experiments (one script, ``method=`` selects).

    uv run python hydra_script/train.py method=sft
    uv run python hydra_script/train.py method=kd
    uv run python hydra_script/train.py method=grpo
    uv run python hydra_script/train.py method=grpo_distill

``cfg.algo`` (supervised|grpo) plus the toggles ``cfg.distill.enabled`` /
``cfg.grpo.reference`` / ``cfg.grpo.kd_aux`` (all set by the method preset) decide
which loop runs and which models load. Metrics go to W&B (TAU-Frugal/distillation-rl).
"""

from __future__ import annotations

import logging
from pathlib import Path

import hydra
import torch
from dotenv import load_dotenv
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from torch.nn import functional as F
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup

from distill_rl.data.mnli import load_split, make_dataloader, maybe_subset, tokenize_split
from distill_rl.distillation.losses import distillation_loss
from distill_rl.models.reference import build_reference
from distill_rl.models.student import load_student
from distill_rl.models.teacher import load_teacher
from distill_rl.training.evaluate import evaluate
from distill_rl.training.grpo_loop import train_grpo
from distill_rl.training.supervised import train_supervised
from distill_rl.utils.device import get_device
from distill_rl.utils.seed import seed_everything
from distill_rl.utils.wandb_logger import WandbLogger

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def make_ce_loss():
    def _loss(model, batch):
        logits = model(input_ids=batch.input_ids, attention_mask=batch.attention_mask).logits
        loss = F.cross_entropy(logits, batch.labels)
        return loss, {"loss": float(loss), "ce": float(loss)}
    return _loss


def make_kd_loss(teacher, temperature, alpha):
    def _loss(model, batch):
        logits = model(input_ids=batch.input_ids, attention_mask=batch.attention_mask).logits
        with torch.no_grad():
            t_logits = teacher(batch.input_ids, batch.attention_mask)
        total, soft, hard = distillation_loss(
            logits, t_logits, batch.labels, temperature=temperature, alpha=alpha,
        )
        return total, {"loss": float(total), "soft": float(soft), "hard": float(hard)}
    return _loss


def _build_loader(data_cfg, tokenizer, split_key, *, shuffle, batch_size, limit):
    ds = maybe_subset(load_split(data_cfg, split_key), limit)
    ds_tok = tokenize_split(ds, tokenizer, data_cfg)
    return make_dataloader(
        ds_tok, tokenizer, batch_size=batch_size,
        shuffle=shuffle, num_workers=data_cfg["num_workers"],
    )


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    load_dotenv(PROJECT_ROOT / ".env")  # WANDB_API_KEY etc. (gitignored, never committed)
    log = logging.getLogger(__name__)

    seed_everything(int(cfg.seed))
    device = get_device(str(cfg.student.device))
    run_dir = Path(HydraConfig.get().runtime.output_dir)

    algo = str(cfg.algo)
    distill_on = bool(cfg.distill.enabled)
    grpo_ref = str(cfg.grpo.reference)
    grpo_kd_aux = bool(cfg.grpo.kd_aux)
    log.info(
        f"method={cfg.method_name} algo={algo} distill={distill_on} "
        f"grpo.reference={grpo_ref} grpo.kd_aux={grpo_kd_aux} device={device} run_dir={run_dir}"
    )

    logger = WandbLogger(
        OmegaConf.to_container(cfg.wandb, resolve=True),
        run_name=str(cfg.run_name),
        config=OmegaConf.to_container(cfg, resolve=True),
        out_dir=str(run_dir),
    )

    # ----- models -----
    student, tokenizer = load_student(OmegaConf.to_container(cfg.student, resolve=True), device=str(device))

    teacher_needed = distill_on or (algo == "grpo" and grpo_ref == "teacher") or grpo_kd_aux
    teacher = None
    if teacher_needed:
        teacher, _ = load_teacher(OmegaConf.to_container(cfg.teacher, resolve=True), device=str(device))

    # ----- data -----
    data_cfg = OmegaConf.to_container(cfg.data, resolve=True)
    train_loader = _build_loader(
        data_cfg, tokenizer, "train", shuffle=True,
        batch_size=data_cfg["train_batch_size"], limit=data_cfg.get("train_limit"),
    )
    val_loader = _build_loader(
        data_cfg, tokenizer, "val", shuffle=False,
        batch_size=data_cfg["eval_batch_size"], limit=data_cfg.get("val_limit"),
    )

    # ----- optimizer + linear-warmup scheduler -----
    grad_accum = int(cfg.grad_accum_steps)
    total_optim_steps = max(1, len(train_loader) // grad_accum) * int(cfg.epochs)
    optimizer = AdamW(
        [p for p in student.parameters() if p.requires_grad],
        lr=float(cfg.optimizer.lr),
        weight_decay=float(cfg.optimizer.weight_decay),
    )
    warmup_steps = int(float(cfg.optimizer.warmup_ratio) * total_optim_steps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_optim_steps,
    )

    common = dict(
        model=student if algo == "supervised" else None,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        epochs=int(cfg.epochs),
        grad_accum_steps=grad_accum,
        max_grad_norm=float(cfg.max_grad_norm),
        eval_every_optim_steps=int(cfg.eval.every_n_steps),
        log_every_optim_steps=int(cfg.log.every_n_steps),
        logger=logger,
        log=log,
        run_dir=run_dir,
    )

    # ----- dispatch -----
    if algo == "supervised":
        if distill_on:
            compute_loss = make_kd_loss(teacher, float(cfg.distill.temperature), float(cfg.distill.alpha))
        else:
            compute_loss = make_ce_loss()
        train_supervised(compute_loss=compute_loss, **common)
    elif algo == "grpo":
        reference = build_reference(grpo_ref, student=student, teacher=teacher)
        kd_teacher = teacher if grpo_kd_aux else None
        kd_cfg = {"temperature": float(cfg.distill.temperature), "lambda_kd": float(cfg.distill.lambda_kd)}
        common.pop("model")
        train_grpo(
            student=student,
            reference=reference,
            grpo_cfg=OmegaConf.to_container(cfg.grpo, resolve=True),
            kd_teacher=kd_teacher,
            kd_cfg=kd_cfg,
            **common,
        )
    else:
        raise ValueError(f"unknown algo {algo!r} (expected 'supervised' or 'grpo')")

    # ----- final eval on the best checkpoint -----
    # MNLI's `test` split is the GLUE-style unlabeled set (labels = -1), so we
    # report on `validation` (matched) -- the standard MNLI practice.
    best_ckpt = run_dir / "best.pt"
    if best_ckpt.exists():
        student.load_state_dict(torch.load(best_ckpt, map_location=device))
    final_metrics = {k.replace("val/", "final/"): v for k, v in evaluate(student, val_loader, device).items()}
    log.info("[final] " + " ".join(f"{k}={v:.4f}" for k, v in final_metrics.items()))
    logger.log(final_metrics)
    logger.set_summary(final_metrics)
    logger.finish()


if __name__ == "__main__":
    main()
