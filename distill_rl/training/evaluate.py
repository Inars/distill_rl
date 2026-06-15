"""Validation: overall + per-class accuracy and cross-entropy on a loader.

Always uses greedy argmax (deterministic), so the metric is comparable across
SFT / KD / GRPO regardless of how each was trained.
"""

from __future__ import annotations

import torch
from torch.nn import functional as F

from distill_rl.data.mnli import CANONICAL_LABEL_ORDER, MNLIBatch


@torch.no_grad()
def evaluate(model: torch.nn.Module, loader, device: torch.device | str) -> dict[str, float]:
    was_training = model.training
    model.eval()

    n_classes = len(CANONICAL_LABEL_ORDER)
    correct = 0
    loss_sum = 0.0
    n = 0
    per_class_correct = [0] * n_classes
    per_class_total = [0] * n_classes

    for batch in loader:
        batch: MNLIBatch = batch.to(device)
        logits = model(input_ids=batch.input_ids, attention_mask=batch.attention_mask).logits
        loss_sum += float(F.cross_entropy(logits, batch.labels, reduction="sum"))
        preds = logits.argmax(dim=-1)
        correct += int((preds == batch.labels).sum())
        n += len(batch)
        for c in range(n_classes):
            mask = batch.labels == c
            per_class_total[c] += int(mask.sum())
            per_class_correct[c] += int((preds[mask] == c).sum())

    if was_training:
        model.train()

    metrics = {
        "val/accuracy": correct / max(n, 1),
        "val/loss": loss_sum / max(n, 1),
    }
    for c, name in enumerate(CANONICAL_LABEL_ORDER):
        metrics[f"val/acc_{name}"] = per_class_correct[c] / max(per_class_total[c], 1)
    return metrics
