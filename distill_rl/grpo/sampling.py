"""Action sampling and log-probabilities for the categorical (3-class) policy."""

from __future__ import annotations

import torch
from torch.nn import functional as F


def sample_actions(
    logits: torch.Tensor,
    group_size: int,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Sample ``group_size`` actions per example from ``Categorical(softmax(logits))``.

    ``logits``: (B, C). Returns long tensor (B, G). Sampling is with replacement
    (the 3-class support is small, so duplicates within a group are expected and
    fine -- the group baseline still works).
    """
    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=group_size, replacement=True, generator=generator)


def gather_logprobs(logits: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
    """Log-prob of each sampled action. ``logits`` (B, C), ``actions`` (B, G) -> (B, G)."""
    logp = F.log_softmax(logits, dim=-1)
    return torch.gather(logp, dim=-1, index=actions)


def policy_entropy(logits: torch.Tensor) -> torch.Tensor:
    """Per-example entropy of the policy (nats). ``logits`` (B, C) -> (B,)."""
    logp = F.log_softmax(logits, dim=-1)
    return -(logp.exp() * logp).sum(dim=-1)
