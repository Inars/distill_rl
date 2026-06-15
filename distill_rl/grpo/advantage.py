"""Rewards and group-relative advantages (the 'GR' in GRPO)."""

from __future__ import annotations

import torch


def correctness_reward(
    actions: torch.Tensor,
    labels: torch.Tensor,
    *,
    correct: float = 1.0,
    wrong: float = 0.0,
) -> torch.Tensor:
    """Reward = ``correct`` if the sampled label matches gold else ``wrong``.

    ``actions`` (B, G), ``labels`` (B,) -> (B, G) float.
    """
    match = (actions == labels.unsqueeze(1)).float()
    return correct * match + wrong * (1.0 - match)


def group_relative_advantage(
    rewards: torch.Tensor,
    *,
    scale_rewards: bool = True,
    eps: float = 1e-4,
) -> torch.Tensor:
    """Center each example's rewards by its group mean; optionally divide by group std.

    ``rewards`` (B, G) -> advantages (B, G). When ``scale_rewards`` is True this is
    TRL's default ``(r - mean)/(std + eps)``; otherwise just the mean-subtracted
    baseline. Groups with zero variance (all-correct or all-wrong) yield ~0
    advantage and thus no gradient, as expected.
    """
    mean = rewards.mean(dim=1, keepdim=True)
    adv = rewards - mean
    if scale_rewards:
        std = rewards.std(dim=1, keepdim=True)
        adv = adv / (std + eps)
    return adv
