"""The GRPO objective: PPO-clipped policy loss + KL penalty to a reference.

For one batch of ``B`` examples each with a group of ``G`` sampled actions:

    ratio_i  = pi_theta(a_i|x) / pi_theta_old(a_i|x)
    L_pg     = - mean_i  min( ratio_i * A_i , clip(ratio_i, 1±eps) * A_i )
    L        = L_pg + beta * D_KL( pi_theta(.|x) || pi_ref(.|x) )

Notes
-----
* ``old_logprobs`` are the (detached) log-probs under the rollout policy. With a
  single inner update (mu=1, TRL's default) they equal ``policy_logprobs.detach()``
  so ``ratio == 1`` in value but its gradient is the REINFORCE term
  ``-A * grad(log pi)`` -- i.e. the loss VALUE may be ~0 while the GRADIENT is the
  correct policy gradient. Always compute ``ratio`` via ``exp`` (never hardcode 1).
* The reward-penalty form ``r_tilde = r - beta*log(pi/pi_ref)`` is equivalent in
  spirit; we use the TRL-faithful KL-in-loss form here.
* KL is over the 3-way categorical: ``exact`` computes it in closed form (lower
  variance, the natural choice for a tiny support); ``k3`` reproduces TRL's
  sample-based estimator on the drawn actions.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch.nn import functional as F

from distill_rl.grpo.sampling import gather_logprobs


@dataclass
class GRPOMetrics:
    policy_loss: float
    kl: float
    loss: float
    ratio_mean: float
    ratio_max: float
    frac_clipped: float

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


def categorical_kl(p_logits: torch.Tensor, q_logits: torch.Tensor) -> torch.Tensor:
    """Exact ``KL(softmax(p) || softmax(q))`` per row. (B, C),(B, C) -> (B,)."""
    p_logp = F.log_softmax(p_logits, dim=-1)
    q_logp = F.log_softmax(q_logits, dim=-1)
    return (p_logp.exp() * (p_logp - q_logp)).sum(dim=-1)


def grpo_loss(
    policy_logits: torch.Tensor,   # (B, C) with grad
    old_logprobs: torch.Tensor,    # (B, G) detached
    actions: torch.Tensor,         # (B, G) long
    advantages: torch.Tensor,      # (B, G) detached
    ref_logits: torch.Tensor,      # (B, C) detached
    *,
    beta: float,
    clip_eps: float,
    kl_estimator: str = "exact",
) -> tuple[torch.Tensor, GRPOMetrics]:
    """Return ``(loss, metrics)`` for one GRPO update on a batch."""
    policy_logprobs = gather_logprobs(policy_logits, actions)  # (B, G), grad
    ratio = torch.exp(policy_logprobs - old_logprobs)          # (B, G)

    unclipped = ratio * advantages
    clipped = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantages
    per_sample = torch.min(unclipped, clipped)                 # (B, G)
    policy_loss = -per_sample.mean()

    if kl_estimator == "exact":
        kl = categorical_kl(policy_logits, ref_logits).mean()
    elif kl_estimator == "k3":
        ref_logprobs = gather_logprobs(ref_logits, actions)    # (B, G), detached
        diff = ref_logprobs - policy_logprobs
        kl = (torch.exp(diff) - diff - 1.0).mean()
    else:
        raise ValueError(f"unknown kl_estimator {kl_estimator!r} (expected 'exact' or 'k3')")

    loss = policy_loss + beta * kl

    with torch.no_grad():
        clip_binding = (unclipped > clipped).float().mean()
        metrics = GRPOMetrics(
            policy_loss=float(policy_loss),
            kl=float(kl),
            loss=float(loss),
            ratio_mean=float(ratio.mean()),
            ratio_max=float(ratio.max()),
            frac_clipped=float(clip_binding),
        )
    return loss, metrics
