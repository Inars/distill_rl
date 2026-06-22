"""TRRD: Trust Region Ratio Distillation (RLAD, Zhang et al., arXiv:2602.22495).

Instead of adding a KL distillation *regularizer*, TRRD edits the GRPO importance
**ratio** so its trust region is anchored on a geometric mixture of the old student
policy and the teacher (their Eq. 3):

    r_TRRD_i(theta) = ( pi_theta(a_i)/pi_old(a_i) )^alpha * ( pi_theta(a_i)/pi_T(a_i) )^(1-alpha)
                    = pi_theta(a_i) / [ pi_old(a_i)^alpha * pi_T(a_i)^(1-alpha) ]

Then the ordinary GRPO/PPO clipped objective is applied with ``r_TRRD`` in place of
the usual ratio (Eq. 4). ``alpha=1`` recovers standard GRPO; ``alpha=0`` is a
teacher-anchored, DPO-like update (Appendix C). There is **no separate KD term** --
distillation is folded into the trust region, so teacher influence only acts when the
advantage supports it (selective imitation).

In log space (Eq. 5), gathering all three log-probs on the sampled actions:

    trust_i   = log pi_theta(a_i) - log pi_old(a_i)                  # old detached
    distill_i = clamp( log pi_theta(a_i) - log pi_T(a_i), -c, +c )   # teacher detached; Appendix B
    log r_TRRD_i = alpha * trust_i + (1 - alpha) * distill_i

The ``distill_clip`` ``c`` (Appendix B) clamps the teacher-student log-ratio to keep it
sane early on, when teacher and student are far apart; the standard PPO clip on
``r_TRRD`` is applied on top.

This module is pure tensor math; the loop supplies the teacher log-probs on the
sampled actions and the (optional) reference-KL term, reusing the GRPO scaffolding.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch

from distill_rl.grpo.loss import categorical_kl
from distill_rl.grpo.sampling import gather_logprobs


@dataclass
class TRRDMetrics:
    policy_loss: float
    kl: float                # reference KL(pi_theta || pi_ref), 0 when ref_beta == 0
    loss: float
    ratio_mean: float        # mean r_TRRD
    ratio_max: float
    frac_clipped: float      # fraction where the PPO clip binds
    distill_clip_frac: float  # fraction where the Appendix-B teacher log-ratio clamp binds

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


def trrd_log_ratio(
    policy_logprobs: torch.Tensor,   # (B, G) with grad: log pi_theta(a)
    old_logprobs: torch.Tensor,      # (B, G) detached: log pi_old(a)
    teacher_logprobs: torch.Tensor,  # (B, G) detached: log pi_T(a)
    *,
    alpha: float,
    distill_clip: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(log r_TRRD, clamp_active)`` for the mixture-anchored ratio.

    ``clamp_active`` (bool, B x G) flags where the Appendix-B teacher log-ratio clamp binds.
    """
    trust = policy_logprobs - old_logprobs                  # mu=1 -> value 0, grad = grad log pi
    distill_raw = policy_logprobs - teacher_logprobs        # log(pi_theta / pi_T)
    distill = torch.clamp(distill_raw, -distill_clip, distill_clip)
    log_ratio = alpha * trust + (1.0 - alpha) * distill
    clamp_active = distill_raw.detach().abs() > distill_clip
    return log_ratio, clamp_active


def trrd_loss(
    policy_logits: torch.Tensor,      # (B, C) with grad
    old_logprobs: torch.Tensor,       # (B, G) detached
    teacher_logprobs: torch.Tensor,   # (B, G) detached (teacher log-probs on the sampled actions)
    actions: torch.Tensor,            # (B, G) long
    advantages: torch.Tensor,         # (B, G) detached
    ref_logits: torch.Tensor,         # (B, C) detached -- reference for the KL term
    *,
    alpha: float,
    distill_clip: float,
    beta: float,
    clip_eps: float,
    kl_estimator: str = "exact",
) -> tuple[torch.Tensor, TRRDMetrics]:
    """GRPO clipped objective with the TRRD mixture-anchored ratio, plus optional ref-KL.

    Mirrors ``grpo.loss.grpo_loss`` but swaps the importance ratio for ``r_TRRD``.
    """
    policy_logprobs = gather_logprobs(policy_logits, actions)  # (B, G), grad
    log_ratio, clamp_active = trrd_log_ratio(
        policy_logprobs, old_logprobs, teacher_logprobs,
        alpha=alpha, distill_clip=distill_clip,
    )
    ratio = torch.exp(log_ratio)  # (B, G)

    unclipped = ratio * advantages
    clipped = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantages
    per_sample = torch.min(unclipped, clipped)
    policy_loss = -per_sample.mean()

    # Optional reference KL (kept for TRRD, per the paper's Eq. 4). Same estimators as GRPO.
    if beta == 0.0:
        kl = torch.zeros((), device=policy_logits.device)
    elif kl_estimator == "exact":
        kl = categorical_kl(policy_logits, ref_logits).mean()
    elif kl_estimator == "k3":
        ref_logprobs = gather_logprobs(ref_logits, actions)
        diff = ref_logprobs - policy_logprobs
        kl = (torch.exp(diff) - diff - 1.0).mean()
    else:
        raise ValueError(f"unknown kl_estimator {kl_estimator!r} (expected 'exact' or 'k3')")

    loss = policy_loss + beta * kl

    with torch.no_grad():
        metrics = TRRDMetrics(
            policy_loss=float(policy_loss),
            kl=float(kl),
            loss=float(loss),
            ratio_mean=float(ratio.mean()),
            ratio_max=float(ratio.max()),
            frac_clipped=float((unclipped > clipped).float().mean()),
            distill_clip_frac=float(clamp_active.float().mean()),
        )
    return loss, metrics
