"""KDRL: on-policy reverse-KL distillation added to GRPO via a joint loss.

KDRL (Xu et al., arXiv:2506.02208) augments the GRPO objective with an auxiliary
reverse-KL (RKL) distillation term toward the teacher, estimated on the SAME
on-policy sampled actions used for the policy gradient (their Eq. 8 / Eq. 26-27):

    J_KDRL(theta) = J_GRPO(theta) - beta * D_KL(pi_theta || pi_teacher)

so the *minimized* loss is ``grpo_loss + beta * L_kd`` (``grpo_loss`` already being
``-J_GRPO``). The KD term is the student-anchored REVERSE KL -- not the
forward/Hinton KD used by ``grpo_distill`` -- which is why KDRL "incorporates KD
slightly more". With the per-action teacher log-ratio (teacher detached, policy
with grad)

    R_i(theta) = log pi_T(a_i|x) - log pi_theta(a_i|x)

the RKL is estimated by (paper Appendix A.3):

    k2     : mean[ 1/2 R^2 ]          unbiased GRADIENT  (the paper's default; = Table 3's "mse")
    k3     : mean[ exp(R) - R - 1 ]   unbiased value, biased gradient (= the repo's existing k3 math)
    exact  : closed-form KL over the 3 classes (Top-K with K>=3 collapses to this here)

Reward-guided masking (Section 3.4) optionally drops the KD term where the sampled
action was already correct (``response``) or where the whole group already contains
a correct sample (``group``) -- "selective imitation" of only the hard cases.

This module is pure tensor math (no models); the training loop wires the teacher
forward, the beta schedule, and the optimizer around ``kd_rkl_loss``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch

from distill_rl.grpo.loss import categorical_kl
from distill_rl.grpo.sampling import gather_logprobs


def kd_rkl_mask(correct: torch.Tensor, masking: str) -> torch.Tensor:
    """Per-action keep-mask (1 = apply KD, 0 = skip) for reward-guided masking.

    ``correct`` (B, G) bool: whether each sampled action matched the gold label
    (the paper's binary outcome reward ``r_i`` -- masking keys on correctness, not
    the possibly-shaped numeric reward).

      none     -> all ones                         (plain KDRL)
      response -> 1[action incorrect]              (drop KD on already-correct samples; = "KDRL-Masking")
      group    -> 1[group has NO correct sample]   (KD only on fully-hard examples)
    """
    if masking == "none":
        return torch.ones_like(correct, dtype=torch.float)
    if masking == "response":
        return (~correct).float()
    if masking == "group":
        keep = ~correct.any(dim=1, keepdim=True)  # (B, 1): True iff no correct sample
        return keep.expand_as(correct).float()
    raise ValueError(f"unknown masking {masking!r} (expected 'none', 'response' or 'group')")


@dataclass
class KDRLMetrics:
    kd_rkl: float          # KD-RKL loss value (before the beta weight)
    kl_to_teacher: float   # exact KL(pi_theta || pi_T) diagnostic (estimator-independent)
    kd_mask_frac: float    # fraction of KD terms kept after masking

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean of ``values`` over entries kept by ``mask`` (0 when nothing is kept)."""
    return (values * mask).sum() / mask.sum().clamp(min=1.0)


def kd_rkl_loss(
    policy_logits: torch.Tensor,    # (B, C) with grad
    teacher_logits: torch.Tensor,   # (B, C) detached (teacher is frozen)
    actions: torch.Tensor,          # (B, G) long, the on-policy sampled actions
    correct: torch.Tensor,          # (B, G) bool, action == gold label
    *,
    estimator: str = "k2",
    masking: str = "none",
) -> tuple[torch.Tensor, KDRLMetrics]:
    """KD-RKL distillation loss ``D_KL(pi_theta || pi_T)`` and diagnostics.

    The returned loss is the *unweighted* KD term; the loop multiplies it by the
    (possibly annealed) coefficient ``beta`` before adding it to the GRPO loss.
    """
    teacher_logits = teacher_logits.detach()
    mask = kd_rkl_mask(correct, masking)  # (B, G) float

    if estimator in ("k2", "k3"):
        policy_logp = gather_logprobs(policy_logits, actions)    # (B, G) grad
        teacher_logp = gather_logprobs(teacher_logits, actions)  # (B, G) detached
        R = teacher_logp - policy_logp                           # (B, G), grad via -policy_logp
        if estimator == "k2":
            term = 0.5 * R.pow(2)                                # unbiased gradient (Eq. 21)
        else:  # k3
            term = torch.exp(R) - R - 1.0                        # Eq. 22-23
        loss = _masked_mean(term, mask)
    elif estimator == "exact":
        # Closed-form reverse KL per example; masking becomes a per-example weight
        # (response -> fraction of incorrect samples; group -> all-incorrect indicator),
        # since the exact KL has no per-action resolution.
        kl_b = categorical_kl(policy_logits, teacher_logits)     # (B,) grad
        w = mask.mean(dim=1)                                     # (B,) in [0, 1]
        loss = (w * kl_b).sum() / w.sum().clamp(min=1.0)
    else:
        raise ValueError(f"unknown estimator {estimator!r} (expected 'k2', 'k3' or 'exact')")

    with torch.no_grad():
        metrics = KDRLMetrics(
            kd_rkl=float(loss),
            kl_to_teacher=float(categorical_kl(policy_logits, teacher_logits).mean()),
            kd_mask_frac=float(mask.mean()),
        )
    return loss, metrics


def anneal_beta(step: int, *, beta_init: float, beta_min: float, delta: float) -> float:
    """Linearly decayed KD coefficient ``beta = max(beta_init - delta*step, beta_min)`` (Section 3.3)."""
    return max(float(beta_init) - float(delta) * int(step), float(beta_min))


def resolve_anneal_delta(
    *,
    beta_init: float,
    beta_min: float,
    anneal_frac: float,
    total_steps: int,
    delta: float | None = None,
) -> float:
    """Decay rate ``delta`` for the beta anneal.

    By default ``beta`` reaches ``beta_min`` after ``anneal_frac * total_steps``
    optimizer steps -- an adaptation of the paper's raw ``delta=5e-5`` (calibrated to
    ~280 LLM steps) to this project's far larger MNLI step count. An explicit positive
    ``delta`` overrides the auto-computed value.
    """
    if delta is not None and float(delta) > 0:
        return float(delta)
    horizon = max(1.0, float(anneal_frac) * float(total_steps))
    return max(0.0, (float(beta_init) - float(beta_min)) / horizon)
