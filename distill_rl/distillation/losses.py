"""Logit / response knowledge-distillation (Hinton et al., 2015).

    L = alpha * T^2 * KL( softmax(s/T) || softmax(t/T) )  +  (1 - alpha) * CE(s, y)

The T^2 factor preserves gradient magnitude as T changes. Student and teacher
logits are both expected in the project's canonical label order
(entailment=0, neutral=1, contradiction=2) -- the teacher wrapper guarantees this.

Used in two places:
  * KD run: full Hinton loss (soft + hard) is the training objective.
  * GRPO+distill run: only the soft term is added as an auxiliary loss (the
    reward already supplies the hard-label signal), via ``soft_kd_loss``.
"""

from __future__ import annotations

import torch
from torch.nn import functional as F


def soft_kd_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    *,
    temperature: float,
) -> torch.Tensor:
    """Temperature-scaled soft KL term, ``T^2 * KL(softmax(s/T) || softmax(t/T))``."""
    if temperature <= 0:
        raise ValueError(f"temperature must be > 0, got {temperature}")
    T = temperature
    s_log_probs = F.log_softmax(student_logits / T, dim=-1)
    t_probs = F.softmax(teacher_logits.detach() / T, dim=-1)
    return F.kl_div(s_log_probs, t_probs, reduction="batchmean") * (T * T)


def distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    temperature: float,
    alpha: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return ``(total, soft, hard)`` for the Hinton KD objective.

    ``alpha`` weights the soft (teacher-matching) term vs the hard (gold-label CE)
    term: ``total = alpha*soft + (1-alpha)*hard``.
    """
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")

    soft = soft_kd_loss(student_logits, teacher_logits, temperature=temperature)
    hard = F.cross_entropy(student_logits, labels)
    total = alpha * soft + (1.0 - alpha) * hard
    return total, soft.detach(), hard.detach()
