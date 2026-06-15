"""CPU tests for the Hinton KD loss."""

from __future__ import annotations

import torch
from torch.nn import functional as F

from distill_rl.distillation.losses import distillation_loss, soft_kd_loss


def test_soft_loss_zero_when_logits_match():
    logits = torch.randn(8, 3)
    soft = soft_kd_loss(logits, logits.clone(), temperature=2.0)
    assert torch.allclose(soft, torch.zeros(()), atol=1e-6)


def test_alpha_extremes():
    s = torch.randn(8, 3, requires_grad=True)
    t = torch.randn(8, 3)
    y = torch.randint(0, 3, (8,))

    total0, soft0, hard0 = distillation_loss(s, t, y, temperature=2.0, alpha=0.0)
    assert torch.allclose(total0, F.cross_entropy(s, y))  # alpha=0 -> pure hard CE

    total1, soft1, hard1 = distillation_loss(s, t, y, temperature=2.0, alpha=1.0)
    assert torch.allclose(total1, soft_kd_loss(s, t, temperature=2.0))  # alpha=1 -> pure soft


def test_temperature_squared_scaling():
    s = torch.randn(16, 3)
    t = torch.randn(16, 3)
    # KL(softmax(s/T)||softmax(t/T)) * T^2 ; check the explicit T^2 factor is applied
    T = 4.0
    s_lp = F.log_softmax(s / T, dim=-1)
    t_p = F.softmax(t / T, dim=-1)
    expected = F.kl_div(s_lp, t_p, reduction="batchmean") * (T * T)
    assert torch.allclose(soft_kd_loss(s, t, temperature=T), expected)


def test_grad_flows_to_student_not_teacher():
    s = torch.randn(8, 3, requires_grad=True)
    t = torch.randn(8, 3, requires_grad=True)
    y = torch.randint(0, 3, (8,))
    total, _, _ = distillation_loss(s, t, y, temperature=2.0, alpha=0.5)
    total.backward()
    assert s.grad is not None and s.grad.abs().sum() > 0
    assert t.grad is None  # teacher targets are detached
