"""CPU tests for the KDRL KD-RKL loss (k2/k3/exact), masking, and beta schedule."""

from __future__ import annotations

import torch
from torch.nn import functional as F

from distill_rl.grpo.kdrl import (
    anneal_beta,
    kd_rkl_loss,
    kd_rkl_mask,
    resolve_anneal_delta,
)
from distill_rl.grpo.sampling import sample_actions

ESTIMATORS = ("k2", "k3", "exact")


# ---------- loss is zero when the teacher equals the student ----------

def test_kd_rkl_zero_when_teacher_equals_student():
    logits = torch.randn(4, 3, requires_grad=True)
    gen = torch.Generator().manual_seed(0)
    actions = sample_actions(logits.detach(), 6, gen)
    correct = torch.zeros(4, 6, dtype=torch.bool)
    for est in ESTIMATORS:
        loss, m = kd_rkl_loss(logits, logits.detach(), actions, correct, estimator=est)
        assert abs(loss.detach().item()) < 1e-6, f"{est}: loss={loss.detach().item()}"
        assert abs(m.kl_to_teacher) < 1e-6


# ---------- k2 gradient has the right sign ----------

def test_k2_gradient_sign_increases_teacher_preferred_prob():
    # Uniform student, teacher peaked on class 0, sampled actions fixed to class 0:
    # one gradient-descent step on the k2 loss must raise p(class 0).
    logits = torch.zeros(1, 3, requires_grad=True)
    teacher = torch.tensor([[3.0, 0.0, 0.0]])
    actions = torch.zeros(1, 4, dtype=torch.long)
    correct = torch.zeros(1, 4, dtype=torch.bool)
    loss, _ = kd_rkl_loss(logits, teacher, actions, correct, estimator="k2")
    loss.backward()
    with torch.no_grad():
        stepped = logits - 0.5 * logits.grad
    p0_before = F.softmax(logits.detach(), -1)[0, 0]
    p0_after = F.softmax(stepped, -1)[0, 0]
    assert p0_after > p0_before


# ---------- exact reverse-KL minimization converges to the teacher ----------

def test_exact_minimization_matches_teacher():
    logits = torch.zeros(1, 3, requires_grad=True)
    teacher = torch.tensor([[3.0, 1.0, 0.0]])  # full support -> reverse KL -> teacher
    actions = torch.zeros(1, 4, dtype=torch.long)  # unused by the exact KL value
    correct = torch.zeros(1, 4, dtype=torch.bool)
    opt = torch.optim.SGD([logits], lr=0.5)
    for _ in range(500):
        opt.zero_grad()
        loss, _ = kd_rkl_loss(logits, teacher, actions, correct, estimator="exact")
        loss.backward()
        opt.step()
    p = F.softmax(logits.detach(), -1)[0]
    t = F.softmax(teacher[0], -1)
    assert torch.allclose(p, t, atol=1e-2), f"{p} != {t}"


# ---------- masking ----------

def test_kd_rkl_mask_levels():
    correct = torch.tensor([[True, False, False],
                            [False, False, False]])
    assert torch.equal(kd_rkl_mask(correct, "none"), torch.ones(2, 3))
    assert torch.equal(
        kd_rkl_mask(correct, "response"),
        torch.tensor([[0.0, 1.0, 1.0], [1.0, 1.0, 1.0]]),
    )
    # row 0 has a correct sample -> whole group dropped; row 1 fully hard -> kept.
    assert torch.equal(
        kd_rkl_mask(correct, "group"),
        torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]),
    )


def test_response_masking_zeroes_when_all_correct():
    logits = torch.randn(2, 3, requires_grad=True)
    teacher = torch.randn(2, 3)
    actions = torch.zeros(2, 5, dtype=torch.long)
    correct = torch.ones(2, 5, dtype=torch.bool)  # everything already correct
    for est in ESTIMATORS:
        loss, m = kd_rkl_loss(
            logits, teacher, actions, correct, estimator=est, masking="response"
        )
        assert loss.detach().item() == 0.0, est
        assert m.kd_mask_frac == 0.0


def test_group_masking_keeps_only_fully_hard_examples():
    logits = torch.randn(2, 3, requires_grad=True)
    teacher = torch.randn(2, 3)
    actions = torch.zeros(2, 4, dtype=torch.long)
    correct = torch.tensor([[True, False, False, False],   # example 0: has a win -> masked out
                            [False, False, False, False]])  # example 1: fully hard -> kept
    _, m = kd_rkl_loss(logits, teacher, actions, correct, estimator="k2", masking="group")
    assert m.kd_mask_frac == 0.5  # 4 of 8 entries kept


# ---------- beta schedule ----------

def test_anneal_beta_decays_and_clamps():
    assert anneal_beta(0, beta_init=5e-3, beta_min=1e-3, delta=1e-4) == 5e-3
    assert abs(anneal_beta(10, beta_init=5e-3, beta_min=1e-3, delta=1e-4) - 4e-3) < 1e-12
    assert anneal_beta(10_000, beta_init=5e-3, beta_min=1e-3, delta=1e-4) == 1e-3


def test_resolve_anneal_delta():
    # reach beta_min at step 500 -> delta = (5e-3 - 1e-3) / 500 = 8e-6
    d = resolve_anneal_delta(beta_init=5e-3, beta_min=1e-3, anneal_frac=0.5, total_steps=1000)
    assert abs(d - 8e-6) < 1e-12
    # explicit positive delta passes through untouched
    assert resolve_anneal_delta(
        beta_init=5e-3, beta_min=1e-3, anneal_frac=0.5, total_steps=1000, delta=2e-5
    ) == 2e-5
