"""CPU tests for TRRD: the mixture-anchored ratio and clipped surrogate loss."""

from __future__ import annotations

import torch
from torch.nn import functional as F

from distill_rl.grpo.advantage import correctness_reward, group_relative_advantage
from distill_rl.grpo.loss import grpo_loss
from distill_rl.grpo.sampling import gather_logprobs, sample_actions
from distill_rl.grpo.trrd import trrd_log_ratio, trrd_loss


# ---------- alpha = 1 recovers standard GRPO ----------

def test_alpha_one_recovers_grpo():
    torch.manual_seed(0)
    logits = torch.randn(4, 3, requires_grad=True)
    gen = torch.Generator().manual_seed(0)
    actions = sample_actions(logits.detach(), 6, gen)
    old = gather_logprobs(logits.detach(), actions) - 0.1  # perturb so ratio != 1
    teacher_logits = torch.randn(4, 3)
    teacher_logp = gather_logprobs(teacher_logits, actions)
    rewards = correctness_reward(actions, torch.randint(0, 3, (4,)))
    adv = group_relative_advantage(rewards).detach()
    ref_logits = torch.randn(4, 3)

    g_loss, gm = grpo_loss(
        logits, old, actions, adv, ref_logits, beta=0.04, clip_eps=0.2, kl_estimator="exact",
    )
    t_loss, tm = trrd_loss(
        logits, old, teacher_logp, actions, adv, ref_logits,
        alpha=1.0, distill_clip=1.0, beta=0.04, clip_eps=0.2, kl_estimator="exact",
    )
    assert torch.allclose(g_loss, t_loss, atol=1e-6)
    assert abs(gm.policy_loss - tm.policy_loss) < 1e-6
    assert abs(gm.ratio_mean - tm.ratio_mean) < 1e-6
    assert abs(gm.kl - tm.kl) < 1e-6


# ---------- alpha = 0 is the (clamped) teacher ratio ----------

def test_alpha_zero_is_teacher_ratio():
    logits = torch.randn(3, 3, requires_grad=True)
    gen = torch.Generator().manual_seed(1)
    actions = sample_actions(logits.detach(), 5, gen)
    old = gather_logprobs(logits.detach(), actions)
    teacher_logits = torch.randn(3, 3)
    teacher_logp = gather_logprobs(teacher_logits, actions)
    adv = torch.zeros(3, 5)

    _, tm = trrd_loss(
        logits, old, teacher_logp, actions, adv, logits.detach(),
        alpha=0.0, distill_clip=100.0, beta=0.0, clip_eps=0.2,  # huge clip -> no clamping
    )
    policy_logp = gather_logprobs(logits, actions)
    expected = torch.exp(policy_logp - teacher_logp).detach()
    assert abs(tm.ratio_mean - float(expected.mean())) < 1e-5


# ---------- Appendix-B clamp bounds the teacher log-ratio ----------

def test_distill_clip_binds_and_bounds_log_ratio():
    logits = torch.tensor([[5.0, 0.0, -5.0]], requires_grad=True)   # student prefers class 0
    teacher_logits = torch.tensor([[-5.0, 0.0, 5.0]])               # teacher prefers class 2
    actions = torch.tensor([[0]])
    old = gather_logprobs(logits.detach(), actions)
    teacher_logp = gather_logprobs(teacher_logits, actions)

    log_r, clamp_active = trrd_log_ratio(
        gather_logprobs(logits, actions), old, teacher_logp, alpha=0.0, distill_clip=1.0,
    )
    assert bool(clamp_active.all())                     # |log(pi/pi_T)| on class 0 exceeds the clamp
    assert abs(log_r.detach().item()) <= 1.0 + 1e-6     # alpha=0 -> log_r == clamped distill in [-1, 1]


# ---------- positive advantage raises the favored action's probability ----------

def test_trrd_increases_prob_of_positive_advantage_action():
    torch.manual_seed(0)
    logits = torch.zeros(1, 3, requires_grad=True)
    teacher_logits = torch.tensor([[2.0, 0.0, 0.0]])  # teacher agrees: prefers class 0
    actions = torch.tensor([[0, 1, 2, 0, 1, 2]])
    advantages = torch.tensor([[1.0, -0.5, -0.5, 1.0, -0.5, -0.5]])
    teacher_logp = gather_logprobs(teacher_logits, actions)
    opt = torch.optim.SGD([logits], lr=1.0)
    p0_before = F.softmax(logits.detach(), -1)[0, 0].item()
    for _ in range(80):
        opt.zero_grad()
        old = gather_logprobs(logits, actions).detach()
        loss, _ = trrd_loss(
            logits, old, teacher_logp, actions, advantages, logits.detach(),
            alpha=0.5, distill_clip=1.0, beta=0.0, clip_eps=0.2,
        )
        loss.backward()
        opt.step()
    p0_after = F.softmax(logits.detach(), -1)[0, 0].item()
    assert p0_after > p0_before + 0.1, f"prob did not rise: {p0_before:.3f} -> {p0_after:.3f}"


# ---------- the PPO clip still binds on large ratios ----------

def test_ppo_clip_binds_on_large_ratio():
    logits = torch.zeros(1, 3, requires_grad=True)
    actions = torch.tensor([[0]])
    old = torch.tensor([[-5.0]])  # tiny old prob -> huge trust ratio
    teacher_logp = gather_logprobs(logits.detach(), actions)  # teacher == student -> distill ~ 0
    adv = torch.tensor([[1.0]])
    _, m = trrd_loss(
        logits, old, teacher_logp, actions, adv, logits.detach(),
        alpha=1.0, distill_clip=1.0, beta=0.0, clip_eps=0.2,
    )
    assert m.frac_clipped == 1.0
    assert abs(m.loss - (-1.2)) < 1e-4  # clipped (1 + eps) * A = 1.2
