"""CPU tests for the GRPO core (sampling, advantage, loss)."""

from __future__ import annotations

import math

import torch
from torch.nn import functional as F

from distill_rl.grpo.advantage import correctness_reward, group_relative_advantage
from distill_rl.grpo.loss import categorical_kl, grpo_loss
from distill_rl.grpo.sampling import gather_logprobs, policy_entropy, sample_actions


# ---------- sampling ----------

def test_sample_actions_shape_and_range():
    logits = torch.randn(5, 3)
    a = sample_actions(logits, group_size=7)
    assert a.shape == (5, 7)
    assert a.min() >= 0 and a.max() <= 2
    assert a.dtype == torch.long


def test_sample_actions_reproducible_with_generator():
    logits = torch.randn(4, 3)
    g1 = torch.Generator().manual_seed(123)
    g2 = torch.Generator().manual_seed(123)
    assert torch.equal(sample_actions(logits, 8, g1), sample_actions(logits, 8, g2))


def test_sample_actions_peaked_logits():
    logits = torch.tensor([[0.0, 100.0, 0.0]])  # all mass on class 1
    a = sample_actions(logits, group_size=20)
    assert (a == 1).all()


def test_gather_logprobs_matches_manual():
    logits = torch.randn(3, 3)
    actions = torch.tensor([[0, 2], [1, 1], [2, 0]])
    lp = gather_logprobs(logits, actions)
    manual = torch.gather(F.log_softmax(logits, -1), -1, actions)
    assert torch.allclose(lp, manual)


def test_policy_entropy_bounds():
    uniform = torch.zeros(1, 3)
    assert torch.allclose(policy_entropy(uniform), torch.tensor([math.log(3)]), atol=1e-6)
    peaked = torch.tensor([[0.0, 100.0, 0.0]])
    assert policy_entropy(peaked).item() < 1e-3


# ---------- advantage ----------

def test_correctness_reward():
    actions = torch.tensor([[0, 1, 2]])
    labels = torch.tensor([1])
    r = correctness_reward(actions, labels)
    assert torch.equal(r, torch.tensor([[0.0, 1.0, 0.0]]))
    r2 = correctness_reward(actions, labels, correct=1.0, wrong=-1.0)
    assert torch.equal(r2, torch.tensor([[-1.0, 1.0, -1.0]]))


def test_group_relative_advantage_centered():
    rewards = torch.tensor([[1.0, 0.0, 0.0, 1.0]])
    adv = group_relative_advantage(rewards, scale_rewards=False)
    assert torch.allclose(adv.mean(dim=1), torch.zeros(1), atol=1e-6)


def test_group_relative_advantage_zero_variance():
    rewards = torch.ones(2, 5)  # all correct -> no signal
    adv = group_relative_advantage(rewards, scale_rewards=True)
    assert torch.allclose(adv, torch.zeros_like(adv), atol=1e-3)


def test_group_relative_advantage_scaled_unit_std():
    rewards = torch.tensor([[0.0, 1.0, 0.0, 1.0, 0.0, 1.0]])
    adv = group_relative_advantage(rewards, scale_rewards=True, eps=0.0)
    assert abs(adv.std(dim=1, correction=1).item() - 1.0) < 1e-4


# ---------- KL ----------

def test_categorical_kl_zero_when_equal():
    logits = torch.randn(6, 3)
    assert torch.allclose(categorical_kl(logits, logits.clone()), torch.zeros(6), atol=1e-6)


def test_categorical_kl_nonnegative_and_known_value():
    p = torch.log(torch.tensor([[0.5, 0.25, 0.25]]))
    q = torch.log(torch.tensor([[1 / 3, 1 / 3, 1 / 3]]))
    kl = categorical_kl(p, q)
    expected = 0.5 * math.log(0.5 / (1 / 3)) + 2 * 0.25 * math.log(0.25 / (1 / 3))
    assert kl.item() >= 0
    assert abs(kl.item() - expected) < 1e-5


# ---------- grpo_loss ----------

def _mu1_old(logits, actions):
    """Old log-probs for the mu=1 case = current policy log-probs, detached."""
    return gather_logprobs(logits, actions).detach()


def test_grpo_loss_kl_zero_and_ratio_one_at_mu1():
    logits = torch.randn(4, 3, requires_grad=True)
    gen = torch.Generator().manual_seed(0)
    actions = sample_actions(logits.detach(), 6, gen)
    rewards = correctness_reward(actions, torch.randint(0, 3, (4,)))
    adv = group_relative_advantage(rewards).detach()
    old = _mu1_old(logits, actions)
    loss, m = grpo_loss(
        logits, old, actions, adv, ref_logits=logits.detach(),
        beta=0.04, clip_eps=0.2, kl_estimator="exact",
    )
    assert abs(m.ratio_mean - 1.0) < 1e-6 and abs(m.ratio_max - 1.0) < 1e-6
    assert abs(m.kl) < 1e-6  # ref == policy


def test_grpo_loss_clipping_binds_on_large_ratio():
    logits = torch.zeros(1, 3, requires_grad=True)
    actions = torch.tensor([[0]])
    adv = torch.tensor([[1.0]])
    old = torch.tensor([[-5.0]])  # tiny old prob -> huge ratio
    _, m = grpo_loss(logits, old, actions, adv, logits.detach(), beta=0.0, clip_eps=0.2)
    assert m.frac_clipped == 1.0
    assert abs(m.loss - (-1.2)) < 1e-4  # clipped value = (1+eps)*A = 1.2


def test_grpo_increases_prob_of_positive_advantage_action():
    torch.manual_seed(0)
    logits = torch.zeros(1, 3, requires_grad=True)  # uniform start
    actions = torch.tensor([[0, 1, 2, 0, 1, 2]])
    advantages = torch.tensor([[1.0, -0.5, -0.5, 1.0, -0.5, -0.5]])  # action 0 favored
    opt = torch.optim.SGD([logits], lr=1.0)
    p_before = F.softmax(logits.detach(), -1)[0, 0].item()
    for _ in range(50):
        opt.zero_grad()
        old = gather_logprobs(logits, actions).detach()
        loss, _ = grpo_loss(
            logits, old, actions, advantages, logits.detach(),
            beta=0.0, clip_eps=0.2,
        )
        loss.backward()
        opt.step()
    p_after = F.softmax(logits.detach(), -1)[0, 0].item()
    assert p_after > p_before + 0.2, f"prob did not rise: {p_before:.3f} -> {p_after:.3f}"


def test_grpo_k3_kl_zero_when_ref_equals_policy():
    logits = torch.randn(3, 3, requires_grad=True)
    gen = torch.Generator().manual_seed(1)
    actions = sample_actions(logits.detach(), 5, gen)
    adv = torch.zeros(3, 5)
    old = _mu1_old(logits, actions)
    _, m = grpo_loss(
        logits, old, actions, adv, ref_logits=logits.detach(),
        beta=0.1, clip_eps=0.2, kl_estimator="k3",
    )
    assert abs(m.kl) < 1e-6
