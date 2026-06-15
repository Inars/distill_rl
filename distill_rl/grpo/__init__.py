"""Single-step (bandit) GRPO for NLI classification.

We treat each MNLI example as a one-step episode: the student policy
``pi_theta(a|x) = softmax(logits)`` defines a categorical over the 3 labels, we
sample a group of ``G`` actions per example, reward correctness, form
group-relative advantages, and update with the PPO-clipped objective plus a KL
penalty to a reference (a frozen student snapshot for plain GRPO, or the teacher
for the altered GRPO). The loss math mirrors TRL's ``GRPOTrainer``; only the
"sequence" is a single label token, so per-token reduces to per-sample.
"""

from distill_rl.grpo.advantage import correctness_reward, group_relative_advantage
from distill_rl.grpo.loss import GRPOMetrics, categorical_kl, grpo_loss
from distill_rl.grpo.sampling import gather_logprobs, policy_entropy, sample_actions

__all__ = [
    "sample_actions",
    "gather_logprobs",
    "policy_entropy",
    "correctness_reward",
    "group_relative_advantage",
    "categorical_kl",
    "grpo_loss",
    "GRPOMetrics",
]
