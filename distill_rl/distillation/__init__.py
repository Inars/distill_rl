"""Knowledge-distillation losses."""

from distill_rl.distillation.losses import distillation_loss, soft_kd_loss

__all__ = ["distillation_loss", "soft_kd_loss"]
