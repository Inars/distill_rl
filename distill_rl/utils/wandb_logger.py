"""Thin W&B wrapper. No-ops cleanly when disabled so loops stay logger-agnostic.

Auth comes from the environment (``WANDB_API_KEY`` / ``wandb login`` cache); the
key is never read from a tracked file. Entity/project default to
``TAU-Frugal`` / ``distillation-rl`` (overridable via config or env).
"""

from __future__ import annotations

from typing import Any


class WandbLogger:
    def __init__(
        self,
        wandb_cfg: dict[str, Any],
        *,
        run_name: str,
        config: dict[str, Any],
        out_dir: str | None = None,
    ) -> None:
        mode = str(wandb_cfg.get("mode", "online"))
        self.enabled = bool(wandb_cfg.get("enabled", True)) and mode != "disabled"
        self.run = None
        self._wandb = None
        if not self.enabled:
            return

        import wandb

        self._wandb = wandb
        self.run = wandb.init(
            entity=wandb_cfg.get("entity"),
            project=wandb_cfg.get("project"),
            name=run_name,
            group=wandb_cfg.get("group"),
            tags=list(wandb_cfg.get("tags", []) or []),
            mode=mode,
            config=config,
            dir=out_dir,
        )

    def log(self, metrics: dict[str, Any], step: int | None = None) -> None:
        if self.run is not None:
            self._wandb.log(metrics, step=step)

    def set_summary(self, summary: dict[str, Any]) -> None:
        if self.run is not None:
            for k, v in summary.items():
                self.run.summary[k] = v

    def finish(self) -> None:
        if self.run is not None:
            self._wandb.finish()
            self.run = None
