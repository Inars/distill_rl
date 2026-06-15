"""Teacher: FacebookAI/roberta-large-mnli, frozen, with label-order remap.

The HF checkpoint's labels are {CONTRADICTION:0, NEUTRAL:1, ENTAILMENT:2}. The
project's canonical order is {entailment:0, neutral:1, contradiction:2} (matches
SetFit/mnli). This wrapper permutes the output logits so every caller sees the
canonical order.

The teacher plays two roles in this project, both with this same canonical-order
output:
  * KD teacher (logit/response distillation) in the KD and GRPO+distill runs.
  * GRPO KL / reward-penalty reference in the GRPO+distill (altered) run.
Both roberta-base and roberta-large share the same BPE tokenizer/vocab, so the
student's input_ids feed the teacher directly (no re-tokenization needed).
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from distill_rl.data.mnli import CANONICAL_LABEL_ORDER

_DTYPES = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


class FrozenTeacher(nn.Module):
    """Frozen roberta-large-mnli with logits permuted to the canonical label order."""

    def __init__(self, model: nn.Module, permutation: torch.Tensor) -> None:
        super().__init__()
        self.model = model
        # permutation[i] = index in the native logits for canonical class i.
        # Registered as a buffer so .to(device) carries it.
        self.register_buffer("permutation", permutation, persistent=False)

        for p in self.model.parameters():
            p.requires_grad_(False)
        self.model.eval()

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        out = self.model(input_ids=input_ids, attention_mask=attention_mask)
        native_logits = out.logits  # (B, 3) native CONTRADICTION/NEUTRAL/ENTAILMENT order
        return native_logits.index_select(dim=-1, index=self.permutation)


def _build_permutation(native_order: list[str]) -> torch.Tensor:
    """Return idx tensor such that canonical[i] = native[idx[i]]."""
    native_lower = [s.lower() for s in native_order]
    return torch.tensor(
        [native_lower.index(name) for name in CANONICAL_LABEL_ORDER],
        dtype=torch.long,
    )


def load_teacher(
    cfg: dict[str, Any],
    device: str,
) -> tuple[FrozenTeacher, PreTrainedTokenizerBase]:
    name = cfg["model_name"]
    dtype = _DTYPES[cfg.get("dtype", "float32")]

    model = AutoModelForSequenceClassification.from_pretrained(name, dtype=dtype)
    tokenizer = AutoTokenizer.from_pretrained(name)

    perm = _build_permutation(cfg["native_label_order"])
    teacher = FrozenTeacher(model=model, permutation=perm).to(device)
    teacher.eval()
    return teacher, tokenizer
