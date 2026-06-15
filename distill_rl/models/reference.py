"""GRPO reference model: the distribution the KL / reward-penalty regularizes toward.

Standard GRPO uses a frozen snapshot of the policy as the reference. The *altered*
GRPO in this project swaps that for the teacher (roberta-large-mnli). Both expose
the same interface: ``logits(input_ids, attention_mask) -> (B, 3)`` in canonical
order, under ``no_grad`` and in eval mode.
"""

from __future__ import annotations

import copy

import torch
from torch import nn

from distill_rl.models.teacher import FrozenTeacher


class Reference(nn.Module):
    """Interface: frozen, eval-mode producer of canonical-order class logits."""

    @torch.no_grad()
    def logits(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class StudentSnapshotReference(Reference):
    """A frozen deep copy of the student, taken at the start of RL (the GRPO default).

    Snapshot at RL start so the KL anchors the policy to its initial behaviour,
    exactly like TRL's reference model.
    """

    def __init__(self, student: nn.Module) -> None:
        super().__init__()
        self.model = copy.deepcopy(student).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def logits(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        return self.model(input_ids=input_ids, attention_mask=attention_mask).logits


class TeacherReference(Reference):
    """The teacher (roberta-large-mnli) used as the GRPO reference (altered GRPO)."""

    def __init__(self, teacher: FrozenTeacher) -> None:
        super().__init__()
        self.teacher = teacher

    @torch.no_grad()
    def logits(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        return self.teacher(input_ids, attention_mask)


def build_reference(
    kind: str,
    *,
    student: nn.Module | None = None,
    teacher: FrozenTeacher | None = None,
) -> Reference:
    """Factory: ``kind='student'`` -> frozen student snapshot; ``'teacher'`` -> teacher."""
    if kind == "student":
        if student is None:
            raise ValueError("reference kind='student' requires a student model")
        return StudentSnapshotReference(student)
    if kind == "teacher":
        if teacher is None:
            raise ValueError("reference kind='teacher' requires a loaded teacher")
        return TeacherReference(teacher)
    raise ValueError(f"unknown reference kind: {kind!r} (expected 'student' or 'teacher')")
