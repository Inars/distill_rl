"""CPU tests for student / teacher / reference. Teacher label-permutation is the
most error-prone bit, so it is checked structurally AND end-to-end."""

from __future__ import annotations

import pytest
import torch
from transformers import AutoTokenizer

from distill_rl.data.mnli import CANONICAL_LABEL_ORDER
from distill_rl.models.reference import StudentSnapshotReference, TeacherReference, build_reference
from distill_rl.models.student import load_student
from distill_rl.models.teacher import _build_permutation, load_teacher

STUDENT_CFG = {"model_name": "FacebookAI/roberta-base", "num_labels": 3, "classifier_dropout": 0.1}
TEACHER_CFG = {
    "model_name": "FacebookAI/roberta-large-mnli",
    "dtype": "float32",
    "native_label_order": ["contradiction", "neutral", "entailment"],
}


@pytest.fixture(scope="module")
def tokenizer():
    return AutoTokenizer.from_pretrained("FacebookAI/roberta-base")


@pytest.fixture(scope="module")
def student():
    model, _ = load_student(STUDENT_CFG, device="cpu")
    return model.eval()


@pytest.fixture(scope="module")
def teacher():
    t, _ = load_teacher(TEACHER_CFG, device="cpu")
    return t


def _encode(tokenizer, premise, hypothesis):
    enc = tokenizer(premise, hypothesis, return_tensors="pt", truncation=True, max_length=128)
    return enc["input_ids"], enc["attention_mask"]


def test_permutation_reverses_standard_order():
    # native = [contradiction, neutral, entailment]; canonical = [entailment, neutral, contradiction]
    perm = _build_permutation(TEACHER_CFG["native_label_order"])
    assert perm.tolist() == [2, 1, 0]


def test_student_head_is_3way(student):
    assert student.classifier.out_proj.out_features == len(CANONICAL_LABEL_ORDER) == 3


def test_student_forward_shape(student, tokenizer):
    ids, mask = _encode(tokenizer, "A man plays guitar.", "A man plays an instrument.")
    logits = student(input_ids=ids, attention_mask=mask).logits
    assert logits.shape == (1, 3)


@pytest.mark.parametrize(
    "premise,hypothesis,expected",
    [
        ("A man is playing a guitar.", "A man is playing an instrument.", "entailment"),
        ("A man is sleeping in his bed.", "A man is running a marathon outside.", "contradiction"),
    ],
)
def test_teacher_predicts_in_canonical_order(teacher, tokenizer, premise, hypothesis, expected):
    ids, mask = _encode(tokenizer, premise, hypothesis)
    logits = teacher(ids, mask)
    assert logits.shape == (1, 3)
    pred = CANONICAL_LABEL_ORDER[int(logits.argmax(-1))]
    assert pred == expected, f"teacher said {pred!r}, expected {expected!r}"


def test_student_snapshot_is_frozen_and_independent(student, tokenizer):
    ref = StudentSnapshotReference(student)
    assert all(not p.requires_grad for p in ref.model.parameters())
    ids, mask = _encode(tokenizer, "x.", "y.")
    before = ref.logits(ids, mask).clone()
    # mutate the live student; snapshot must not move
    with torch.no_grad():
        for p in student.parameters():
            p.add_(1.0)
    after = ref.logits(ids, mask)
    assert torch.allclose(before, after)
    # restore student
    with torch.no_grad():
        for p in student.parameters():
            p.add_(-1.0)


def test_build_reference_dispatch(student, teacher):
    assert isinstance(build_reference("student", student=student), StudentSnapshotReference)
    assert isinstance(build_reference("teacher", teacher=teacher), TeacherReference)
    with pytest.raises(ValueError):
        build_reference("bogus", student=student)
