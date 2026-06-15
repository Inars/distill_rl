"""SetFit/mnli loading, tokenization, and DataLoaders.

SetFit/mnli columns:
    text1 (premise), text2 (hypothesis), label (int), idx, label_text

Label order (verified against the dataset) is already this project's canonical
order:
    0 = entailment, 1 = neutral, 2 = contradiction

The teacher (FacebookAI/roberta-large-mnli) uses the *reverse* order
(contradiction/neutral/entailment) and is remapped to canonical in
``distill_rl.models.teacher``. Keeping a single canonical order means the student
head, the gold labels, the KD targets, and the GRPO reference all line up.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from datasets import Dataset, load_dataset
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerBase

CANONICAL_LABEL_ORDER = ["entailment", "neutral", "contradiction"]


@dataclass
class MNLIBatch:
    """A tokenized, padded batch. Shared by the supervised and GRPO loops."""

    input_ids: torch.Tensor       # (B, L) long
    attention_mask: torch.Tensor  # (B, L) long
    labels: torch.Tensor          # (B,)   long, canonical order

    def to(self, device: torch.device | str) -> "MNLIBatch":
        return MNLIBatch(
            input_ids=self.input_ids.to(device),
            attention_mask=self.attention_mask.to(device),
            labels=self.labels.to(device),
        )

    def __len__(self) -> int:
        return self.labels.shape[0]


def load_split(cfg: dict[str, Any], split_key: str) -> Dataset:
    """Load one MNLI split via the ``dataset_name`` + ``splits`` mapping in cfg.

    ``split_key`` is one of the keys under ``cfg['splits']`` (``train``/``val``/``test``).
    """
    split_name = cfg["splits"][split_key]
    return load_dataset(cfg["dataset_name"], split=split_name)


def maybe_subset(ds: Dataset, limit: int | None) -> Dataset:
    """Optionally cap a split to its first ``limit`` rows (for fast smoke runs)."""
    if limit is not None and 0 < int(limit) < len(ds):
        return ds.select(range(int(limit)))
    return ds


def tokenize_split(
    ds: Dataset,
    tokenizer: PreTrainedTokenizerBase,
    cfg: dict[str, Any],
) -> Dataset:
    """Tokenize (premise, hypothesis) pairs; keep only model inputs + labels."""
    text1, text2, label = cfg["text1_field"], cfg["text2_field"], cfg["label_field"]
    max_len = cfg["max_seq_len"]

    def _enc(batch: dict[str, list[Any]]) -> dict[str, list[Any]]:
        enc = tokenizer(
            batch[text1],
            batch[text2],
            truncation=True,
            max_length=max_len,
            padding=False,
        )
        enc["labels"] = batch[label]
        return enc

    keep = ["input_ids", "attention_mask", "labels"]
    remove = [c for c in ds.column_names if c not in keep]
    return ds.map(_enc, batched=True, remove_columns=remove)


def make_collator(tokenizer: PreTrainedTokenizerBase):
    """Dynamic-padding collator producing an ``MNLIBatch``."""
    pad_id = tokenizer.pad_token_id

    def _collate(rows: list[dict[str, Any]]) -> MNLIBatch:
        max_len = max(len(r["input_ids"]) for r in rows)
        input_ids = torch.full((len(rows), max_len), pad_id, dtype=torch.long)
        attn = torch.zeros((len(rows), max_len), dtype=torch.long)
        labels = torch.empty(len(rows), dtype=torch.long)
        for i, r in enumerate(rows):
            n = len(r["input_ids"])
            input_ids[i, :n] = torch.tensor(r["input_ids"], dtype=torch.long)
            attn[i, :n] = torch.tensor(r["attention_mask"], dtype=torch.long)
            labels[i] = r["labels"]
        return MNLIBatch(input_ids=input_ids, attention_mask=attn, labels=labels)

    return _collate


def make_dataloader(
    ds: Dataset,
    tokenizer: PreTrainedTokenizerBase,
    batch_size: int,
    *,
    shuffle: bool,
    num_workers: int = 0,
) -> DataLoader:
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=make_collator(tokenizer),
        pin_memory=torch.cuda.is_available(),  # pinned host mem speeds H2D copies on CUDA
    )
