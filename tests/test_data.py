"""CPU tests for the MNLI data pipeline."""

from __future__ import annotations

import pytest
import torch
from transformers import AutoTokenizer

from distill_rl.data.mnli import (
    CANONICAL_LABEL_ORDER,
    MNLIBatch,
    load_split,
    make_dataloader,
    maybe_subset,
    tokenize_split,
)

DATA_CFG = {
    "dataset_name": "SetFit/mnli",
    "splits": {"train": "train", "val": "validation", "test": "test"},
    "text1_field": "text1",
    "text2_field": "text2",
    "label_field": "label",
    "max_seq_len": 64,
}


@pytest.fixture(scope="module")
def tokenizer():
    return AutoTokenizer.from_pretrained("FacebookAI/roberta-base")


def test_canonical_label_order():
    assert CANONICAL_LABEL_ORDER == ["entailment", "neutral", "contradiction"]


def test_load_and_tokenize_shapes(tokenizer):
    ds = maybe_subset(load_split(DATA_CFG, "val"), 32)
    assert len(ds) == 32
    tok = tokenize_split(ds, tokenizer, DATA_CFG)
    assert set(tok.column_names) == {"input_ids", "attention_mask", "labels"}
    # truncation respected
    assert all(len(x) <= DATA_CFG["max_seq_len"] for x in tok["input_ids"])
    # canonical 3-class labels
    assert set(tok["labels"]).issubset({0, 1, 2})


def test_collator_pads_and_batches(tokenizer):
    ds = maybe_subset(load_split(DATA_CFG, "val"), 16)
    tok = tokenize_split(ds, tokenizer, DATA_CFG)
    loader = make_dataloader(tok, tokenizer, batch_size=8, shuffle=False, num_workers=0)
    batch = next(iter(loader))
    assert isinstance(batch, MNLIBatch)
    B, L = batch.input_ids.shape
    assert B == 8
    assert batch.attention_mask.shape == (B, L)
    assert batch.labels.shape == (B,)
    assert batch.input_ids.dtype == torch.long
    # padded positions are masked out
    pad_id = tokenizer.pad_token_id
    assert torch.equal((batch.input_ids == pad_id), (batch.attention_mask == 0)) or (
        # last real token may equal pad_id only via attention_mask==1; cross-check loosely
        (batch.attention_mask == 0).sum() == (batch.input_ids == pad_id).sum()
    )


def test_batch_to_device_cpu(tokenizer):
    ds = maybe_subset(load_split(DATA_CFG, "val"), 4)
    tok = tokenize_split(ds, tokenizer, DATA_CFG)
    loader = make_dataloader(tok, tokenizer, batch_size=4, shuffle=False)
    batch = next(iter(loader)).to("cpu")
    assert len(batch) == 4
    assert batch.input_ids.device.type == "cpu"
