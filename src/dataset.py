"""Dataset and collator for emoji retrieval training with hard negatives."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from src.preprocess import normalize_text


class EmojiRetrievalDataset(Dataset):
    """Loads train.jsonl and resolves emoji_id → label_text via catalog.

    Each sample returns query text, positive label text, and hard negative label texts.
    """

    def __init__(self, data_path: str, emoji_catalog_path: str):
        self.samples: list[dict[str, Any]] = []
        with open(data_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    self.samples.append(json.loads(line))

        with open(emoji_catalog_path) as f:
            catalog = json.load(f)
        self.emoji_to_label: dict[str, str] = {
            entry["emoji_id"]: entry["label_text"] for entry in catalog
        }

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        sample = self.samples[idx]
        query_text = normalize_text(sample["query"])
        label_text = self.emoji_to_label[sample["positive_label"]]

        neg_texts = []
        for neg_id in sample.get("negative_labels", []):
            if neg_id in self.emoji_to_label:
                neg_texts.append(self.emoji_to_label[neg_id])

        return {
            "query_text": query_text,
            "label_text": label_text,
            "neg_texts": neg_texts,
        }


QUERY_TEMPLATE = (
    "Instruct: 与えられたSlack投稿に対して最も適切なリアクション名テキストを検索せよ\n"
    "Query: {text}"
)
LABEL_TEMPLATE = "{text}"
TERMINATOR = "<|endoftext|>"


class EmojiCollator:
    """Tokenizes query, positive label, and hard negative texts.

    Hard negatives are padded to the max count (K) within the batch.
    """

    def __init__(self, tokenizer: Any, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batch: list[dict[str, Any]]) -> dict:
        # Query texts
        query_texts = [
            QUERY_TEMPLATE.format(text=item["query_text"]) + TERMINATOR
            for item in batch
        ]
        # Positive label texts
        label_texts = [
            LABEL_TEMPLATE.format(text=item["label_text"]) + TERMINATOR
            for item in batch
        ]

        query_enc = self.tokenizer(
            query_texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        label_enc = self.tokenizer(
            label_texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        result = {
            "query_input_ids": query_enc["input_ids"],
            "query_attention_mask": query_enc["attention_mask"],
            "label_input_ids": label_enc["input_ids"],
            "label_attention_mask": label_enc["attention_mask"],
        }

        # Hard negatives: pad to max K within batch
        max_k = max(len(item["neg_texts"]) for item in batch)
        if max_k > 0:
            all_neg_texts = []
            neg_counts = []
            for item in batch:
                negs = item["neg_texts"]
                neg_counts.append(len(negs))
                for neg in negs:
                    all_neg_texts.append(
                        LABEL_TEMPLATE.format(text=neg) + TERMINATOR
                    )
                # Pad with empty strings (will be masked later)
                for _ in range(max_k - len(negs)):
                    all_neg_texts.append(TERMINATOR)

            neg_enc = self.tokenizer(
                all_neg_texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            B = len(batch)
            seq_len = neg_enc["input_ids"].size(1)
            result["neg_input_ids"] = neg_enc["input_ids"].view(B, max_k, seq_len)
            result["neg_attention_mask"] = neg_enc["attention_mask"].view(B, max_k, seq_len)

            # Mask for real vs padded negatives: (B, K)
            neg_valid_mask = torch.zeros(B, max_k)
            for i, count in enumerate(neg_counts):
                neg_valid_mask[i, :count] = 1.0
            result["neg_valid_mask"] = neg_valid_mask

        return result
