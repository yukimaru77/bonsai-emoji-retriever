"""BonsaiEmbedder: LoRA-adapted causal LM for text embedding with MRL support."""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig as PeftLoraConfig, get_peft_model, TaskType

from src.config import Config

logger = logging.getLogger(__name__)

QUERY_TEMPLATE = (
    "Instruct: 与えられたSlack投稿に対して最も適切なリアクション名テキストを検索せよ\n"
    "Query: {text}"
)
LABEL_TEMPLATE = "{text}"
TERMINATOR = "<|endoftext|>"


def _try_load_quantization_config(config: Config) -> dict[str, Any] | None:
    """Try to build BitsAndBytesConfig for QLoRA. Returns None if unavailable."""
    if not config.quantization.enabled:
        return None

    try:
        from transformers import BitsAndBytesConfig

        dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16}
        compute_dtype = dtype_map.get(
            config.quantization.bnb_4bit_compute_dtype, torch.float16
        )

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=config.quantization.load_in_4bit,
            bnb_4bit_quant_type=config.quantization.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=config.quantization.bnb_4bit_use_double_quant,
        )
        logger.info("QLoRA enabled: 4-bit NF4 quantization")
        return {"quantization_config": bnb_config}
    except ImportError:
        logger.warning(
            "bitsandbytes not available (common on aarch64). "
            "Falling back to FP16 without quantization."
        )
        return None
    except Exception as e:
        logger.warning("QLoRA setup failed: %s. Falling back to FP16.", e)
        return None


class BonsaiEmbedder(nn.Module):
    """Shared encoder for queries and labels.

    - Last-token pooling on the <|endoftext|> terminator
    - L2 normalization
    - LoRA adapters on attention projections
    - Optional QLoRA (4-bit NF4)
    - MRL-compatible: returns full-dim embeddings, truncation done in trainer
    """

    def __init__(self, config: Config):
        super().__init__()
        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        torch_dtype = dtype_map.get(config.model.torch_dtype, torch.float16)

        self.tokenizer = AutoTokenizer.from_pretrained(
            config.model.model_id,
            trust_remote_code=True,
        )
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # QLoRA or FP16
        extra_kwargs = _try_load_quantization_config(config) or {}

        self.base_model = AutoModelForCausalLM.from_pretrained(
            config.model.model_id,
            dtype=torch_dtype,
            trust_remote_code=True,
            **extra_kwargs,
        )
        self.base_model.config.use_cache = False

        peft_config = PeftLoraConfig(
            r=config.lora.r,
            lora_alpha=config.lora.lora_alpha,
            lora_dropout=config.lora.lora_dropout,
            target_modules=config.lora.target_modules,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        self.base_model = get_peft_model(self.base_model, peft_config)

        self.endoftext_id = self.tokenizer.convert_tokens_to_ids(TERMINATOR)
        self.max_length = config.model.max_length

    def _tokenize(self, texts: list[str]) -> dict:
        return self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

    def _pool_last_token(
        self, hidden_states: Tensor, attention_mask: Tensor
    ) -> Tensor:
        """Extract hidden state at the last non-pad token for each sample."""
        last_token_indices = attention_mask.sum(dim=1) - 1
        batch_indices = torch.arange(
            hidden_states.size(0), device=hidden_states.device
        )
        pooled = hidden_states[batch_indices, last_token_indices]
        return F.normalize(pooled, p=2, dim=-1)

    def forward(self, input_ids: Tensor, attention_mask: Tensor) -> Tensor:
        """Forward pass → last-token pooling → L2 normalization.

        Returns: (batch_size, hidden_dim) normalized embeddings.
        """
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        hidden_states = outputs.hidden_states[-1]
        return self._pool_last_token(hidden_states, attention_mask)

    def encode_batch(
        self, input_ids: Tensor, attention_mask: Tensor
    ) -> Tensor:
        """Encode a batch of (B, K, seq_len) hard negatives.

        Reshapes to (B*K, seq_len), encodes, reshapes back to (B, K, D).
        """
        B, K, seq_len = input_ids.shape
        flat_ids = input_ids.view(B * K, seq_len)
        flat_mask = attention_mask.view(B * K, seq_len)
        flat_embs = self.forward(flat_ids, flat_mask)
        return flat_embs.view(B, K, -1)

    def _apply_template_and_encode(
        self, texts: list[str], template: str
    ) -> Tensor:
        formatted = [template.format(text=t) + TERMINATOR for t in texts]
        encoded = self._tokenize(formatted)
        encoded = {k: v.to(self.base_model.device) for k, v in encoded.items()}
        return self.forward(encoded["input_ids"], encoded["attention_mask"])

    def encode_queries(self, texts: list[str]) -> Tensor:
        return self._apply_template_and_encode(texts, QUERY_TEMPLATE)

    def encode_labels(self, texts: list[str]) -> Tensor:
        return self._apply_template_and_encode(texts, LABEL_TEMPLATE)

    def save_adapter(self, path: str) -> None:
        self.base_model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
