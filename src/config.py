"""Configuration loader: YAML → dataclasses."""

from __future__ import annotations

import yaml
from dataclasses import dataclass, field
from pathlib import Path
from typing import List


@dataclass
class ModelConfig:
    model_id: str = "prism-ml/Bonsai-8B-unpacked"
    max_length: int = 512
    torch_dtype: str = "float16"


@dataclass
class LoraConfig:
    r: int = 16
    lora_alpha: int = 64
    lora_dropout: float = 0.1
    target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"]
    )


@dataclass
class QuantizationConfig:
    enabled: bool = False
    load_in_4bit: bool = True
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_compute_dtype: str = "float16"
    bnb_4bit_use_double_quant: bool = True


@dataclass
class TrainingConfig:
    num_epochs: int = 1
    batch_size: int = 2
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    seed: int = 42


@dataclass
class LossConfig:
    init_temperature: float = 0.05
    temperature_lr: float = 1e-3
    fn_margin: float = 0.1


@dataclass
class MRLConfig:
    enabled: bool = True
    dims: List[int] = field(default_factory=lambda: [256, 1024, 4096])


@dataclass
class DataConfig:
    train_path: str = "data/train.jsonl"
    emoji_catalog_path: str = "data/emoji_catalog.json"


@dataclass
class OutputConfig:
    save_dir: str = "outputs/"


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    lora: LoraConfig = field(default_factory=LoraConfig)
    quantization: QuantizationConfig = field(default_factory=QuantizationConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    mrl: MRLConfig = field(default_factory=MRLConfig)
    data: DataConfig = field(default_factory=DataConfig)
    output: OutputConfig = field(default_factory=OutputConfig)


def load_config(path: str | Path) -> Config:
    """Load config from YAML file, merging with defaults."""
    with open(path) as f:
        raw = yaml.safe_load(f) or {}

    config = Config()
    section_map = {
        "model": ModelConfig,
        "lora": LoraConfig,
        "quantization": QuantizationConfig,
        "training": TrainingConfig,
        "loss": LossConfig,
        "mrl": MRLConfig,
        "data": DataConfig,
        "output": OutputConfig,
    }
    for section_name, section_cls in section_map.items():
        if section_name in raw:
            setattr(config, section_name, section_cls(**raw[section_name]))

    return config
