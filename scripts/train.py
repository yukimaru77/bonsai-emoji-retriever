#!/usr/bin/env python3
"""Entry point for training."""

import argparse
import logging
import sys
from pathlib import Path

# Allow running from project root: `python scripts/train.py`
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config
from src.trainer import train


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Bonsai Emoji Retriever")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to YAML config file",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    config = load_config(args.config)
    train(config)


if __name__ == "__main__":
    main()
