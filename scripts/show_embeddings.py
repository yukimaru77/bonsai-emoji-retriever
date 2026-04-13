#!/usr/bin/env python3
"""Debug script: show cosine similarity matrix for training data."""

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from src.config import load_config
from src.model import BonsaiEmbedder
from src.preprocess import normalize_text


def main() -> None:
    parser = argparse.ArgumentParser(description="Show embedding similarity matrix")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--adapter", type=str, default="outputs/",
                        help="Path to saved LoRA adapter (or 'none' for base model)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

    config = load_config(args.config)
    project_root = Path(__file__).resolve().parent.parent

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    logging.info("Loading model...")
    model = BonsaiEmbedder(config)

    adapter_path = project_root / args.adapter
    if args.adapter != "none" and (adapter_path / "adapter_model.safetensors").exists():
        logging.info("Loading adapter from %s", adapter_path)
        model.base_model.load_adapter(str(adapter_path), adapter_name="default")
    else:
        logging.info("Using base model (no adapter loaded)")

    model.to(device)
    model.eval()

    # Load data
    data_path = project_root / config.data.train_path
    catalog_path = project_root / config.data.emoji_catalog_path

    with open(data_path) as f:
        samples = [json.loads(line) for line in f if line.strip()]
    with open(catalog_path) as f:
        catalog = json.load(f)
    emoji_to_label = {e["emoji_id"]: e["label_text"] for e in catalog}

    queries = [normalize_text(s["query"]) for s in samples]
    label_ids = [s["positive_label"] for s in samples]
    label_texts = [emoji_to_label[eid] for eid in label_ids]

    # Encode
    with torch.no_grad():
        q_embs = model.encode_queries(queries)
        l_embs = model.encode_labels(label_texts)

    # Similarity matrix
    sim_matrix = q_embs @ l_embs.T

    print("\n=== Cosine Similarity Matrix ===")
    print(f"{'':>40s}", end="")
    for eid in label_ids:
        print(f"  {eid:>20s}", end="")
    print()

    for i, q in enumerate(queries):
        display_q = q[:37] + "..." if len(q) > 40 else q
        print(f"{display_q:>40s}", end="")
        for j in range(len(label_ids)):
            val = sim_matrix[i, j].item()
            marker = " *" if i == j else "  "
            print(f"  {val:>18.4f}{marker}", end="")
        print()

    print("\n(* = expected positive pair)")


if __name__ == "__main__":
    main()
