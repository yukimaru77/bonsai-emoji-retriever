"""Training loop with Extended InfoNCE, MRL, and hard negatives."""

from __future__ import annotations

import logging
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader

from src.config import Config
from src.dataset import EmojiRetrievalDataset, EmojiCollator
from src.loss import ExtendedInfoNCELoss
from src.model import BonsaiEmbedder

logger = logging.getLogger(__name__)


def _compute_mrl_loss(
    loss_fn: ExtendedInfoNCELoss,
    query_embs: torch.Tensor,
    pos_label_embs: torch.Tensor,
    hard_neg_embs: torch.Tensor | None,
    mrl_dims: list[int],
) -> torch.Tensor:
    """Compute Matryoshka Representation Learning loss.

    For each target dimension, truncate embeddings, re-normalize,
    and compute Extended InfoNCE. Return the mean across dimensions.
    """
    total_loss = torch.tensor(0.0, device=query_embs.device)

    for dim in mrl_dims:
        q_trunc = F.normalize(query_embs[:, :dim], p=2, dim=-1)
        l_trunc = F.normalize(pos_label_embs[:, :dim], p=2, dim=-1)

        hn_trunc = None
        if hard_neg_embs is not None:
            hn_trunc = F.normalize(hard_neg_embs[:, :, :dim], p=2, dim=-1)

        total_loss = total_loss + loss_fn(q_trunc, l_trunc, hn_trunc)

    return total_loss / len(mrl_dims)


def train(config: Config) -> None:
    """Run the training loop."""
    torch.manual_seed(config.training.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    # --- Model ---
    logger.info("Loading model: %s", config.model.model_id)
    model = BonsaiEmbedder(config)
    model.base_model.print_trainable_parameters()
    model.to(device)

    # --- Loss ---
    loss_fn = ExtendedInfoNCELoss(
        init_temperature=config.loss.init_temperature,
        fn_margin=config.loss.fn_margin,
    )
    loss_fn.to(device)

    # --- Data ---
    project_root = Path(__file__).resolve().parent.parent
    dataset = EmojiRetrievalDataset(
        data_path=str(project_root / config.data.train_path),
        emoji_catalog_path=str(project_root / config.data.emoji_catalog_path),
    )
    collator = EmojiCollator(
        tokenizer=model.tokenizer,
        max_length=config.model.max_length,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        collate_fn=collator,
        drop_last=False,
    )

    # --- Optimizer ---
    lora_params = [p for p in model.parameters() if p.requires_grad]
    temp_params = [loss_fn.log_temperature]

    optimizer = AdamW(
        [
            {
                "params": lora_params,
                "lr": config.training.learning_rate,
                "weight_decay": config.training.weight_decay,
            },
            {
                "params": temp_params,
                "lr": config.loss.temperature_lr,
                "weight_decay": 0.0,
            },
        ],
    )

    # MRL config
    mrl_enabled = config.mrl.enabled
    mrl_dims = config.mrl.dims
    if mrl_enabled:
        logger.info("MRL enabled with dims: %s", mrl_dims)

    # --- Training ---
    model.train()
    loss_fn.train()
    global_step = 0

    for epoch in range(config.training.num_epochs):
        logger.info("=== Epoch %d/%d ===", epoch + 1, config.training.num_epochs)

        for batch_idx, batch in enumerate(dataloader):
            batch = {k: v.to(device) for k, v in batch.items()}

            # Encode queries and positive labels
            query_embs = model(
                batch["query_input_ids"], batch["query_attention_mask"]
            )
            label_embs = model(
                batch["label_input_ids"], batch["label_attention_mask"]
            )

            # Encode hard negatives if present
            hard_neg_embs = None
            if "neg_input_ids" in batch:
                hard_neg_embs = model.encode_batch(
                    batch["neg_input_ids"], batch["neg_attention_mask"]
                )
                # Apply valid mask: zero out padded negatives
                neg_mask = batch["neg_valid_mask"].unsqueeze(2).to(device)
                hard_neg_embs = hard_neg_embs * neg_mask

            # Compute loss
            if mrl_enabled:
                loss = _compute_mrl_loss(
                    loss_fn, query_embs, label_embs, hard_neg_embs, mrl_dims
                )
            else:
                loss = loss_fn(query_embs, label_embs, hard_neg_embs)

            loss.backward()

            grad_norm = torch.nn.utils.clip_grad_norm_(
                list(model.parameters()) + [loss_fn.log_temperature],
                config.training.max_grad_norm,
            )

            optimizer.step()
            optimizer.zero_grad()

            global_step += 1
            logger.info(
                "step=%d  loss=%.4f  temp=%.4f  grad_norm=%.4f",
                global_step,
                loss.item(),
                loss_fn.temperature.item(),
                grad_norm.item() if hasattr(grad_norm, "item") else grad_norm,
            )

    # --- Save ---
    save_dir = str(project_root / config.output.save_dir)
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    model.save_adapter(save_dir)
    torch.save(
        {"log_temperature": loss_fn.log_temperature.data},
        Path(save_dir) / "loss_state.pt",
    )
    logger.info("Saved adapter and loss state to %s", save_dir)
