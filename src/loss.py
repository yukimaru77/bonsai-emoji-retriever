"""Extended InfoNCE loss with false negative mask (Qwen3 Embedding, arXiv 2506.05176v3)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class ExtendedInfoNCELoss(nn.Module):
    """Contrastive loss following Qwen3 Embedding paper.

    Extends standard InfoNCE with:
    - In-batch query-query, doc-doc, and cross query-doc negatives
    - Hard negatives per query
    - False negative mitigation mask (margin-based)
    - Learnable temperature in log-space
    """

    def __init__(self, init_temperature: float = 0.05, fn_margin: float = 0.1):
        super().__init__()
        self.log_temperature = nn.Parameter(
            torch.log(torch.tensor(init_temperature))
        )
        self.fn_margin = fn_margin

    @property
    def temperature(self) -> Tensor:
        return torch.clamp(self.log_temperature.exp(), min=0.01, max=1.0)

    def _build_fn_mask(
        self,
        sim_values: Tensor,
        pos_sim: Tensor,
    ) -> Tensor:
        """Build false negative mask.

        Masks out negatives whose similarity exceeds the positive + margin,
        as they are likely false negatives.

        Args:
            sim_values: (B, N) similarity of query_i to each negative candidate
            pos_sim: (B,) similarity of query_i to its positive

        Returns:
            (B, N) binary mask, 1 = keep as negative, 0 = mask out
        """
        threshold = pos_sim.unsqueeze(1) + self.fn_margin
        mask = (sim_values <= threshold).float()
        return mask

    def forward(
        self,
        query_embs: Tensor,
        pos_label_embs: Tensor,
        hard_neg_embs: Tensor | None = None,
    ) -> Tensor:
        """Compute Extended InfoNCE loss.

        Args:
            query_embs: (B, D) L2-normalized query embeddings
            pos_label_embs: (B, D) L2-normalized positive label embeddings
            hard_neg_embs: (B, K, D) L2-normalized hard negative embeddings, or None

        Returns:
            Scalar loss.
        """
        # Cast to float32 for numerical stability (embeddings may be float16)
        query_embs = query_embs.float()
        pos_label_embs = pos_label_embs.float()
        if hard_neg_embs is not None:
            hard_neg_embs = hard_neg_embs.float()

        B = query_embs.size(0)
        tau = self.temperature

        # Positive similarities: (B,)
        pos_sim = (query_embs * pos_label_embs).sum(dim=-1) / tau

        # === In-batch negatives ===

        # Query-document cross: (B, B) — q_i vs d_j for j != i
        qd_sim = query_embs @ pos_label_embs.T / tau
        # Query-query: (B, B) — q_i vs q_j for j != i
        qq_sim = query_embs @ query_embs.T / tau
        # Doc-doc: (B, B) — d_i vs d_j for j != i
        dd_sim = pos_label_embs @ pos_label_embs.T / tau

        # Self-exclusion mask (diagonal = 0, off-diagonal = 1)
        diag_mask = 1.0 - torch.eye(B, device=query_embs.device)

        # False negative masks for each in-batch term
        qd_fn_mask = self._build_fn_mask(qd_sim * tau, pos_sim * tau)
        qq_fn_mask = self._build_fn_mask(qq_sim * tau, pos_sim * tau)
        dd_fn_mask = self._build_fn_mask(dd_sim * tau, pos_sim * tau)

        # Combine masks: self-exclusion AND false negative
        qd_mask = diag_mask * qd_fn_mask
        qq_mask = diag_mask * qq_fn_mask
        dd_mask = diag_mask * dd_fn_mask

        # Masked exp-sims
        qd_neg = (torch.exp(qd_sim) * qd_mask).sum(dim=1)
        qq_neg = (torch.exp(qq_sim) * qq_mask).sum(dim=1)
        dd_neg = (torch.exp(dd_sim) * dd_mask).sum(dim=1)

        # === Hard negatives ===
        hn_neg = torch.zeros(B, device=query_embs.device)
        if hard_neg_embs is not None and hard_neg_embs.size(1) > 0:
            # hard_neg_embs: (B, K, D)
            # Similarity: (B, K)
            hn_sim = torch.bmm(
                hard_neg_embs, query_embs.unsqueeze(2)
            ).squeeze(2) / tau
            hn_fn_mask = self._build_fn_mask(hn_sim * tau, pos_sim * tau)
            hn_neg = (torch.exp(hn_sim) * hn_fn_mask).sum(dim=1)

        # === Denominator ===
        pos_exp = torch.exp(pos_sim)
        denominator = pos_exp + qd_neg + qq_neg + dd_neg + hn_neg

        # === Loss ===
        loss = -torch.log(pos_exp / denominator).mean()

        return loss
