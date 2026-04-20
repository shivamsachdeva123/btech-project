from __future__ import annotations

import torch
from torch import nn


class PriceOnlyLateFusionRegressorNet(nn.Module):
    """Price-only late-fusion model for fair ablation.

    Keeps the same design philosophy as hybrid model:
    - Price LSTM branch
    - Company embedding branch
    - Shared ANN backbone + task-specific heads
    """

    def __init__(
        self,
        company_vocab_size: int,
        price_input_dim: int = 1,
        price_hidden_dim: int = 32,
        price_num_layers: int = 1,
        lstm_dropout: float = 0.0,
        company_emb_dim: int = 16,
        ann_hidden_dim: int = 64,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        price_lstm_dropout = lstm_dropout if price_num_layers > 1 else 0.0
        self.price_lstm = nn.LSTM(
            input_size=price_input_dim,
            hidden_size=price_hidden_dim,
            batch_first=True,
            num_layers=price_num_layers,
            dropout=price_lstm_dropout,
        )
        self.company_embedding = nn.Embedding(company_vocab_size, company_emb_dim)

        fusion_dim = price_hidden_dim + company_emb_dim
        self.ann_head = nn.Sequential(
            nn.Linear(fusion_dim, ann_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ann_hidden_dim, ann_hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        head_dim = ann_hidden_dim // 2
        self.return_head = nn.Linear(head_dim, 1)
        self.direction_head = nn.Linear(head_dim, 1)
        self.volatility_head = nn.Linear(head_dim, 1)

    def forward(self, price_seq: torch.Tensor, company_id: torch.Tensor) -> torch.Tensor:
        _, (price_h_n, _) = self.price_lstm(price_seq)

        price_repr = price_h_n[-1]
        company_repr = self.company_embedding(company_id).squeeze(1)

        fused = torch.cat([price_repr, company_repr], dim=1)
        shared = self.ann_head(fused)
        # Output order: [return_pred, volatility_raw, direction_logit]
        return torch.cat(
            [
                self.return_head(shared),
                self.volatility_head(shared),
                self.direction_head(shared),
            ],
            dim=1,
        )
