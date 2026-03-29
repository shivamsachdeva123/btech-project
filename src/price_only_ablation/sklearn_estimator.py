from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import mean_squared_error
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from price_only_ablation.model import PriceOnlyLateFusionRegressorNet


@dataclass(frozen=True)
class InputSchema:
    window_size: int
    price_feature_dim: int


class PriceOnlyLateFusionEstimator(BaseEstimator, RegressorMixin):
    def __init__(
        self,
        window_size: int,
        company_vocab_size: int,
        price_feature_dim: int = 1,
        price_hidden_dim: int = 32,
        price_num_layers: int = 1,
        lstm_dropout: float = 0.0,
        company_emb_dim: int = 16,
        ann_hidden_dim: int = 64,
        dropout: float = 0.2,
        learning_rate: float = 1e-3,
        optimizer_name: str = "adam",
        weight_decay: float = 0.0,
        batch_size: int = 64,
        epochs: int = 20,
        device: str = "auto",
        random_state: int = 42,
        verbose: bool = False,
    ) -> None:
        self.window_size = window_size
        self.company_vocab_size = company_vocab_size
        self.price_feature_dim = price_feature_dim
        self.price_hidden_dim = price_hidden_dim
        self.price_num_layers = price_num_layers
        self.lstm_dropout = lstm_dropout
        self.company_emb_dim = company_emb_dim
        self.ann_hidden_dim = ann_hidden_dim
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.optimizer_name = optimizer_name
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.epochs = epochs
        self.device = device
        self.random_state = random_state
        self.verbose = verbose

        self.schema_ = InputSchema(
            window_size=window_size,
            price_feature_dim=price_feature_dim,
        )
        self.model_: PriceOnlyLateFusionRegressorNet | None = None

    def _resolve_device(self) -> torch.device:
        pref = str(self.device).lower().strip()
        if pref in {"cuda", "gpu"} and torch.cuda.is_available():
            return torch.device("cuda")
        if pref in {"mps", "apple"} and torch.backends.mps.is_available():
            return torch.device("mps")
        if pref == "cpu":
            return torch.device("cpu")
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def _decode_flat_features(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        n_samples = X.shape[0]
        p_count = self.window_size * self.price_feature_dim

        price_flat = X[:, :p_count]
        company_flat = X[:, p_count]

        price_seq = price_flat.reshape(n_samples, self.window_size, self.price_feature_dim).astype(np.float32)
        company_id = company_flat.astype(np.int64).reshape(-1, 1)
        return price_seq, company_id

    def _build_model(self) -> PriceOnlyLateFusionRegressorNet:
        torch.manual_seed(self.random_state)
        model = PriceOnlyLateFusionRegressorNet(
            company_vocab_size=self.company_vocab_size,
            price_input_dim=self.price_feature_dim,
            price_hidden_dim=self.price_hidden_dim,
            price_num_layers=self.price_num_layers,
            lstm_dropout=self.lstm_dropout,
            company_emb_dim=self.company_emb_dim,
            ann_hidden_dim=self.ann_hidden_dim,
            dropout=self.dropout,
        )
        return model

    def _build_optimizer(self) -> torch.optim.Optimizer:
        if self.model_ is None:
            raise RuntimeError("Model must be created before optimizer initialization")

        name = str(self.optimizer_name).lower()
        params = self.model_.parameters()
        if name == "adam":
            return torch.optim.Adam(params, lr=self.learning_rate, weight_decay=self.weight_decay)
        if name == "adamw":
            return torch.optim.AdamW(params, lr=self.learning_rate, weight_decay=self.weight_decay)
        if name == "rmsprop":
            return torch.optim.RMSprop(params, lr=self.learning_rate, weight_decay=self.weight_decay)

        raise ValueError(f"Unsupported optimizer_name={self.optimizer_name}")

    def fit(self, X: np.ndarray, y: np.ndarray) -> "PriceOnlyLateFusionEstimator":
        X = np.asarray(X)
        y = np.asarray(y, dtype=np.float32)

        price_seq, company_id = self._decode_flat_features(X)
        dataset = TensorDataset(
            torch.from_numpy(price_seq),
            torch.from_numpy(company_id),
            torch.from_numpy(y),
        )
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.model_ = self._build_model()
        device = self._resolve_device()
        self.model_ = self.model_.to(device)
        optimizer = self._build_optimizer()
        criterion = nn.MSELoss()

        self.model_.train()
        for _ in range(self.epochs):
            for price_b, company_b, y_b in loader:
                price_b = price_b.to(device)
                company_b = company_b.to(device)
                y_b = y_b.to(device)
                optimizer.zero_grad()
                pred = self.model_(price_b, company_b)
                loss = criterion(pred, y_b)
                loss.backward()
                optimizer.step()

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.model_ is None:
            raise RuntimeError("Model is not trained. Call fit() first.")

        X = np.asarray(X)
        price_seq, company_id = self._decode_flat_features(X)
        device = self._resolve_device()

        self.model_.eval()
        with torch.inference_mode():
            preds = self.model_(
                torch.from_numpy(price_seq).to(device),
                torch.from_numpy(company_id).to(device),
            )
        return preds.cpu().numpy()

    def score(self, X: Any, y: Any, sample_weight: Any = None) -> float:
        y = np.asarray(y, dtype=np.float32)
        pred = self.predict(X)
        rmse = np.sqrt(mean_squared_error(y, pred, sample_weight=sample_weight))
        return -float(rmse)

    def save_model(self, output_path: Path) -> None:
        if self.model_ is None:
            raise RuntimeError("Model is not trained. Nothing to save.")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model_.state_dict(), output_path)
