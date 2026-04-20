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

from hybrid_model.late_fusion_model import HybridLateFusionRegressorNet


@dataclass(frozen=True)
class InputSchema:
    window_size: int
    price_feature_dim: int
    sentiment_feature_dim: int


class HybridLateFusionEstimator(BaseEstimator, RegressorMixin):
    def __init__(
        self,
        window_size: int,
        company_vocab_size: int,
        sentiment_feature_dim: int = 6,
        price_feature_dim: int = 1,
        price_hidden_dim: int = 32,
        sentiment_hidden_dim: int = 32,
        price_num_layers: int = 1,
        sentiment_num_layers: int = 1,
        lstm_dropout: float = 0.0,
        sentiment_dropout: float = 0.5,
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
        sentiment_strength_index: int = 3,
    ) -> None:
        self.window_size = window_size
        self.company_vocab_size = company_vocab_size
        self.sentiment_feature_dim = sentiment_feature_dim
        self.price_feature_dim = price_feature_dim
        self.price_hidden_dim = price_hidden_dim
        self.sentiment_hidden_dim = sentiment_hidden_dim
        self.price_num_layers = price_num_layers
        self.sentiment_num_layers = sentiment_num_layers
        self.lstm_dropout = lstm_dropout
        self.sentiment_dropout = sentiment_dropout
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
        self.sentiment_strength_index = sentiment_strength_index

        self.schema_ = InputSchema(
            window_size=window_size,
            price_feature_dim=price_feature_dim,
            sentiment_feature_dim=sentiment_feature_dim,
        )
        self.model_: HybridLateFusionRegressorNet | None = None
        self.price_mean_: np.ndarray | None = None
        self.price_std_: np.ndarray | None = None
        self.sent_mean_: np.ndarray | None = None
        self.sent_std_: np.ndarray | None = None
        self.scaling_checks_: dict[str, float] | None = None
        self.last_loss_components_: dict[str, float] | None = None

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

    def _decode_flat_features(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        n_samples = X.shape[0]
        p_count = self.window_size * self.price_feature_dim
        s_count = self.window_size * self.sentiment_feature_dim
        expected_feature_count = p_count + s_count + 1

        if X.ndim != 2:
            raise ValueError(f"Expected 2D array X, got shape={X.shape}")
        if X.shape[1] != expected_feature_count:
            raise ValueError(
                "Hybrid input feature size mismatch: "
                f"expected {expected_feature_count} (= {self.window_size}*{self.price_feature_dim} "
                f"+ {self.window_size}*{self.sentiment_feature_dim} + 1 company_id), "
                f"got {X.shape[1]}. Rebuild model input NPZ with current window settings."
            )

        price_flat = X[:, :p_count]
        sent_flat = X[:, p_count : p_count + s_count]
        company_flat = X[:, p_count + s_count]

        price_seq = price_flat.reshape(n_samples, self.window_size, self.price_feature_dim).astype(np.float32)
        sent_seq = sent_flat.reshape(n_samples, self.window_size, self.sentiment_feature_dim).astype(np.float32)
        company_id = company_flat.astype(np.int64).reshape(-1, 1)
        return price_seq, sent_seq, company_id

    def _build_model(self) -> HybridLateFusionRegressorNet:
        torch.manual_seed(self.random_state)
        model = HybridLateFusionRegressorNet(
            company_vocab_size=self.company_vocab_size,
            price_input_dim=self.price_feature_dim,
            sentiment_input_dim=self.sentiment_feature_dim,
            price_hidden_dim=self.price_hidden_dim,
            sentiment_hidden_dim=self.sentiment_hidden_dim,
            price_num_layers=self.price_num_layers,
            sentiment_num_layers=self.sentiment_num_layers,
            lstm_dropout=self.lstm_dropout,
            sentiment_dropout=self.sentiment_dropout,
            company_emb_dim=self.company_emb_dim,
            ann_hidden_dim=self.ann_hidden_dim,
            dropout=self.dropout,
            sentiment_strength_index=self.sentiment_strength_index,
        )
        return model

    def _fit_feature_scalers(self, price_seq: np.ndarray, sent_seq: np.ndarray) -> None:
        # Global scaling over all train samples and timesteps.
        price_flat = price_seq.reshape(-1, self.price_feature_dim)
        sent_flat = sent_seq.reshape(-1, self.sentiment_feature_dim)

        price_mean = np.mean(price_flat, axis=0, keepdims=True).astype(np.float32)
        price_std = np.std(price_flat, axis=0, keepdims=True).astype(np.float32)
        price_std = np.where(price_std > 1e-8, price_std, 1.0).astype(np.float32)

        sent_mean = np.mean(sent_flat, axis=0, keepdims=True).astype(np.float32)
        sent_std = np.std(sent_flat, axis=0, keepdims=True).astype(np.float32)
        sent_std = np.where(sent_std > 1e-8, sent_std, 1.0).astype(np.float32)

        self.price_mean_ = price_mean
        self.price_std_ = price_std
        self.sent_mean_ = sent_mean
        self.sent_std_ = sent_std

    def _transform_features(self, price_seq: np.ndarray, sent_seq: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.price_mean_ is None or self.price_std_ is None or self.sent_mean_ is None or self.sent_std_ is None:
            raise RuntimeError("Feature scalers are not fitted. Call fit() before predict().")

        price_scaled = (price_seq - self.price_mean_.reshape(1, 1, -1)) / self.price_std_.reshape(1, 1, -1)
        sent_scaled = (sent_seq - self.sent_mean_.reshape(1, 1, -1)) / self.sent_std_.reshape(1, 1, -1)
        return price_scaled.astype(np.float32), sent_scaled.astype(np.float32)

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

    def _split_targets(self, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        y = np.asarray(y, dtype=np.float32)
        if y.ndim == 2 and y.shape[1] >= 3:
            y_return = y[:, 0].astype(np.float32)
            y_volatility = y[:, 1].astype(np.float32)
            y_direction = y[:, 2].astype(np.float32)
            return y_return, y_direction, y_volatility

        raise ValueError(
            "Expected y as shape (n_samples, 3+) with "
            "columns [target_log_return, volatility, direction]."
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> "HybridLateFusionEstimator":
        X = np.asarray(X)
        y_return, y_direction, y_volatility = self._split_targets(y)

        price_seq_raw, sent_seq_raw, company_id = self._decode_flat_features(X)
        self._fit_feature_scalers(price_seq_raw, sent_seq_raw)
        price_seq, sent_seq = self._transform_features(price_seq_raw, sent_seq_raw)

        if (
            np.isnan(price_seq).any()
            or np.isnan(sent_seq).any()
            or np.isnan(y_return).any()
            or np.isnan(y_direction).any()
            or np.isnan(y_volatility).any()
        ):
            raise ValueError("NaNs detected after scaling in hybrid estimator fit().")

        train_price_mean = float(np.mean(price_seq))
        train_price_std = float(np.std(price_seq))
        train_sent_mean = float(np.mean(sent_seq))
        train_sent_std = float(np.std(sent_seq))
        self.scaling_checks_ = {
            "train_price_mean": train_price_mean,
            "train_price_std": train_price_std,
            "train_sent_mean": train_sent_mean,
            "train_sent_std": train_sent_std,
        }

        if self.verbose:
            print(
                "Hybrid scaling checks: "
                f"price_mean={train_price_mean:.6f}, price_std={train_price_std:.6f}, "
                f"sent_mean={train_sent_mean:.6f}, sent_std={train_sent_std:.6f}"
            )

        dataset = TensorDataset(
            torch.from_numpy(price_seq),
            torch.from_numpy(sent_seq),
            torch.from_numpy(company_id),
            torch.from_numpy(y_return),
            torch.from_numpy(y_direction),
            torch.from_numpy(y_volatility),
        )
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.model_ = self._build_model()
        device = self._resolve_device()
        self.model_ = self.model_.to(device)
        optimizer = self._build_optimizer()
        mse = nn.MSELoss()
        bce_logits = nn.BCEWithLogitsLoss()

        self.model_.train()
        last_return_loss = 0.0
        last_direction_loss = 0.0
        last_volatility_loss = 0.0
        for _ in range(self.epochs):
            for price_b, sent_b, company_b, y_return_b, y_direction_b, y_volatility_b in loader:
                price_b = price_b.to(device)
                sent_b = sent_b.to(device)
                company_b = company_b.to(device)
                y_return_b = y_return_b.to(device)
                y_direction_b = y_direction_b.to(device)
                y_volatility_b = y_volatility_b.to(device)

                optimizer.zero_grad()
                pred = self.model_(price_b, sent_b, company_b)
                return_pred = pred[:, 0]
                volatility_pred = torch.relu(pred[:, 1])
                direction_logit = pred[:, 2]

                return_loss = mse(return_pred, y_return_b)
                direction_loss = bce_logits(direction_logit, y_direction_b)
                volatility_loss = mse(volatility_pred, y_volatility_b)
                loss = return_loss + direction_loss + volatility_loss
                loss.backward()
                optimizer.step()

                last_return_loss = float(return_loss.detach().cpu().item())
                last_direction_loss = float(direction_loss.detach().cpu().item())
                last_volatility_loss = float(volatility_loss.detach().cpu().item())

        self.last_loss_components_ = {
            "return_loss": last_return_loss,
            "direction_loss": last_direction_loss,
            "volatility_loss": last_volatility_loss,
        }

        if self.verbose:
            print(
                "Hybrid final batch losses: "
                f"return={last_return_loss:.6f}, "
                f"direction={last_direction_loss:.6f}, "
                f"volatility={last_volatility_loss:.6f}"
            )

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.model_ is None:
            raise RuntimeError("Model is not trained. Call fit() first.")

        X = np.asarray(X)
        price_seq_raw, sent_seq_raw, company_id = self._decode_flat_features(X)
        price_seq, sent_seq = self._transform_features(price_seq_raw, sent_seq_raw)
        device = self._resolve_device()

        self.model_.eval()
        with torch.inference_mode():
            preds = self.model_(
                torch.from_numpy(price_seq).to(device),
                torch.from_numpy(sent_seq).to(device),
                torch.from_numpy(company_id).to(device),
            )
        preds_np = preds.cpu().numpy().astype(np.float32)

        # Output order: [return_pred, volatility_pred, direction_prob]
        preds_np[:, 1] = np.maximum(preds_np[:, 1], 0.0)
        preds_np[:, 2] = 1.0 / (1.0 + np.exp(-preds_np[:, 2]))
        return preds_np

    def score(self, X: Any, y: Any, sample_weight: Any = None) -> float:
        y = np.asarray(y, dtype=np.float32)
        if y.ndim == 2:
            y_return = y[:, 0]
        else:
            y_return = y
        pred = self.predict(X)
        pred_return = pred[:, 0] if pred.ndim == 2 else pred
        rmse = np.sqrt(mean_squared_error(y_return, pred_return, sample_weight=sample_weight))
        return -float(rmse)

    def save_model(self, output_path: Path) -> None:
        if self.model_ is None:
            raise RuntimeError("Model is not trained. Nothing to save.")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model_.state_dict(), output_path)
