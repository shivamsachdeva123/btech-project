from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PriceOnlyInputArrays:
    X: np.ndarray
    y: np.ndarray
    previous_close: np.ndarray
    target_close: np.ndarray
    window_size: int
    price_feature_dim: int


def _parse_json_list(value: str) -> list:
    if isinstance(value, list):
        return value
    if not isinstance(value, str):
        raise ValueError(f"Expected JSON string or list, got type={type(value)}")
    return json.loads(value)


def build_flattened_price_only_input(
    aligned_df: pd.DataFrame,
    price_col: str = "close_window",
    company_col: str = "company_id",
    target_col: str = "target_log_return",
    normalize_price_per_company: bool = True,
) -> PriceOnlyInputArrays:
    if aligned_df.empty:
        raise ValueError("Aligned dataframe is empty")

    price_windows = [_parse_json_list(v) for v in aligned_df[price_col].tolist()]

    window_size = len(price_windows[0])
    for i, window in enumerate(price_windows):
        if len(window) != window_size:
            raise ValueError(f"Inconsistent price window length at row {i}: {len(window)} != {window_size}")

    raw_price_arr = np.asarray(price_windows, dtype=np.float32).reshape(-1, window_size)
    tickers = aligned_df["ticker"].astype(str).tolist()

    if normalize_price_per_company:
        ticker_stats: dict[str, tuple[float, float]] = {}
        for ticker in sorted(set(tickers)):
            ticker_windows = raw_price_arr[np.array(tickers) == ticker]
            ticker_values = ticker_windows.reshape(-1)
            mean = float(np.mean(ticker_values))
            std = float(np.std(ticker_values))
            ticker_stats[ticker] = (mean, std if std > 1e-8 else 1.0)

        normalized = np.empty_like(raw_price_arr)
        for i, ticker in enumerate(tickers):
            mean, std = ticker_stats[ticker]
            normalized[i] = (raw_price_arr[i] - mean) / std
        price_arr = normalized.reshape(-1, window_size, 1)

        target_raw = aligned_df[target_col].astype(float).to_numpy(dtype=np.float32)
        target_arr = np.empty_like(target_raw)
        for i, ticker in enumerate(tickers):
            mean, std = ticker_stats[ticker]
            target_arr[i] = (target_raw[i] - mean) / std

        target_close_raw = aligned_df["target_close"].astype(float).to_numpy(dtype=np.float32)
        target_close_arr = np.empty_like(target_close_raw)
        for i, ticker in enumerate(tickers):
            mean, std = ticker_stats[ticker]
            target_close_arr[i] = (target_close_raw[i] - mean) / std
    else:
        price_arr = raw_price_arr.reshape(-1, window_size, 1)
        target_arr = aligned_df[target_col].astype(float).to_numpy(dtype=np.float32)
        target_close_arr = aligned_df["target_close"].astype(float).to_numpy(dtype=np.float32)

    company_arr = aligned_df[company_col].astype(int).to_numpy(dtype=np.int64).reshape(-1, 1)
    previous_close_arr = aligned_df["previous_close"].astype(float).to_numpy(dtype=np.float32)

    X_price_flat = price_arr.reshape(len(price_arr), -1)
    X = np.concatenate([X_price_flat, company_arr.astype(np.float32)], axis=1)

    return PriceOnlyInputArrays(
        X=X,
        y=target_arr,
        previous_close=previous_close_arr,
        target_close=target_close_arr,
        window_size=window_size,
        price_feature_dim=1,
    )


def build_flattened_price_only_input_from_csv(
    aligned_csv_path: Path,
    target_col: str = "target_log_return",
    normalize_price_per_company: bool = True,
) -> PriceOnlyInputArrays:
    df = pd.read_csv(aligned_csv_path)
    return build_flattened_price_only_input(
        aligned_df=df,
        target_col=target_col,
        normalize_price_per_company=normalize_price_per_company,
    )


def save_price_only_input_npz(arrays: PriceOnlyInputArrays, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        X=arrays.X,
        y=arrays.y,
        previous_close=arrays.previous_close,
        target_close=arrays.target_close,
        window_size=np.int64(arrays.window_size),
        price_feature_dim=np.int64(arrays.price_feature_dim),
    )
