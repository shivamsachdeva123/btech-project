from __future__ import annotations

import json
import ast
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
    if isinstance(value, np.ndarray):
        return value.tolist()
    if not isinstance(value, str):
        raise ValueError(f"Expected JSON string or list, got type={type(value)}")
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return ast.literal_eval(value)


def build_flattened_price_only_input(
    aligned_df: pd.DataFrame,
    price_col: str = "return_window",
    company_col: str = "company_id",
    target_col: str = "target_log_return",
    normalize_price_per_company: bool = True,
) -> PriceOnlyInputArrays:
    if aligned_df.empty:
        raise ValueError("Aligned dataframe is empty")
    _ = normalize_price_per_company

    if price_col not in aligned_df.columns:
        # Backward compatibility for older aligned datasets.
        price_col = "close_window"

    price_windows = [_parse_json_list(v) for v in aligned_df[price_col].tolist()]

    window_size = len(price_windows[0])
    for i, window in enumerate(price_windows):
        if len(window) != window_size:
            raise ValueError(f"Inconsistent price window length at row {i}: {len(window)} != {window_size}")

    # Preserve raw windowed values. Fold-aware normalization is handled inside estimator.fit.
    raw_price_arr = np.asarray(price_windows, dtype=np.float32).reshape(-1, window_size)
    price_arr = raw_price_arr.reshape(-1, window_size, 1)
    target_return_arr = aligned_df[target_col].astype(float).to_numpy(dtype=np.float32)
    if "direction" not in aligned_df.columns or "volatility" not in aligned_df.columns:
        raise ValueError("Aligned dataframe must include explicit 'direction' and 'volatility' columns.")
    target_direction_arr = aligned_df["direction"].astype(float).to_numpy(dtype=np.float32)
    target_volatility_arr = aligned_df["volatility"].astype(float).to_numpy(dtype=np.float32)
    target_arr = np.column_stack([target_return_arr, target_volatility_arr, target_direction_arr]).astype(np.float32)
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
