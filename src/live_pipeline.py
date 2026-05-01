from __future__ import annotations

import json
import logging
import sys
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from data_pipeline.collectors.price_collector import collect_price_data
from data_pipeline.processors.alignment import align_modalities
from data_pipeline.processors.news_price_alignment import build_daily_news_price_alignment
from data_pipeline.processors.sentiment_aggregator import build_daily_sentiment
from data_pipeline.utils.logger import build_logger
from hybrid_data_prep.build_model_input import build_flattened_hybrid_input
from hybrid_model.sklearn_estimator import HybridLateFusionEstimator


@dataclass(frozen=True)
class LiveRunArtifacts:
    live_prices_path: Path
    live_predictions_path: Path
    predictions: dict[str, dict[str, float | int]]


def _load_config(config_path: Path) -> dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def _today_utc_date() -> date:
    return datetime.now(timezone.utc).date()


def _fetch_finnhub_company_news(
    ticker: str,
    from_date: date,
    to_date: date,
    api_key: str,
    logger: logging.Logger,
) -> list[dict[str, Any]]:
    url = "https://finnhub.io/api/v1/company-news"
    params = {
        "symbol": ticker,
        "from": from_date.isoformat(),
        "to": to_date.isoformat(),
        "token": api_key,
    }
    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, list):
        logger.warning("Unexpected Finnhub payload for %s: %s", ticker, type(payload))
        return []
    return payload


def _news_to_pipeline_schema(
    ticker: str,
    records: list[dict[str, Any]],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for item in records:
        ts = item.get("datetime")
        if ts is None:
            continue
        try:
            published_date = datetime.fromtimestamp(int(ts), tz=timezone.utc).date()
        except (ValueError, OSError, TypeError):
            continue
        headline = str(item.get("headline", "") or "").strip()
        summary = str(item.get("summary", "") or "").strip()
        news_text = f"{headline} {summary}".strip()
        rows.append(
            {
                "ticker": ticker.upper(),
                "published_date": published_date,
                "news_text": news_text,
            }
        )
    if not rows:
        return pd.DataFrame(columns=["ticker", "published_date", "news_text"])
    return pd.DataFrame(rows)


def _load_company_map(config: dict[str, Any], logger: logging.Logger) -> pd.DataFrame:
    company_map_path = PROJECT_ROOT / config["paths"]["company_map"]
    if not company_map_path.exists():
        raise FileNotFoundError(
            f"Company map not found at {company_map_path}. "
            "Run the existing data pipeline first to generate it."
        )
    company_map_df = pd.read_csv(company_map_path)
    company_map_df["ticker"] = company_map_df["ticker"].astype(str).str.upper().str.strip()
    company_map_df["company_id"] = company_map_df["company_id"].astype(int)
    logger.info("Loaded company map: %s rows", len(company_map_df))
    return company_map_df


def _build_estimator_with_trained_weights(
    config: dict[str, Any],
    company_vocab_size: int,
    window_size: int,
    price_feature_dim: int,
    sentiment_feature_dim: int,
) -> HybridLateFusionEstimator:
    artifact_dir = PROJECT_ROOT / config["paths"]["trained_model_dir"]
    best_params_path = artifact_dir / "hybrid_late_fusion_best_params.json"
    model_path = artifact_dir / "hybrid_late_fusion_best.pt"
    model_input_npz_path = PROJECT_ROOT / config["paths"]["model_input_npz"]

    if not best_params_path.exists():
        raise FileNotFoundError(f"Best params file missing: {best_params_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"Trained model weights missing: {model_path}")
    if not model_input_npz_path.exists():
        raise FileNotFoundError(f"Training model input NPZ missing: {model_input_npz_path}")

    best_payload = json.loads(best_params_path.read_text(encoding="utf-8"))
    params = best_payload["best_params"]
    train_cfg = config.get("model_training", {})

    estimator = HybridLateFusionEstimator(
        window_size=window_size,
        company_vocab_size=company_vocab_size,
        price_feature_dim=price_feature_dim,
        sentiment_feature_dim=sentiment_feature_dim,
        price_hidden_dim=int(params["price_hidden_dim"]),
        sentiment_hidden_dim=int(params["sentiment_hidden_dim"]),
        price_num_layers=int(params.get("price_num_layers", 1)),
        sentiment_num_layers=int(params.get("sentiment_num_layers", 1)),
        lstm_dropout=float(params.get("lstm_dropout", 0.0)),
        sentiment_dropout=float(params.get("sentiment_dropout", 0.5)),
        company_emb_dim=int(params["company_emb_dim"]),
        ann_hidden_dim=int(params["ann_hidden_dim"]),
        dropout=float(params["dropout"]),
        learning_rate=float(params["learning_rate"]),
        optimizer_name=str(params.get("optimizer_name", "adam")),
        batch_size=int(params["batch_size"]),
        epochs=int(train_cfg.get("base_epochs", 8)),
        device=str(train_cfg.get("device", "auto")),
    )

    # Reuse exact estimator scaling logic using the training NPZ.
    train_payload = np.load(model_input_npz_path)
    train_X = np.asarray(train_payload["X"])
    train_price_seq, train_sent_seq, _ = estimator._decode_flat_features(train_X)
    estimator._fit_feature_scalers(train_price_seq, train_sent_seq)

    model = estimator._build_model()
    device = estimator._resolve_device()
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    estimator.model_ = model
    return estimator


def run_live_pipeline(
    config_path: str | Path = "src/config/pipeline_config.yaml",
    finnhub_api_key: str | None = None,
    tickers: list[str] | None = None,
) -> LiveRunArtifacts:
    config_path = Path(config_path)
    config = _load_config(config_path)
    logger = build_logger(config["paths"]["log_dir"])

    if not finnhub_api_key:
        raise ValueError("Finnhub API key is required. Provide finnhub_api_key when calling run_live_pipeline().")

    company_map_df = _load_company_map(config=config, logger=logger)
    available_tickers = set(company_map_df["ticker"].tolist())

    if tickers:
        requested_tickers = [t.upper().strip() for t in tickers]
        selected_tickers = [t for t in requested_tickers if t in available_tickers]
    else:
        selected_tickers = sorted(available_tickers)

    if not selected_tickers:
        raise ValueError("No valid tickers selected for live inference.")

    end_date = _today_utc_date() + timedelta(days=1)
    start_date = _today_utc_date() - timedelta(days=30)

    price_cfg = config["price_data"]
    window_size = int(config["alignment"]["window_size"])
    # align_modalities builds rows for idx in range(window_size, len(grp)),
    # so each ticker needs at least window_size + 1 trading rows.
    min_rows_for_alignment = window_size + 1

    prices_df = collect_price_data(
        tickers=selected_tickers,
        start_date=start_date.isoformat(),
        end_date=end_date.isoformat(),
        interval=str(price_cfg.get("interval", "1d")),
        logger=logger,
    )
    if prices_df.empty:
        raise RuntimeError("No live price data fetched from Yahoo Finance.")

    prices_df = (
        prices_df.sort_values(["Ticker", "Date"])
        .groupby("Ticker", as_index=False)
        .tail(min_rows_for_alignment)
        .reset_index(drop=True)
    )
    live_prices_path = PROJECT_ROOT / "data/processed/live_prices.csv"
    live_prices_path.parent.mkdir(parents=True, exist_ok=True)
    prices_df.to_csv(live_prices_path, index=False)

    news_parts: list[pd.DataFrame] = []
    for ticker, grp in prices_df.groupby("Ticker"):
        trading_dates = sorted(pd.to_datetime(grp["Date"], errors="coerce").dt.date.dropna().unique().tolist())
        if not trading_dates:
            continue
        from_date = trading_dates[0]
        to_date = trading_dates[-1]
        raw_news = _fetch_finnhub_company_news(
            ticker=str(ticker),
            from_date=from_date,
            to_date=to_date,
            api_key=finnhub_api_key,
            logger=logger,
        )
        news_parts.append(_news_to_pipeline_schema(str(ticker), raw_news))

    if news_parts:
        news_df = pd.concat(news_parts, ignore_index=True)
        news_df = news_df.drop_duplicates(subset=["ticker", "published_date", "news_text"]).reset_index(drop=True)
    else:
        news_df = pd.DataFrame(columns=["ticker", "published_date", "news_text"])

    _ = build_daily_news_price_alignment(prices_df=prices_df, news_df=news_df, logger=logger)

    sentiment_cfg = config.get("sentiment_model", {})
    sentiment_df = build_daily_sentiment(
        news_df=news_df,
        prices_df=prices_df,
        logger=logger,
        model_name=str(sentiment_cfg.get("model_name", "ProsusAI/finbert")),
        batch_size=int(sentiment_cfg.get("batch_size", 16)),
        max_length=int(sentiment_cfg.get("max_length", 256)),
        device_preference=str(sentiment_cfg.get("device", "auto")),
        top_k=int(sentiment_cfg.get("top_k", 5)),
        article_level_output_path=PROJECT_ROOT / "data/processed/live_sentiment_article_level.csv",
    )

    live_company_map_df = company_map_df[company_map_df["ticker"].isin(selected_tickers)].copy()
    live_company_map_df = live_company_map_df.sort_values("ticker").reset_index(drop=True)

    align_cfg = config["alignment"]
    aligned_df = align_modalities(
        prices_df=prices_df,
        sentiment_df=sentiment_df,
        company_map_df=live_company_map_df,
        window_size=int(align_cfg["window_size"]),
        target_column=str(align_cfg["target_column"]),
        sentiment_temporal_decay_lambda=float(align_cfg.get("sentiment_temporal_decay_lambda", 0.1)),
        sentiment_strength_threshold=float(align_cfg.get("sentiment_strength_threshold", 0.2)),
        sentiment_scale_factor=float(align_cfg.get("sentiment_scale_factor", 0.2)),
        logger=logger,
    )
    if aligned_df.empty:
        raise RuntimeError("Live alignment produced no rows; not enough price history for requested tickers.")

    latest_rows = (
        aligned_df.sort_values(["ticker", "target_date"]).groupby("ticker", as_index=False).tail(1).reset_index(drop=True)
    )

    model_input_cfg = config.get("model_input", {})
    arrays = build_flattened_hybrid_input(
        aligned_df=latest_rows,
        target_col=str(model_input_cfg.get("target_col", "target_log_return")),
        normalize_price_per_company=bool(model_input_cfg.get("normalize_price_per_company", True)),
    )

    estimator = _build_estimator_with_trained_weights(
        config=config,
        company_vocab_size=int(np.max(arrays.X[:, -1])) + 1,
        window_size=arrays.window_size,
        price_feature_dim=arrays.price_feature_dim,
        sentiment_feature_dim=arrays.sentiment_feature_dim,
    )
    pred = estimator.predict(arrays.X)

    predictions: dict[str, dict[str, float | int]] = {}
    for i, row in latest_rows.reset_index(drop=True).iterrows():
        ticker = str(row["ticker"])
        return_pred = float(pred[i, 0])
        volatility_pred = float(pred[i, 1])
        direction_prob = float(pred[i, 2])
        predictions[ticker] = {
            "return": return_pred,
            "direction": int(direction_prob >= 0.5),
            "volatility": volatility_pred,
        }

    live_predictions_path = PROJECT_ROOT / "data/processed/live_predictions.json"
    live_predictions_path.parent.mkdir(parents=True, exist_ok=True)
    live_predictions_path.write_text(json.dumps(predictions, indent=2), encoding="utf-8")

    return LiveRunArtifacts(
        live_prices_path=live_prices_path,
        live_predictions_path=live_predictions_path,
        predictions=predictions,
    )

