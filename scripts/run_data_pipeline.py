from __future__ import annotations

import argparse
import sys
from pathlib import Path
from datetime import timedelta

import pandas as pd
import yaml

# Allow importing from src/ when running as a script.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from data_pipeline.collectors.price_collector import collect_price_data
from data_pipeline.collectors.research_news_collector import (
    collect_news_from_research_csv,
    load_top_company_windows,
)
from data_pipeline.processors.alignment import (
    align_modalities,
    build_company_id_map,
    filter_aligned_by_company_windows,
)
from data_pipeline.processors.news_price_alignment import build_daily_news_price_alignment
from data_pipeline.processors.sentiment_aggregator import build_daily_sentiment
from data_pipeline.utils.logger import build_logger


def _print_stage_header(title: str) -> None:
    print("\n" + "=" * 88)
    print(title)
    print("=" * 88)


def _save_dataframe(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def _truncate_preview_text(df: pd.DataFrame, text_col: str, max_chars: int = 240) -> pd.DataFrame:
    """Return a copy with long text clipped for readable console previews."""
    preview = df.copy()
    if text_col in preview.columns:
        preview[text_col] = (
            preview[text_col]
            .astype(str)
            .str.replace(r"\s+", " ", regex=True)
            .str.strip()
            .str.slice(0, max_chars)
        )
    return preview


def _load_prepared_research_news(prepared_path: Path, logger) -> pd.DataFrame:
    if not prepared_path.exists():
        return pd.DataFrame(columns=["ticker", "published_date", "news_text"])

    df = pd.read_csv(prepared_path)
    required_cols = {"ticker", "published_date", "news_text"}
    if not required_cols.issubset(df.columns):
        logger.warning("Prepared research news file missing required columns: %s", prepared_path)
        return pd.DataFrame(columns=["ticker", "published_date", "news_text"])

    df["ticker"] = df["ticker"].astype(str).str.strip().str.upper()
    df["published_date"] = pd.to_datetime(df["published_date"], errors="coerce").dt.date
    df = df.dropna(subset=["published_date"]) 
    df = df.sort_values(["ticker", "published_date"]).reset_index(drop=True)
    logger.info("Loaded prepared research news CSV. Rows: %s", len(df))
    return df


def _collect_price_data_by_windows(
    company_windows_df: pd.DataFrame,
    interval: str,
    lookback_days: int,
    logger,
) -> pd.DataFrame:
    price_frames: list[pd.DataFrame] = []

    for _, row in company_windows_df.iterrows():
        ticker = row["ticker"]
        start_date = pd.to_datetime(row["start_date"]).date()
        end_date = pd.to_datetime(row["end_date"]).date()

        query_start = start_date - timedelta(days=lookback_days)
        query_end = end_date + timedelta(days=1)

        ticker_prices = collect_price_data(
            tickers=[ticker],
            start_date=query_start.isoformat(),
            end_date=query_end.isoformat(),
            interval=interval,
            logger=logger,
        )
        if ticker_prices.empty:
            continue

        price_frames.append(ticker_prices)

    if not price_frames:
        return pd.DataFrame(columns=["Date", "Ticker", "Open", "High", "Low", "Close", "Adj Close", "Volume"])

    prices_df = pd.concat(price_frames, ignore_index=True)
    prices_df = prices_df.sort_values(["Ticker", "Date"]).reset_index(drop=True)
    return prices_df


def _filter_prices_to_company_windows(
    prices_df: pd.DataFrame,
    company_windows_df: pd.DataFrame,
    logger,
) -> pd.DataFrame:
    if prices_df.empty or company_windows_df.empty:
        return prices_df

    windows = company_windows_df[["ticker", "start_date", "end_date"]].copy()
    windows["ticker"] = windows["ticker"].astype(str).str.upper().str.strip()
    windows["start_date"] = pd.to_datetime(windows["start_date"], errors="coerce").dt.date
    windows["end_date"] = pd.to_datetime(windows["end_date"], errors="coerce").dt.date

    prices = prices_df.copy()
    prices["Ticker"] = prices["Ticker"].astype(str).str.upper().str.strip()
    prices["Date"] = pd.to_datetime(prices["Date"], errors="coerce").dt.date

    merged = prices.merge(windows, how="left", left_on="Ticker", right_on="ticker")
    in_window = (merged["Date"] >= merged["start_date"]) & (merged["Date"] <= merged["end_date"])
    filtered = merged[in_window].drop(columns=["ticker", "start_date", "end_date"]).reset_index(drop=True)

    logger.info(
        "Applied strict company window filter on prices. Rows before: %s, after: %s",
        len(prices_df),
        len(filtered),
    )
    return filtered


def run_pipeline(config_path: Path, companies_filter: list[str] | None = None, device: str | None = None) -> None:
    with config_path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)

    logger = build_logger(config["paths"]["log_dir"])

    companies = config["companies"]
    
    # Override device if specified
    if device:
        config["sentiment_model"]["device"] = device
        logger.info(f"Overriding device to: {device}")
    price_cfg = config["price_data"]
    news_cfg = config["news_data"]
    sentiment_cfg = config.get("sentiment_model", {})
    align_cfg = config["alignment"]
    company_windows_df = pd.DataFrame()

    _print_stage_header("STAGE 0 - CONFIGURATION")
    company_windows_df = load_top_company_windows(news_cfg["top20_windows_csv"], logger=logger)
    companies = company_windows_df["ticker"].tolist()
    
    # Apply company filter if specified
    if companies_filter:
        companies_filter = [c.upper().strip() for c in companies_filter]
        company_windows_df = company_windows_df[company_windows_df["ticker"].isin(companies_filter)]
        companies = company_windows_df["ticker"].tolist()
        logger.info(f"Filtered to companies: {companies}")
        print(f"[FILTERED] Companies selected ({len(companies)}): {companies}")
    else:
        print(f"Companies selected from research windows ({len(companies)}): {companies}")
    
    if not company_windows_df.empty:
        print("Research window sample:")
        print(company_windows_df.head(10).to_string(index=False))

    print("News provider: research_csv")
    print(f"Price window size for alignment: {align_cfg['window_size']}")
    target_col = str(config.get("model_input", {}).get("target_col", "target_close"))
    if target_col in {"target_log_return", "target_return"}:
        print(f"Effective model sequence length (return-based): {int(align_cfg['window_size']) - 1}")
    else:
        print(f"Effective model sequence length (price-based): {int(align_cfg['window_size'])}")
    print(f"Price date range: {price_cfg['start_date']} -> {price_cfg['end_date']}")

    _print_stage_header("STAGE 1 - PRICE DATA COLLECTION")
    prices_df = _collect_price_data_by_windows(
        company_windows_df=company_windows_df,
        interval=price_cfg["interval"],
        lookback_days=int(price_cfg.get("lookback_buffer_days", 60)),
        logger=logger,
    )
    prices_df = _filter_prices_to_company_windows(
        prices_df=prices_df,
        company_windows_df=company_windows_df,
        logger=logger,
    )

    price_out = PROJECT_ROOT / config["paths"]["raw_prices"]
    _save_dataframe(prices_df, price_out)
    print(f"Saved price rows: {len(prices_df)} -> {price_out}")
    print("Price sample:")
    print(prices_df.head(5).to_string(index=False))
    print("Price rows per ticker:")
    if not prices_df.empty:
        print(prices_df.groupby("Ticker").size().to_string())

    _print_stage_header("STAGE 2 - NEWS DATA COLLECTION")
    prepared_news_path = PROJECT_ROOT / news_cfg.get("prepared_filtered_news_csv", "")
    refresh_prepared = bool(news_cfg.get("refresh_prepared_filtered_news", False))

    if prepared_news_path and prepared_news_path.exists() and not refresh_prepared:
        news_df = _load_prepared_research_news(prepared_news_path, logger=logger)
    else:
        news_df = collect_news_from_research_csv(
            real_companies_news_csv=news_cfg["real_companies_news_csv"],
            company_windows_df=company_windows_df,
            logger=logger,
            chunksize=int(news_cfg.get("chunksize", 200000)),
        )
        if prepared_news_path:
            _save_dataframe(news_df, prepared_news_path)
            logger.info("Saved prepared research news CSV to %s", prepared_news_path)

    news_out = PROJECT_ROOT / config["paths"]["raw_news"]
    _save_dataframe(news_df, news_out)
    print(f"Saved news rows: {len(news_df)} -> {news_out}")
    print("News sample:")
    sample_cols = ["ticker", "published_date", "news_text"]
    if not news_df.empty:
        news_preview = _truncate_preview_text(news_df[sample_cols].head(8), "news_text")
        print(news_preview.to_string(index=False))
        print("News rows per ticker:")
        print(news_df.groupby("ticker").size().to_string())

    _print_stage_header("STAGE 2B - DAILY NEWS + PRICE CONSOLIDATED ALIGNMENT")
    consolidated_df = build_daily_news_price_alignment(
        prices_df=prices_df,
        news_df=news_df,
        logger=logger,
    )
    consolidated_out = PROJECT_ROOT / config["paths"]["news_price_aligned_output"]
    _save_dataframe(consolidated_df, consolidated_out)
    print(f"Saved consolidated news+price rows: {len(consolidated_df)} -> {consolidated_out}")
    if not consolidated_df.empty:
        consolidated_preview = _truncate_preview_text(consolidated_df.head(8), "news")
        print(consolidated_preview.to_string(index=False))

    _print_stage_header("STAGE 3 - DAILY SENTIMENT AGGREGATION")
    sentiment_out = PROJECT_ROOT / config["paths"]["interim_sentiment"]
    reuse_sentiment = bool(sentiment_cfg.get("reuse_existing_if_present", False))

    if reuse_sentiment and sentiment_out.exists():
        sentiment_df = pd.read_csv(sentiment_out)
        sentiment_df["ticker"] = sentiment_df["ticker"].astype(str).str.upper().str.strip()
        sentiment_df["published_date"] = pd.to_datetime(sentiment_df["published_date"], errors="coerce").dt.date
        for col in ["sent_pos", "sent_neu", "sent_neg"]:
            sentiment_df[col] = pd.to_numeric(sentiment_df.get(col), errors="coerce").fillna(0.0)

        if "news_count" not in sentiment_df.columns:
            sentiment_df["news_count"] = 0
        sentiment_df["news_count"] = pd.to_numeric(sentiment_df["news_count"], errors="coerce").fillna(0).astype(int)

        if "has_news" not in sentiment_df.columns:
            sentiment_df["has_news"] = (sentiment_df["news_count"] > 0).astype(int)
        sentiment_df["has_news"] = pd.to_numeric(sentiment_df["has_news"], errors="coerce").fillna(0).astype(int)

        if "sentiment_strength" not in sentiment_df.columns:
            sentiment_df["sentiment_strength"] = (sentiment_df["sent_pos"] - sentiment_df["sent_neg"]).abs()
        sentiment_df["sentiment_strength"] = pd.to_numeric(
            sentiment_df["sentiment_strength"], errors="coerce"
        ).fillna(0.0)

        if "net_sentiment" not in sentiment_df.columns:
            sentiment_df["net_sentiment"] = sentiment_df["sent_pos"] - sentiment_df["sent_neg"]
        sentiment_df["net_sentiment"] = pd.to_numeric(
            sentiment_df["net_sentiment"], errors="coerce"
        ).fillna(0.0)

        sentiment_df = sentiment_df.dropna(subset=["published_date"])
        logger.info("Reused existing sentiment file from %s. Rows: %s", sentiment_out, len(sentiment_df))
    else:
        sentiment_df = build_daily_sentiment(
            news_df=news_df,
            prices_df=prices_df,
            logger=logger,
            model_name=str(sentiment_cfg.get("model_name", "ProsusAI/finbert")),
            batch_size=int(sentiment_cfg.get("batch_size", 16)),
            max_length=int(sentiment_cfg.get("max_length", 256)),
            device_preference=str(sentiment_cfg.get("device", "auto")),
            top_k=int(sentiment_cfg.get("top_k", 5)),
            article_level_output_path=PROJECT_ROOT / config["paths"].get(
                "interim_sentiment_article_level",
                "data/interim/sentiment/news_sentiment_article_level.csv",
            ),
        )
        _save_dataframe(sentiment_df, sentiment_out)

    print(f"Saved daily sentiment rows: {len(sentiment_df)} -> {sentiment_out}")
    print("Daily sentiment sample:")
    print(sentiment_df.head(10).to_string(index=False))

    _print_stage_header("STAGE 4 - COMPANY ID MAP")
    company_map_df = build_company_id_map(companies)
    company_map_out = PROJECT_ROOT / config["paths"]["company_map"]
    _save_dataframe(company_map_df, company_map_out)
    print(f"Saved company id map: {len(company_map_df)} -> {company_map_out}")
    print(company_map_df.to_string(index=False))

    _print_stage_header("STAGE 5 - MULTI-MODAL ALIGNMENT")
    aligned_df = align_modalities(
        prices_df=prices_df,
        sentiment_df=sentiment_df,
        company_map_df=company_map_df,
        window_size=align_cfg["window_size"],
        target_column=align_cfg["target_column"],
        sentiment_temporal_decay_lambda=float(align_cfg.get("sentiment_temporal_decay_lambda", 0.1)),
        sentiment_strength_threshold=float(align_cfg.get("sentiment_strength_threshold", 0.2)),
        sentiment_scale_factor=float(align_cfg.get("sentiment_scale_factor", 0.2)),
        logger=logger,
    )

    aligned_df = filter_aligned_by_company_windows(
        aligned_df=aligned_df,
        company_windows_df=company_windows_df,
        logger=logger,
    )

    aligned_out = PROJECT_ROOT / config["paths"]["aligned_output"]
    _save_dataframe(aligned_df, aligned_out)
    print(f"Saved aligned training rows: {len(aligned_df)} -> {aligned_out}")
    print("Aligned sample:")
    print(aligned_df.head(8).to_string(index=False))

    _print_stage_header("PIPELINE COMPLETE")
    print("Artifacts generated:")
    print(f"- Prices: {price_out}")
    print(f"- News: {news_out}")
    print(f"- Daily consolidated news+price: {consolidated_out}")
    print(f"- Sentiment daily: {sentiment_out}")
    print(f"- Company map: {company_map_out}")
    print(f"- Aligned dataset: {aligned_out}")
    print("Check logs/ for detailed execution logs.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run data collection + alignment pipeline")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("src/config/pipeline_config.yaml"),
        help="Path to the pipeline YAML config",
    )
    parser.add_argument(
        "--companies",
        type=str,
        nargs="+",
        default=None,
        help="Filter to specific companies (e.g., --companies AAPL MSFT). If not provided, uses all companies from config.",
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "gpu", "mps"],
        default=None,
        help="Compute device: cpu, gpu (CUDA), or mps (Apple Metal). If not specified, auto-selects best available (MPS > GPU > CPU).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    # Auto-detect best device if not specified
    device_to_use = args.device
    if not device_to_use:
        import torch
        if torch.backends.mps.is_available():
            device_to_use = "mps"
        elif torch.cuda.is_available():
            device_to_use = "gpu"
        else:
            device_to_use = "cpu"
        print(f"[AUTO-DETECT] Using device: {device_to_use}")
    
    run_pipeline(args.config, companies_filter=args.companies, device=device_to_use)
