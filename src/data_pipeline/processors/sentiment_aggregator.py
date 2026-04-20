from __future__ import annotations

import logging
from bisect import bisect_right
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


LABEL_TO_COLUMN = {
    "positive": "sent_pos",
    "neutral": "sent_neu",
    "negative": "sent_neg",
}


def _select_torch_device(device_preference: str = "auto") -> torch.device:
    pref = str(device_preference).lower().strip()
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


def _assign_news_to_trading_buckets(
    news_df: pd.DataFrame,
    prices_df: pd.DataFrame,
    logger: logging.Logger,
) -> pd.DataFrame:
    if news_df.empty or prices_df.empty:
        return news_df

    news = news_df.copy()
    prices = prices_df.copy()

    news["ticker"] = news["ticker"].astype(str).str.upper().str.strip()
    news["published_date"] = pd.to_datetime(news["published_date"], errors="coerce").dt.date
    news = news.dropna(subset=["published_date"]).reset_index(drop=True)

    prices["Ticker"] = prices["Ticker"].astype(str).str.upper().str.strip()
    prices["Date"] = pd.to_datetime(prices["Date"], errors="coerce").dt.date
    prices = prices.dropna(subset=["Date"])

    trading_by_ticker: dict[str, list] = {
        str(ticker): sorted(grp["Date"].unique().tolist())
        for ticker, grp in prices.groupby("Ticker")
    }

    bucketed_rows: list[dict] = []
    dropped_rows = 0

    for _, row in news.iterrows():
        ticker = row["ticker"]
        published_date = row["published_date"]
        trading_dates = trading_by_ticker.get(ticker, [])

        if not trading_dates:
            dropped_rows += 1
            continue

        if published_date in trading_dates:
            bucket_date = published_date
        else:
            pos = bisect_right(trading_dates, published_date)
            if pos > 0:
                # Weekend/holiday news is attached to the most recent previous trading day.
                bucket_date = trading_dates[pos - 1]
            else:
                # If the news is earlier than the first trading day in range, attach to first day.
                bucket_date = trading_dates[0]

        bucketed_rows.append(
            {
                "ticker": ticker,
                "published_date": bucket_date,
                "news_text": row.get("news_text", ""),
            }
        )

    if dropped_rows:
        logger.warning("Dropped %s news rows due to missing trading calendar for ticker", dropped_rows)

    bucketed = pd.DataFrame(bucketed_rows)
    bucketed = bucketed[bucketed["news_text"].astype(str).str.strip().str.len() > 0]
    bucketed = bucketed.sort_values(["ticker", "published_date"]).reset_index(drop=True)
    return bucketed


def _run_finbert_inference(
    texts: list[str],
    model_name: str,
    batch_size: int,
    max_length: int,
    logger: logging.Logger,
    device_preference: str = "auto",
) -> list[tuple[float, float, float]]:
    if not texts:
        return []

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
    except Exception as exc:  # pragma: no cover - runtime environment dependent
        logger.exception("Failed to load FinBERT model '%s'", model_name)
        raise RuntimeError(
            "Unable to load FinBERT model. Ensure internet access or cached model files are available."
        ) from exc

    device = _select_torch_device(device_preference=device_preference)
    model = model.to(device)
    model.eval()

    id2label = {int(k): str(v).lower() for k, v in model.config.id2label.items()}
    ordered_columns: list[str] = []
    for idx in range(model.config.num_labels):
        label = id2label.get(idx, "")
        if label not in LABEL_TO_COLUMN:
            raise ValueError(f"Unexpected FinBERT label '{label}' in model {model_name}")
        ordered_columns.append(LABEL_TO_COLUMN[label])

    outputs: list[tuple[float, float, float]] = []
    logger.info("FinBERT inference using device=%s, batch_size=%s", device.type, batch_size)
    for start in range(0, len(texts), batch_size):
        batch_texts = texts[start : start + batch_size]
        tokens = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        tokens = {k: v.to(device) for k, v in tokens.items()}
        with torch.inference_mode():
            logits = model(**tokens).logits
            probs = torch.softmax(logits, dim=-1).cpu()

        for row in probs.tolist():
            values = dict(zip(ordered_columns, row))
            outputs.append(
                (
                    float(values["sent_pos"]),
                    float(values["sent_neu"]),
                    float(values["sent_neg"]),
                )
            )

    return outputs


def build_daily_sentiment(
    news_df: pd.DataFrame,
    prices_df: pd.DataFrame,
    logger: logging.Logger,
    model_name: str = "ProsusAI/finbert",
    batch_size: int = 16,
    max_length: int = 256,
    device_preference: str = "auto",
    top_k: int = 5,
    article_level_output_path: Path | str | None = None,
) -> pd.DataFrame:
    if news_df.empty:
        logger.warning("Input news dataframe is empty. Sentiment output will be empty.")
        return pd.DataFrame(
            columns=[
                "ticker",
                "published_date",
                "sent_pos",
                "sent_neu",
                "sent_neg",
                "sentiment_strength",
                "net_sentiment",
                "news_count",
            ]
        )

    bucketed_news_df = _assign_news_to_trading_buckets(news_df=news_df, prices_df=prices_df, logger=logger)

    if bucketed_news_df.empty:
        logger.warning("No bucketed news rows after trading-day assignment. Sentiment output will be empty.")
        return pd.DataFrame(
            columns=[
                "ticker",
                "published_date",
                "sent_pos",
                "sent_neu",
                "sent_neg",
                "sentiment_strength",
                "net_sentiment",
                "news_count",
            ]
        )

    logger.info(
        "Running FinBERT inference for %s news articles using model %s",
        len(bucketed_news_df),
        model_name,
    )
    article_probs = _run_finbert_inference(
        texts=bucketed_news_df["news_text"].astype(str).tolist(),
        model_name=model_name,
        batch_size=batch_size,
        max_length=max_length,
        logger=logger,
        device_preference=device_preference,
    )

    prob_df = pd.DataFrame(article_probs, columns=["sent_pos", "sent_neu", "sent_neg"])
    scored_df = pd.concat(
        [bucketed_news_df[["ticker", "published_date", "news_text"]].reset_index(drop=True), prob_df],
        axis=1,
    )

    # Article-level relevance score used for top-k filtering and weighted aggregation.
    scored_df["relevance_score"] = (scored_df["sent_pos"] - scored_df["sent_neg"]).abs()
    scored_df["article_idx"] = np.arange(len(scored_df), dtype=np.int64)

    top_k = max(int(top_k), 1)
    scored_df = scored_df.sort_values(
        ["ticker", "published_date", "relevance_score"],
        ascending=[True, True, False],
    )
    topk_df = scored_df.groupby(["ticker", "published_date"], as_index=False).head(top_k).copy()

    if article_level_output_path is not None:
        article_level_path = Path(article_level_output_path)
        article_level_path.parent.mkdir(parents=True, exist_ok=True)

        topk_idx = set(topk_df["article_idx"].astype(int).tolist())
        article_level_df = scored_df.copy()
        article_level_df["is_top_k"] = article_level_df["article_idx"].astype(int).isin(topk_idx).astype(int)
        article_level_df = article_level_df[
            [
                "ticker",
                "published_date",
                "news_text",
                "sent_pos",
                "sent_neu",
                "sent_neg",
                "relevance_score",
                "is_top_k",
            ]
        ]
        article_level_df.to_csv(article_level_path, index=False)
        logger.info("Saved per-article sentiment vectors to %s. Rows: %s", article_level_path, len(article_level_df))

    grouped = topk_df.groupby(["ticker", "published_date"], sort=False)
    aggregated = grouped.apply(
        lambda g: pd.Series(
            {
                "weight_sum": float(g["relevance_score"].sum()),
                "sent_pos_mean": float(g["sent_pos"].mean()),
                "sent_neu_mean": float(g["sent_neu"].mean()),
                "sent_neg_mean": float(g["sent_neg"].mean()),
                "sent_pos_weighted": float((g["relevance_score"] * g["sent_pos"]).sum()),
                "sent_neu_weighted": float((g["relevance_score"] * g["sent_neu"]).sum()),
                "sent_neg_weighted": float((g["relevance_score"] * g["sent_neg"]).sum()),
                "news_count": int(len(g)),
            }
        )
    ).reset_index()

    use_weighted = aggregated["weight_sum"] > 1e-12
    aggregated["sent_pos"] = np.where(
        use_weighted,
        aggregated["sent_pos_weighted"] / aggregated["weight_sum"],
        aggregated["sent_pos_mean"],
    )
    aggregated["sent_neu"] = np.where(
        use_weighted,
        aggregated["sent_neu_weighted"] / aggregated["weight_sum"],
        aggregated["sent_neu_mean"],
    )
    aggregated["sent_neg"] = np.where(
        use_weighted,
        aggregated["sent_neg_weighted"] / aggregated["weight_sum"],
        aggregated["sent_neg_mean"],
    )

    aggregated["sentiment_strength"] = (aggregated["sent_pos"] - aggregated["sent_neg"]).abs()
    aggregated["net_sentiment"] = aggregated["sent_pos"] - aggregated["sent_neg"]

    aggregated = aggregated[
        [
            "ticker",
            "published_date",
            "sent_pos",
            "sent_neu",
            "sent_neg",
            "sentiment_strength",
            "net_sentiment",
            "news_count",
        ]
    ]

    # Ensure a complete per-ticker trading-day table so no-news days are explicit zero-signal rows.
    trading_days = prices_df[["Ticker", "Date"]].copy()
    trading_days["ticker"] = trading_days["Ticker"].astype(str).str.upper().str.strip()
    trading_days["published_date"] = pd.to_datetime(trading_days["Date"], errors="coerce").dt.date
    trading_days = trading_days[["ticker", "published_date"]].dropna().drop_duplicates()

    aggregated = trading_days.merge(
        aggregated,
        how="left",
        on=["ticker", "published_date"],
    )

    fill_zero_cols = ["sent_pos", "sent_neu", "sent_neg", "sentiment_strength"]
    for col in fill_zero_cols:
        aggregated[col] = pd.to_numeric(aggregated[col], errors="coerce").fillna(0.0)

    aggregated["news_count"] = pd.to_numeric(aggregated["news_count"], errors="coerce").fillna(0).astype(int)

    aggregated = aggregated.sort_values(["ticker", "published_date"]).reset_index(drop=True)

    logger.info(
        "Built FinBERT daily sentiment table with top-k=%s weighted aggregation. Rows: %s",
        top_k,
        len(aggregated),
    )
    return aggregated
