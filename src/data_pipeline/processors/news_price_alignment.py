from __future__ import annotations

import logging

import pandas as pd


def build_daily_news_price_alignment(
    prices_df: pd.DataFrame,
    news_df: pd.DataFrame,
    logger: logging.Logger,
) -> pd.DataFrame:
    if prices_df.empty or news_df.empty:
        logger.warning("Prices or news input is empty. Consolidated news-price alignment will be empty.")
        return pd.DataFrame(
            columns=[
                "ticker",
                "date",
                "Open",
                "High",
                "Low",
                "Close",
                "Adj Close",
                "Volume",
                "news_count",
                "news_text",
            ]
        )

    price_frame = prices_df.copy()
    price_frame = price_frame.rename(columns={"Ticker": "ticker", "Date": "date"})
    price_frame["ticker"] = price_frame["ticker"].astype(str).str.upper()
    price_frame["date"] = pd.to_datetime(price_frame["date"], errors="coerce").dt.date

    news_frame = news_df.copy()
    news_frame["ticker"] = news_frame["ticker"].astype(str).str.upper()
    news_frame["published_date"] = pd.to_datetime(news_frame["published_date"], errors="coerce").dt.date

    news_daily = (
        news_frame.groupby(["ticker", "published_date"], as_index=False)
        .agg(
            news_count=("news_text", "size"),
            news_text=("news_text", lambda s: " || ".join([x for x in s.astype(str) if x.strip()])),
        )
        .rename(columns={"published_date": "date"})
    )

    # Keep all price days and attach same-day news when available.
    merged = price_frame.merge(news_daily, how="left", on=["ticker", "date"])
    merged["news_count"] = merged["news_count"].fillna(0).astype(int)
    merged["news_text"] = merged["news_text"].fillna("")
    merged = merged.sort_values(["ticker", "date"]).reset_index(drop=True)

    logger.info("Daily news-price alignment complete. Rows: %s", len(merged))
    return merged
