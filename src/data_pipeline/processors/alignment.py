from __future__ import annotations

import logging
import math

import numpy as np
import pandas as pd


DEFAULT_SENTIMENT = {
    "sent_pos": 0.0,
    "sent_neu": 0.0,
    "sent_neg": 0.0,
    "sentiment_strength": 0.0,
    "net_sentiment": 0.0,
    "news_count": 0,
    "has_news": 0,
}


def build_company_id_map(tickers: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ticker": sorted(tickers),
            "company_id": list(range(len(tickers))),
        }
    )


def align_modalities(
    prices_df: pd.DataFrame,
    sentiment_df: pd.DataFrame,
    company_map_df: pd.DataFrame,
    window_size: int,
    target_column: str,
    logger: logging.Logger,
    sentiment_temporal_decay_lambda: float = 0.1,
    sentiment_strength_threshold: float = 0.25,
    sentiment_scale_factor: float = 0.05,
) -> pd.DataFrame:
    if prices_df.empty:
        logger.warning("Prices dataframe is empty. No aligned rows can be built.")
        return pd.DataFrame()

    price_data = prices_df[["Date", "Ticker", target_column]].copy()
    price_data = price_data.rename(columns={"Date": "date", "Ticker": "ticker", target_column: "target_value"})

    if sentiment_df.empty:
        sentiment_data = pd.DataFrame(
            columns=[
                "ticker",
                "published_date",
                "sent_pos",
                "sent_neu",
                "sent_neg",
                "sentiment_strength",
                "net_sentiment",
                "news_count",
                "has_news",
                "has_news",
            ]
        )
    else:
        sentiment_data = sentiment_df.copy()

    sentiment_data = sentiment_data.rename(columns={"published_date": "date"})

    if "has_news" not in sentiment_data.columns:
        if "news_count" in sentiment_data.columns:
            sentiment_data["has_news"] = (pd.to_numeric(sentiment_data["news_count"], errors="coerce").fillna(0) > 0).astype(int)
        else:
            sentiment_data["has_news"] = 1

    merged = price_data.merge(sentiment_data, on=["ticker", "date"], how="left")
    for col, val in DEFAULT_SENTIMENT.items():
        merged[col] = merged[col].fillna(val)

    id_map = dict(zip(company_map_df["ticker"], company_map_df["company_id"]))

    aligned_rows: list[dict] = []
    for ticker, grp in merged.groupby("ticker"):
        grp = grp.sort_values("date").reset_index(drop=True)
        company_id = id_map[ticker]

        for idx in range(window_size, len(grp)):
            window_slice = grp.iloc[idx - window_size : idx]
            target_row = grp.iloc[idx]
            close_window_np = window_slice["target_value"].to_numpy(dtype=float)
            safe_close_window_np = np.maximum(close_window_np, 1e-8)
            return_window = np.log(safe_close_window_np[1:] / safe_close_window_np[:-1]).round(6).tolist()

            # Build sentiment windows with 7 features: sent_pos, sent_neu, sent_neg, sentiment_strength, net_sentiment, news_count, has_news.
            sentiment_window_np = window_slice[
                ["sent_pos", "sent_neu", "sent_neg", "sentiment_strength", "net_sentiment", "news_count", "has_news"]
            ].to_numpy(dtype=float)
            # Align sentiment with returns: return_t corresponds to sentiment_t.
            sentiment_window_np = sentiment_window_np[1:]

            # Apply hard gating: mask out low-signal days where sentiment_strength < threshold.
            # sentiment_strength is at index 3.
            use_news = (sentiment_window_np[:, 3] >= sentiment_strength_threshold).astype(float).reshape(-1, 1)
            sentiment_window_np = sentiment_window_np * use_news

            # Apply temporal decay so older timesteps have lower influence.
            # The most recent timestep has days_ago=0 and decay=1.0.
            if sentiment_temporal_decay_lambda > 0.0:
                days_ago = np.arange(len(sentiment_window_np) - 1, -1, -1, dtype=np.float32)
                decay = np.exp(-float(sentiment_temporal_decay_lambda) * days_ago).reshape(-1, 1)
                sentiment_window_np = sentiment_window_np * decay

            # Scale down sentiment features to reduce noise.
            sentiment_window_np = sentiment_window_np * float(sentiment_scale_factor)

            sentiment_window = sentiment_window_np.round(6).tolist()

            previous_close = float(close_window_np[-1])
            target_close = float(target_row["target_value"])
            safe_prev = max(previous_close, 1e-8)
            safe_target = max(target_close, 1e-8)
            target_return = float(math.log(safe_target / safe_prev))
            direction = int(target_return > 0.0)
            volatility = float(abs(target_return))

            aligned_rows.append(
                {
                    "ticker": ticker,
                    "company_id": company_id,
                    "target_date": target_row["date"],
                    "window_start": window_slice["date"].iloc[0],
                    "window_end": window_slice["date"].iloc[-1],
                    "close_window": close_window_np.round(6).tolist(),
                    "return_window": return_window,
                    "sentiment_window": sentiment_window,
                    "news_count_window": window_slice["news_count"].astype(int).tolist(),
                    "target_sent_pos": float(target_row["sent_pos"]),
                    "target_sent_neu": float(target_row["sent_neu"]),
                    "target_sent_neg": float(target_row["sent_neg"]),
                    "target_sentiment_strength": float(target_row["sentiment_strength"]),
                    "target_news_count": int(target_row["news_count"]),
                    "previous_close": previous_close,
                    "target_return": target_return,
                    "target_log_return": target_return,
                    "direction": direction,
                    "volatility": volatility,
                    "target_close": target_close,
                }
            )

    aligned_df = pd.DataFrame(aligned_rows)
    logger.info("Alignment complete. Final training rows: %s", len(aligned_df))
    return aligned_df


def filter_aligned_by_company_windows(
    aligned_df: pd.DataFrame,
    company_windows_df: pd.DataFrame,
    logger: logging.Logger,
) -> pd.DataFrame:
    if aligned_df.empty or company_windows_df.empty:
        return aligned_df

    windows = company_windows_df.copy()
    windows = windows[["ticker", "start_date", "end_date"]].copy()
    windows["ticker"] = windows["ticker"].astype(str).str.upper()
    windows["start_date"] = pd.to_datetime(windows["start_date"], errors="coerce").dt.date
    windows["end_date"] = pd.to_datetime(windows["end_date"], errors="coerce").dt.date
    windows = windows.dropna(subset=["ticker", "start_date", "end_date"])

    merged = aligned_df.merge(windows, how="left", on="ticker")
    merged["target_date"] = pd.to_datetime(merged["target_date"], errors="coerce").dt.date

    in_window = (
        (merged["target_date"] >= merged["start_date"])
        & (merged["target_date"] <= merged["end_date"])
    )
    filtered = merged[in_window].drop(columns=["start_date", "end_date"]).reset_index(drop=True)

    logger.info(
        "Applied company window filter on aligned data. Rows before: %s, after: %s",
        len(aligned_df),
        len(filtered),
    )
    return filtered
