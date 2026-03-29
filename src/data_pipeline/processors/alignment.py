from __future__ import annotations

import json
import logging
import math

import pandas as pd


DEFAULT_SENTIMENT = {"sent_pos": 0.2, "sent_neu": 0.6, "sent_neg": 0.2, "news_count": 0}


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
) -> pd.DataFrame:
    if prices_df.empty:
        logger.warning("Prices dataframe is empty. No aligned rows can be built.")
        return pd.DataFrame()

    price_data = prices_df[["Date", "Ticker", target_column]].copy()
    price_data = price_data.rename(columns={"Date": "date", "Ticker": "ticker", target_column: "target_value"})

    if sentiment_df.empty:
        sentiment_data = pd.DataFrame(columns=["ticker", "published_date", "sent_pos", "sent_neu", "sent_neg", "news_count"])
    else:
        sentiment_data = sentiment_df.copy()

    sentiment_data = sentiment_data.rename(columns={"published_date": "date"})

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
            previous_close = float(window_slice["target_value"].iloc[-1])
            target_close = float(target_row["target_value"])
            safe_prev = max(previous_close, 1e-8)
            safe_target = max(target_close, 1e-8)
            target_log_return = float(math.log(safe_target / safe_prev))

            aligned_rows.append(
                {
                    "ticker": ticker,
                    "company_id": company_id,
                    "target_date": target_row["date"],
                    "window_start": window_slice["date"].iloc[0],
                    "window_end": window_slice["date"].iloc[-1],
                    "close_window": json.dumps(window_slice["target_value"].round(6).tolist()),
                    "sentiment_window": json.dumps(
                        window_slice[["sent_pos", "sent_neu", "sent_neg"]].round(6).values.tolist()
                    ),
                    "news_count_window": json.dumps(window_slice["news_count"].astype(int).tolist()),
                    "target_sent_pos": float(target_row["sent_pos"]),
                    "target_sent_neu": float(target_row["sent_neu"]),
                    "target_sent_neg": float(target_row["sent_neg"]),
                    "target_news_count": int(target_row["news_count"]),
                    "previous_close": previous_close,
                    "target_log_return": target_log_return,
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
