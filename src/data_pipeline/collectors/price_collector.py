from __future__ import annotations

import logging
from typing import Iterable

import pandas as pd
import yfinance as yf


REQUIRED_COLUMNS = ["Date", "Ticker", "Open", "High", "Low", "Close", "Adj Close", "Volume"]


def collect_price_data(
    tickers: Iterable[str],
    start_date: str,
    end_date: str | None,
    interval: str,
    logger: logging.Logger,
) -> pd.DataFrame:
    all_frames: list[pd.DataFrame] = []

    for ticker in tickers:
        logger.info("Downloading price data for %s", ticker)
        raw_frame = yf.download(
            ticker,
            start=start_date,
            end=end_date,
            interval=interval,
            auto_adjust=False,
            progress=False,
        )

        if raw_frame is None:
            logger.warning("Unexpected price response type for %s", ticker)
            continue

        frame = pd.DataFrame(raw_frame)

        if frame.empty:
            logger.warning("No price data returned for %s", ticker)
            continue

        if isinstance(frame.columns, pd.MultiIndex):
            frame.columns = frame.columns.get_level_values(0)

        frame = frame.reset_index()
        frame["Ticker"] = ticker

        missing_columns = set(REQUIRED_COLUMNS) - set(frame.columns)
        if missing_columns:
            logger.warning("Ticker %s missing columns: %s", ticker, sorted(missing_columns))

        usable_columns = [col for col in REQUIRED_COLUMNS if col in frame.columns]
        frame = frame[usable_columns].copy()

        frame["Date"] = pd.to_datetime(frame["Date"]).dt.date
        all_frames.append(frame)
        logger.info("Collected %s price rows for %s", len(frame), ticker)

    if not all_frames:
        return pd.DataFrame(columns=REQUIRED_COLUMNS)

    result = pd.concat(all_frames, ignore_index=True)
    result = result.sort_values(["Ticker", "Date"]).reset_index(drop=True)
    logger.info("Price collection complete. Total rows: %s", len(result))
    return result
