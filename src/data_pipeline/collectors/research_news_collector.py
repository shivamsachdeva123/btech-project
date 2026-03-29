from __future__ import annotations

import logging
import csv
import sys
from pathlib import Path

import pandas as pd


RESEARCH_OUTPUT_COLUMNS = ["ticker", "published_date", "news_text"]

# The research CSV has very large article text fields; raise parser limit.
csv.field_size_limit(sys.maxsize)


def load_top_company_windows(top20_windows_csv: str, logger: logging.Logger) -> pd.DataFrame:
    windows_path = Path(top20_windows_csv)
    if not windows_path.exists():
        logger.error("Top20 windows CSV not found: %s", windows_path)
        return pd.DataFrame(columns=["ticker", "start_date", "end_date", "news_count"])

    windows_df = pd.read_csv(windows_path)
    required_cols = {"Company", "Start_Date", "End_Date", "News_Count"}
    missing_cols = required_cols - set(windows_df.columns)
    if missing_cols:
        logger.error("Top20 windows CSV missing columns: %s", sorted(missing_cols))
        return pd.DataFrame(columns=["ticker", "start_date", "end_date", "news_count"])

    windows_df = windows_df.rename(
        columns={
            "Company": "ticker",
            "Start_Date": "start_date",
            "End_Date": "end_date",
            "News_Count": "news_count",
        }
    )

    windows_df["ticker"] = windows_df["ticker"].astype(str).str.strip().str.upper()
    windows_df["start_date"] = pd.to_datetime(windows_df["start_date"], errors="coerce").dt.date
    windows_df["end_date"] = pd.to_datetime(windows_df["end_date"], errors="coerce").dt.date
    windows_df = windows_df.dropna(subset=["ticker", "start_date", "end_date"]).reset_index(drop=True)

    logger.info("Loaded %s company windows from %s", len(windows_df), windows_path)
    return windows_df[["ticker", "start_date", "end_date", "news_count"]]


def collect_news_from_research_csv(
    real_companies_news_csv: str,
    company_windows_df: pd.DataFrame,
    logger: logging.Logger,
    chunksize: int = 200000,
) -> pd.DataFrame:
    news_path = Path(real_companies_news_csv)
    if not news_path.exists():
        logger.error("Research news CSV not found: %s", news_path)
        return pd.DataFrame(columns=RESEARCH_OUTPUT_COLUMNS)

    if company_windows_df.empty:
        logger.warning("Company windows are empty; no research news can be filtered.")
        return pd.DataFrame(columns=RESEARCH_OUTPUT_COLUMNS)

    windows_map = {
        row["ticker"]: (row["start_date"], row["end_date"]) for _, row in company_windows_df.iterrows()
    }
    selected_tickers = set(windows_map.keys())

    usecols = ["Date", "Article_title", "Stock_symbol", "Publisher", "Lsa_summary"]
    filtered_parts: list[pd.DataFrame] = []

    for chunk_idx, chunk in enumerate(
        pd.read_csv(
            news_path,
            usecols=usecols,
            chunksize=chunksize,
            on_bad_lines="skip",
            engine="python",
        )
    ):
        chunk["Stock_symbol"] = chunk["Stock_symbol"].astype(str).str.strip().str.upper()
        chunk = chunk[chunk["Stock_symbol"].isin(selected_tickers)].copy()
        if chunk.empty:
            continue

        chunk["Date"] = pd.to_datetime(chunk["Date"], errors="coerce", utc=True)
        chunk = chunk.dropna(subset=["Date"])
        if chunk.empty:
            continue

        chunk["published_date"] = chunk["Date"].dt.date

        keep_mask = pd.Series(False, index=chunk.index)
        for ticker, (start_date, end_date) in windows_map.items():
            ticker_mask = chunk["Stock_symbol"] == ticker
            date_mask = (chunk["published_date"] >= start_date) & (chunk["published_date"] <= end_date)
            keep_mask = keep_mask | (ticker_mask & date_mask)

        chunk = chunk[keep_mask].copy()
        if chunk.empty:
            continue

        normalized = pd.DataFrame(
            {
                "ticker": chunk["Stock_symbol"],
                "published_date": chunk["published_date"],
                "title": chunk["Article_title"].fillna("").astype(str).str.strip(),
                "summary": chunk["Lsa_summary"].fillna("").astype(str).str.strip(),
            }
        )

        normalized["news_text"] = (
            normalized["title"].fillna("").astype(str).str.strip()
            + " "
            + normalized["summary"].fillna("").astype(str).str.strip()
        ).str.replace(r"\s+", " ", regex=True).str.strip()

        normalized = normalized[normalized["news_text"].str.len() > 0]
        normalized = normalized[["ticker", "published_date", "news_text"]]
        if not normalized.empty:
            filtered_parts.append(normalized)

        if (chunk_idx + 1) % 10 == 0:
            logger.info("Processed %s chunks from research CSV", chunk_idx + 1)

    if not filtered_parts:
        logger.warning("No rows matched selected company windows from research CSV")
        return pd.DataFrame(columns=RESEARCH_OUTPUT_COLUMNS)

    news_df = pd.concat(filtered_parts, ignore_index=True)
    news_df = news_df.drop_duplicates(subset=["ticker", "published_date", "news_text"])
    news_df = news_df.sort_values(["ticker", "published_date"]).reset_index(drop=True)

    logger.info("Research CSV news filtering complete. Rows: %s", len(news_df))
    return news_df