from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from live_pipeline import run_live_pipeline


LAST_RUN_PATH = PROJECT_ROOT / "last_run.txt"
LIVE_PRICES_PATH = PROJECT_ROOT / "data/processed/live_prices.csv"
LIVE_PREDICTIONS_PATH = PROJECT_ROOT / "data/processed/live_predictions.json"


def _read_last_run_timestamp() -> datetime | None:
    if not LAST_RUN_PATH.exists():
        return None
    raw = LAST_RUN_PATH.read_text(encoding="utf-8").strip()
    if not raw:
        return None
    try:
        return datetime.fromisoformat(raw)
    except ValueError:
        return None


def _write_last_run_timestamp(ts: datetime) -> None:
    LAST_RUN_PATH.write_text(ts.isoformat(), encoding="utf-8")


def _should_run_pipeline(now: datetime) -> bool:
    last_run = _read_last_run_timestamp()
    if last_run is None:
        return True
    return (now - last_run) > timedelta(hours=24)


def _run_pipeline_and_update_timestamp() -> str | None:
    api_key = os.getenv("FINNHUB_API_KEY", "").strip()
    if not api_key:
        return "Missing FINNHUB_API_KEY environment variable."
    run_live_pipeline(
        config_path=PROJECT_ROOT / "src/config/pipeline_config.yaml",
        finnhub_api_key=api_key,
    )
    _write_last_run_timestamp(datetime.now(timezone.utc))
    return None


def _load_prices() -> pd.DataFrame:
    if not LIVE_PRICES_PATH.exists():
        return pd.DataFrame()
    prices = pd.read_csv(LIVE_PRICES_PATH)
    prices["Ticker"] = prices["Ticker"].astype(str).str.upper().str.strip()
    prices["Date"] = pd.to_datetime(prices["Date"], errors="coerce")
    return prices.dropna(subset=["Date"])


def _load_predictions() -> dict[str, dict]:
    if not LIVE_PREDICTIONS_PATH.exists():
        return {}
    return json.loads(LIVE_PREDICTIONS_PATH.read_text(encoding="utf-8"))


def _render_price_page(prices: pd.DataFrame, selected_tickers: list[str]) -> None:
    st.subheader("Price Data")
    filtered = prices[prices["Ticker"].isin(selected_tickers)].copy()
    if filtered.empty:
        st.info("No live price data available for selected tickers.")
        return

    cols = st.columns(2)
    for idx, ticker in enumerate(selected_tickers):
        with cols[idx % 2]:
            ticker_df = filtered[filtered["Ticker"] == ticker].sort_values("Date")
            if ticker_df.empty:
                continue
            st.markdown(f"**{ticker}**")
            chart_df = ticker_df.set_index("Date")[["Close"]]
            st.line_chart(chart_df)


def _render_predictions_page(predictions: dict[str, dict], selected_tickers: list[str]) -> None:
    st.subheader("Predictions")
    filtered_items = [(t, predictions[t]) for t in selected_tickers if t in predictions]
    if not filtered_items:
        st.info("No live predictions available for selected tickers.")
        return

    cols_per_row = 3
    for i in range(0, len(filtered_items), cols_per_row):
        cols = st.columns(cols_per_row)
        for j, (ticker, pred) in enumerate(filtered_items[i : i + cols_per_row]):
            direction = int(pred.get("direction", 0))
            arrow = "↑" if direction == 1 else "↓"
            color = "green" if direction == 1 else "red"
            pred_return = float(pred.get("return", 0.0)) * 100.0
            pred_vol = float(pred.get("volatility", 0.0))
            with cols[j]:
                st.markdown(
                    f"""
                    <div style="border:1px solid #ddd;border-radius:10px;padding:14px;margin-bottom:12px;">
                      <h4 style="margin:0 0 8px 0;">{ticker}</h4>
                      <div style="font-size:22px;color:{color};margin-bottom:8px;">{arrow}</div>
                      <div><b>Predicted return:</b> {pred_return:.3f}%</div>
                      <div><b>Predicted volatility:</b> {pred_vol:.6f}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )


def main() -> None:
    st.set_page_config(page_title="Stock Insights Dashboard", layout="wide")
    now = datetime.now(timezone.utc)

    run_error: str | None = None
    if _should_run_pipeline(now):
        with st.spinner("Running live pipeline..."):
            run_error = _run_pipeline_and_update_timestamp()

    if run_error:
        st.error(run_error)

    prices = _load_prices()
    predictions = _load_predictions()
    all_tickers = sorted(set(prices["Ticker"].unique().tolist()) | set(predictions.keys()))

    left, right = st.columns([3, 2])
    with left:
        st.title("Stock Insights Dashboard")
    with right:
        default_tickers = all_tickers if all_tickers else []
        selected_tickers = st.multiselect(
            "Tickers",
            options=all_tickers,
            default=default_tickers,
        )

    if not all_tickers:
        st.warning("No live data yet. Set FINNHUB_API_KEY and refresh.")
        return

    if not selected_tickers:
        st.warning("Select at least one ticker.")
        return

    if st.button("Force Refresh"):
        with st.spinner("Force refreshing live pipeline..."):
            force_error = _run_pipeline_and_update_timestamp()
            if force_error:
                st.error(force_error)
            else:
                st.success("Live pipeline refreshed.")
                st.rerun()

    page = st.radio("Navigation", ["Price Data", "Predictions"], horizontal=True)
    if page == "Price Data":
        _render_price_page(prices, selected_tickers)
    else:
        _render_predictions_page(predictions, selected_tickers)


if __name__ == "__main__":
    main()

