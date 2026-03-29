from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from data_pipeline.collectors.research_news_collector import load_top_company_windows
from data_pipeline.processors.alignment import (
    align_modalities,
    build_company_id_map,
    filter_aligned_by_company_windows,
)
from data_pipeline.utils.logger import build_logger


def run(config_path: Path) -> None:
    config = yaml.safe_load(config_path.read_text())
    logger = build_logger(config["paths"]["log_dir"])

    prices = pd.read_csv(PROJECT_ROOT / config["paths"]["raw_prices"])
    sentiment = pd.read_csv(PROJECT_ROOT / config["paths"]["interim_sentiment"])

    windows = load_top_company_windows(config["news_data"]["top20_windows_csv"], logger=logger)
    company_map = build_company_id_map(windows["ticker"].tolist())

    aligned = align_modalities(
        prices_df=prices,
        sentiment_df=sentiment,
        company_map_df=company_map,
        window_size=int(config["alignment"]["window_size"]),
        target_column=str(config["alignment"]["target_column"]),
        logger=logger,
    )
    aligned = filter_aligned_by_company_windows(
        aligned_df=aligned,
        company_windows_df=windows,
        logger=logger,
    )

    out = PROJECT_ROOT / config["paths"]["aligned_output"]
    out.parent.mkdir(parents=True, exist_ok=True)
    aligned.to_csv(out, index=False)

    print(f"Saved aligned training rows: {len(aligned)} -> {out}")
    print("Columns:")
    print(list(aligned.columns))


if __name__ == "__main__":
    run(PROJECT_ROOT / "src/config/pipeline_config.yaml")
