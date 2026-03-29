from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from data_pipeline.collectors.research_news_collector import (
    collect_news_from_research_csv,
    load_top_company_windows,
)
from data_pipeline.utils.logger import build_logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare filtered research news CSV from top20 windows")
    parser.add_argument(
        "--top20-windows",
        type=Path,
        default=Path("data/reference/research/top20_companies_best_1yr_window.csv"),
        help="Path to top20 company windows CSV",
    )
    parser.add_argument(
        "--real-news",
        type=Path,
        default=Path("datasets/data/real_companies_news.csv"),
        help="Path to real_companies_news.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/raw/news/research_top20_filtered_news.csv"),
        help="Output path for filtered research news CSV",
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=200000,
        help="Chunk size for reading large source CSV",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logger = build_logger(str(PROJECT_ROOT / "logs"))

    windows_df = load_top_company_windows(str(PROJECT_ROOT / args.top20_windows), logger=logger)
    news_df = collect_news_from_research_csv(
        real_companies_news_csv=str(args.real_news),
        company_windows_df=windows_df,
        logger=logger,
        chunksize=args.chunksize,
    )

    out_path = PROJECT_ROOT / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    news_df.to_csv(out_path, index=False)

    print(f"WINDOW_ROWS {len(windows_df)}")
    print(f"FILTERED_NEWS_ROWS {len(news_df)}")
    print(f"OUTPUT {out_path}")

    if not news_df.empty:
        counts = news_df.groupby("ticker").size().sort_values(ascending=False)
        print("TICKER_COUNTS")
        print(counts.to_string())


if __name__ == "__main__":
    main()
