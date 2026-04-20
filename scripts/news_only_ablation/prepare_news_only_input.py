from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from news_only_ablation.build_model_input import (
    build_flattened_news_only_input_from_csv,
    save_news_only_input_npz,
)


def run(config_path: Path) -> None:
    with config_path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)

    aligned_csv = PROJECT_ROOT / config["paths"]["aligned_output"]
    out_npz = PROJECT_ROOT / config["paths"]["model_input_npz"]
    target_col = str(config.get("model_input", {}).get("target_col", "target_log_return"))

    arrays = build_flattened_news_only_input_from_csv(
        aligned_csv_path=aligned_csv,
        target_col=target_col,
    )
    save_news_only_input_npz(arrays, out_npz)

    print(f"Saved model input NPZ: {out_npz}")
    print(f"X shape: {arrays.X.shape}, y shape: {arrays.y.shape}")
    print(
        f"window_size={arrays.window_size}, "
        f"sentiment_feature_dim={arrays.sentiment_feature_dim}"
    )
    print(f"target_col={target_col}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare model-ready flattened inputs for news-only training")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("src/config/news_only_ablation_config.yaml"),
        help="Path to news-only ablation YAML config",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args.config)
