from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml
from sklearn.model_selection import TimeSeriesSplit, cross_val_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from hybrid_model.sklearn_estimator import HybridLateFusionEstimator


def run(config_path: Path) -> None:
    with config_path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)

    model_input_path = PROJECT_ROOT / config["paths"]["model_input_npz"]
    artifact_dir = PROJECT_ROOT / config["paths"]["trained_model_dir"]

    payload = np.load(model_input_path)
    X = payload["X"]
    y = payload["y"]
    window_size = int(payload["window_size"])
    price_feature_dim = int(payload["price_feature_dim"])
    sentiment_feature_dim = int(payload["sentiment_feature_dim"])
    company_vocab_size = int(np.max(X[:, -1])) + 1

    best = json.loads((artifact_dir / "hybrid_late_fusion_best_params.json").read_text())
    params = best["best_params"]

    est = HybridLateFusionEstimator(
        window_size=window_size,
        company_vocab_size=company_vocab_size,
        price_feature_dim=price_feature_dim,
        sentiment_feature_dim=sentiment_feature_dim,
        price_hidden_dim=int(params["price_hidden_dim"]),
        sentiment_hidden_dim=int(params["sentiment_hidden_dim"]),
        price_num_layers=int(params.get("price_num_layers", 1)),
        sentiment_num_layers=int(params.get("sentiment_num_layers", 1)),
        lstm_dropout=float(params.get("lstm_dropout", 0.0)),
        company_emb_dim=int(params["company_emb_dim"]),
        ann_hidden_dim=int(params["ann_hidden_dim"]),
        dropout=float(params["dropout"]),
        learning_rate=float(params["learning_rate"]),
        optimizer_name=str(params.get("optimizer_name", "adam")),
        batch_size=int(params["batch_size"]),
        epochs=int(config.get("model_training", {}).get("base_epochs", 8)),
        device=str(config.get("model_training", {}).get("device", "auto")),
    )

    cv_folds = int(config.get("model_training", {}).get("cv_folds", 3))
    cv_strategy = str(config.get("model_training", {}).get("cv_strategy", "kfold")).strip().lower()
    if cv_strategy == "timeseries":
        cv = TimeSeriesSplit(n_splits=cv_folds)
    else:
        cv = cv_folds

    scores = cross_val_score(
        est,
        X,
        y,
        cv=cv,
        scoring="r2",
        n_jobs=1,
    )

    result = {
        "r2_folds": [float(s) for s in scores],
        "r2_mean": float(scores.mean()),
        "r2_std": float(scores.std()),
        "best_params": params,
        "cv_strategy": cv_strategy,
        "cv_folds": cv_folds,
    }

    out_path = artifact_dir / "hybrid_r2_scores.json"
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print("Hybrid model R2")
    print("r2_folds=" + ",".join(f"{s:.6f}" for s in scores))
    print("r2_mean=" + str(float(scores.mean())))
    print("r2_std=" + str(float(scores.std())))
    print(f"cv_strategy={cv_strategy}, cv_folds={cv_folds}")
    print(f"Saved metrics: {out_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate R2 for hybrid model")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("src/config/pipeline_config.yaml"),
        help="Path to pipeline YAML config",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args.config)
