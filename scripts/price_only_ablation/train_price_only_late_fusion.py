from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from price_only_ablation.sklearn_estimator import PriceOnlyLateFusionEstimator


def run(config_path: Path) -> None:
    with config_path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)

    model_input_path = PROJECT_ROOT / config["paths"]["model_input_npz"]
    artifact_dir = PROJECT_ROOT / config["paths"]["trained_model_dir"]
    train_cfg = config.get("model_training", {})
    artifact_dir.mkdir(parents=True, exist_ok=True)

    payload = np.load(model_input_path)
    X = payload["X"]
    y = payload["y"]
    target_close = payload["target_close"]
    window_size = int(payload["window_size"])
    price_feature_dim = int(payload["price_feature_dim"])

    company_vocab_size = int(np.max(X[:, -1])) + 1

    estimator = PriceOnlyLateFusionEstimator(
        window_size=window_size,
        company_vocab_size=company_vocab_size,
        price_feature_dim=price_feature_dim,
        epochs=int(train_cfg.get("base_epochs", 8)),
        device=str(train_cfg.get("device", "auto")),
        batch_size=64,
    )

    param_grid = train_cfg.get(
        "param_grid",
        {
            "price_hidden_dim": [32, 64],
            "price_num_layers": [1, 2],
            "lstm_dropout": [0.0],
            "company_emb_dim": [8],
            "ann_hidden_dim": [64, 128],
            "dropout": [0.1, 0.2],
            "optimizer_name": ["adam", "adamw"],
            "learning_rate": [1e-3],
            "batch_size": [64],
        },
    )

    cv_folds = int(train_cfg.get("cv_folds", 3))
    cv_strategy = str(train_cfg.get("cv_strategy", "kfold")).strip().lower()
    if cv_strategy == "timeseries":
        cv = TimeSeriesSplit(n_splits=cv_folds)
    else:
        cv = cv_folds

    grid = GridSearchCV(
        estimator=estimator,
        param_grid=param_grid,
        cv=cv,
        scoring=str(train_cfg.get("scoring", "neg_root_mean_squared_error")),
        n_jobs=1,
        verbose=2,
        refit=True,
    )

    grid.fit(X, y)

    best_model_path = artifact_dir / "price_only_late_fusion_best.pt"
    best_params_path = artifact_dir / "price_only_late_fusion_best_params.json"

    best_estimator = grid.best_estimator_
    best_estimator.save_model(best_model_path)

    pred_close = best_estimator.predict(X)
    close_rmse = float(np.sqrt(mean_squared_error(target_close, pred_close)))
    close_mae = float(mean_absolute_error(target_close, pred_close))

    with best_params_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "best_score": float(grid.best_score_),
                "best_params": grid.best_params_,
                "window_size": window_size,
                "price_feature_dim": price_feature_dim,
                "company_vocab_size": company_vocab_size,
                "target_col": str(config.get("model_input", {}).get("target_col", "target_close")),
                "reconstructed_price_rmse": close_rmse,
                "reconstructed_price_mae": close_mae,
            },
            f,
            indent=2,
        )

    print(f"Best CV score (neg RMSE): {grid.best_score_}")
    print(f"Best params: {grid.best_params_}")
    print(f"Reconstructed close RMSE (fit data): {close_rmse}")
    print(f"Reconstructed close MAE (fit data): {close_mae}")
    print(f"Saved best model: {best_model_path}")
    print(f"Saved best params: {best_params_path}")
    print(f"CV strategy: {cv_strategy}, folds={cv_folds}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train price-only late-fusion model with GridSearchCV")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("src/config/price_only_ablation_config.yaml"),
        help="Path to price-only ablation YAML config",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args.config)
