from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from hybrid_data_prep.build_model_input import convert_log_returns_to_prices
from news_only_ablation.sklearn_estimator import NewsOnlyLateFusionEstimator


def _require_multitask_targets(y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    y_arr = np.asarray(y, dtype=np.float32)
    if y_arr.ndim != 2 or y_arr.shape[1] < 3:
        raise ValueError("Expected y with shape (n_samples, 3+) and order [return, volatility, direction].")
    return y_arr[:, 0], y_arr[:, 1], y_arr[:, 2]


def _neg_return_rmse(estimator: NewsOnlyLateFusionEstimator, X: np.ndarray, y: np.ndarray) -> float:
    y = np.asarray(y)
    y_return = y[:, 0] if y.ndim == 2 else y
    pred = estimator.predict(X)
    pred_return = pred[:, 0] if pred.ndim == 2 else pred
    rmse = np.sqrt(mean_squared_error(y_return, pred_return))
    return -float(rmse)


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
    previous_close = payload["previous_close"]
    target_close = payload["target_close"]
    window_size = int(payload["window_size"])
    sentiment_feature_dim = int(payload["sentiment_feature_dim"])
    target_col = str(config.get("model_input", {}).get("target_col", "target_close"))

    company_vocab_size = int(np.max(X[:, -1])) + 1

    estimator = NewsOnlyLateFusionEstimator(
        window_size=window_size,
        company_vocab_size=company_vocab_size,
        sentiment_feature_dim=sentiment_feature_dim,
        epochs=int(train_cfg.get("base_epochs", 8)),
        device=str(train_cfg.get("device", "auto")),
        batch_size=64,
    )

    param_grid = train_cfg.get(
        "param_grid",
        {
            "sentiment_hidden_dim": [16, 32],
            "sentiment_num_layers": [1],
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

    scoring_cfg = str(train_cfg.get("scoring", "neg_root_mean_squared_error"))
    scoring = _neg_return_rmse if np.asarray(y).ndim == 2 else scoring_cfg

    grid = GridSearchCV(
        estimator=estimator,
        param_grid=param_grid,
        cv=cv,
        scoring=scoring,
        n_jobs=1,
        verbose=2,
        refit=True,
    )

    grid.fit(X, y)

    best_model_path = artifact_dir / "news_only_late_fusion_best.pt"
    best_params_path = artifact_dir / "news_only_late_fusion_best_params.json"

    best_estimator = grid.best_estimator_
    best_estimator.save_model(best_model_path)

    pred_target = best_estimator.predict(X)
    pred_return = pred_target[:, 0] if np.asarray(pred_target).ndim == 2 else pred_target
    pred_volatility = pred_target[:, 1]
    pred_direction_prob = pred_target[:, 2]

    if target_col == "target_log_return" or target_col == "target_return":
        pred_close = convert_log_returns_to_prices(pred_return, previous_close)
    elif target_col == "target_close":
        pred_close = pred_return
    else:
        raise ValueError(f"Unsupported model_input.target_col for close reconstruction: {target_col}")

    _, true_volatility, true_direction = _require_multitask_targets(y)

    direction_accuracy = float(accuracy_score(true_direction, (pred_direction_prob >= 0.5).astype(float)))
    volatility_rmse = float(np.sqrt(mean_squared_error(true_volatility, pred_volatility)))
    volatility_mae = float(mean_absolute_error(true_volatility, pred_volatility))

    close_rmse = float(np.sqrt(mean_squared_error(target_close, pred_close)))
    close_mae = float(mean_absolute_error(target_close, pred_close))

    with best_params_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "best_score": float(grid.best_score_),
                "best_params": grid.best_params_,
                "window_size": window_size,
                "sentiment_feature_dim": sentiment_feature_dim,
                "company_vocab_size": company_vocab_size,
                "target_col": target_col,
                "reconstructed_price_rmse": close_rmse,
                "reconstructed_price_mae": close_mae,
                "direction_accuracy_fit": direction_accuracy,
                "volatility_rmse_fit": volatility_rmse,
                "volatility_mae_fit": volatility_mae,
            },
            f,
            indent=2,
        )

    print(f"Best CV score (neg RMSE): {grid.best_score_}")
    print(f"Best params: {grid.best_params_}")
    print(f"Reconstructed close RMSE (fit data): {close_rmse}")
    print(f"Reconstructed close MAE (fit data): {close_mae}")
    print(f"Direction accuracy (fit data): {direction_accuracy}")
    print(f"Volatility RMSE (fit data): {volatility_rmse}")
    print(f"Volatility MAE (fit data): {volatility_mae}")
    print(f"Saved best model: {best_model_path}")
    print(f"Saved best params: {best_params_path}")
    print(f"CV strategy: {cv_strategy}, folds={cv_folds}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train news-only late-fusion model with GridSearchCV")
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
