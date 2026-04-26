from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, TimeSeriesSplit

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from news_only_ablation.sklearn_estimator import NewsOnlyLateFusionEstimator


def _extract_return_target(y: np.ndarray) -> np.ndarray:
    y_arr = np.asarray(y, dtype=np.float32)
    if y_arr.ndim == 1:
        return y_arr
    if y_arr.ndim == 2 and y_arr.shape[1] >= 1:
        return y_arr[:, 0]
    raise ValueError("Expected y with return in column 0.")


def run(config_path: Path) -> None:
    with config_path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)

    model_input_path = PROJECT_ROOT / config["paths"]["model_input_npz"]
    artifact_dir = PROJECT_ROOT / config["paths"]["trained_model_dir"]
    metrics_dir = PROJECT_ROOT / config["paths"].get("metrics_dir", "data/processed/news_only_ablation/metrics")
    metrics_dir.mkdir(parents=True, exist_ok=True)

    payload = np.load(model_input_path)
    X = payload["X"]
    y = payload["y"]
    window_size = int(payload["window_size"])
    sentiment_feature_dim = int(payload["sentiment_feature_dim"])
    company_vocab_size = int(np.max(X[:, -1])) + 1

    best = json.loads((artifact_dir / "news_only_late_fusion_best_params.json").read_text())
    params = best["best_params"]

    est = NewsOnlyLateFusionEstimator(
        window_size=window_size,
        company_vocab_size=company_vocab_size,
        sentiment_feature_dim=sentiment_feature_dim,
        sentiment_hidden_dim=int(params["sentiment_hidden_dim"]),
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
        splitter = TimeSeriesSplit(n_splits=cv_folds)
    else:
        splitter = KFold(n_splits=cv_folds, shuffle=False)

    y_return_all = _extract_return_target(y)

    return_r2_folds: list[float] = []
    return_rmse_folds: list[float] = []
    return_mae_folds: list[float] = []
    direction_acc_folds: list[float] = []
    confidence_mean_folds: list[float] = []
    confidence_std_folds: list[float] = []
    confidence_threshold = float(config.get("model_training", {}).get("confidence_threshold", 0.02))
    if confidence_threshold <= 0:
        raise ValueError("model_training.confidence_threshold must be > 0")

    for train_idx, test_idx in splitter.split(X):
        est.fit(X[train_idx], y[train_idx])
        pred_return = np.asarray(est.predict(X[test_idx]), dtype=np.float32).reshape(-1)

        true_return = y_return_all[test_idx]
        true_direction = (true_return > 0.0).astype(np.float32)
        pred_direction = (pred_return > 0.0).astype(np.float32)
        confidence = np.minimum(1.0, np.abs(pred_return) / confidence_threshold)

        return_r2_folds.append(float(r2_score(true_return, pred_return)))
        return_rmse_folds.append(float(np.sqrt(mean_squared_error(true_return, pred_return))))
        return_mae_folds.append(float(mean_absolute_error(true_return, pred_return)))
        direction_acc_folds.append(float(accuracy_score(true_direction, pred_direction)))
        confidence_mean_folds.append(float(np.mean(confidence)))
        confidence_std_folds.append(float(np.std(confidence)))

    result = {
        "return_r2_folds": return_r2_folds,
        "return_r2_mean": float(np.mean(return_r2_folds)),
        "return_r2_std": float(np.std(return_r2_folds)),
        "return_rmse_folds": return_rmse_folds,
        "return_rmse_mean": float(np.mean(return_rmse_folds)),
        "return_rmse_std": float(np.std(return_rmse_folds)),
        "return_mae_folds": return_mae_folds,
        "return_mae_mean": float(np.mean(return_mae_folds)),
        "return_mae_std": float(np.std(return_mae_folds)),
        "direction_accuracy_folds": direction_acc_folds,
        "direction_accuracy_mean": float(np.mean(direction_acc_folds)),
        "direction_accuracy_std": float(np.std(direction_acc_folds)),
        "confidence_threshold": confidence_threshold,
        "confidence_mean_folds": confidence_mean_folds,
        "confidence_mean": float(np.mean(confidence_mean_folds)),
        "confidence_std_folds": confidence_std_folds,
        "confidence_std": float(np.mean(confidence_std_folds)),
        "best_params": params,
        "cv_strategy": cv_strategy,
        "cv_folds": cv_folds,
    }

    out_path = metrics_dir / "news_only_r2_scores.json"
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print("News-only multitask metrics")
    print("return_r2_folds=" + ",".join(f"{s:.6f}" for s in return_r2_folds))
    print("return_r2_mean=" + str(float(np.mean(return_r2_folds))))
    print("return_rmse_folds=" + ",".join(f"{s:.6f}" for s in return_rmse_folds))
    print("return_rmse_mean=" + str(float(np.mean(return_rmse_folds))))
    print("direction_accuracy_folds=" + ",".join(f"{s:.6f}" for s in direction_acc_folds))
    print("direction_accuracy_mean=" + str(float(np.mean(direction_acc_folds))))
    print("confidence_mean_folds=" + ",".join(f"{s:.6f}" for s in confidence_mean_folds))
    print("confidence_mean=" + str(float(np.mean(confidence_mean_folds))))
    print(f"cv_strategy={cv_strategy}, cv_folds={cv_folds}")
    print(f"Saved metrics: {out_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate 3-fold metrics for news-only ablation model")
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
