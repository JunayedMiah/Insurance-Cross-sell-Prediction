"""Train the cross-sell model and save it as one artifact.

Usage:
    python -m insurance_api.train --data data/train.csv --out models/model.joblib
"""
import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, precision_recall_curve, roc_auc_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from .features import TARGET, build_preprocessor, clean

SEED = 42


def best_f1_threshold(y_true: np.ndarray, proba: np.ndarray) -> float:
    """Pick the probability cut-off that maximises F1 on validation data."""
    precision, recall, thresholds = precision_recall_curve(y_true, proba)
    f1 = 2 * precision * recall / np.clip(precision + recall, 1e-9, None)
    return float(thresholds[np.argmax(f1[:-1])])


def train(data_path: Path, out_path: Path, params: dict | None = None) -> dict:
    df = pd.read_csv(data_path)
    X, y = clean(df), df[TARGET].astype(int).to_numpy()

    # 70 / 15 / 15: train, validation (threshold), test (reported metrics)
    X_train, X_tmp, y_train, y_tmp = train_test_split(X, y, test_size=0.3, stratify=y, random_state=SEED)
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp, test_size=0.5, stratify=y_tmp, random_state=SEED
    )

    pre = build_preprocessor()
    Xt_train = pre.fit_transform(X_train)

    pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    defaults = {
        "n_estimators": 400,
        "learning_rate": 0.05,
        "max_depth": 5,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "scale_pos_weight": pos_weight,
        "eval_metric": "aucpr",
        "random_state": SEED,
        "n_jobs": -1,
    }
    model = XGBClassifier(**{**defaults, **(params or {})})
    model.fit(Xt_train, y_train)

    threshold = best_f1_threshold(y_val, model.predict_proba(pre.transform(X_val))[:, 1])

    test_proba = model.predict_proba(pre.transform(X_test))[:, 1]
    metrics = {
        "f1": round(f1_score(y_test, test_proba >= threshold), 4),
        "roc_auc": round(roc_auc_score(y_test, test_proba), 4),
        "pr_auc": round(average_precision_score(y_test, test_proba), 4),
        "positive_rate": round(float(y.mean()), 4),
    }

    bundle = {
        "preprocessor": pre,
        "model": model,
        "threshold": threshold,
        "metadata": {
            "model_version": datetime.now(UTC).strftime("%Y%m%d%H%M"),
            "trained_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "threshold": round(threshold, 4),
            "metrics": metrics,
            "n_train_rows": int(len(X_train)),
        },
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, out_path)
    _log_to_mlflow(model.get_params(), metrics, threshold)
    return bundle["metadata"]


def _log_to_mlflow(params: dict, metrics: dict, threshold: float) -> None:
    """Log the run if MLflow is installed; training works without it."""
    try:
        import mlflow
    except ImportError:
        return
    with mlflow.start_run(run_name="xgb-cross-sell"):
        mlflow.log_params({k: v for k, v in params.items() if v is not None})
        mlflow.log_metrics({**metrics, "threshold": threshold})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/train.csv"))
    parser.add_argument("--out", type=Path, default=Path("models/model.joblib"))
    args = parser.parse_args()
    print(json.dumps(train(args.data, args.out), indent=2))


if __name__ == "__main__":
    main()
