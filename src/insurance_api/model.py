"""Load the trained artifact and turn customers into predictions with explanations."""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb

from .features import clean, source_column


class Predictor:
    def __init__(self, path: Path):
        bundle = joblib.load(path)
        self.pre = bundle["preprocessor"]
        self.model = bundle["model"]
        self.threshold: float = bundle["threshold"]
        self.metadata: dict = bundle["metadata"]
        self.columns = list(self.pre.get_feature_names_out())

    def predict(self, rows: list[dict], top_k: int = 5) -> list[dict]:
        raw = pd.DataFrame(rows)
        Xt = self.pre.transform(clean(raw))
        proba = self.model.predict_proba(Xt)[:, 1]

        # Exact TreeSHAP values from XGBoost itself (log-odds scale); last column is the bias
        # No feature_names here: XGBoost rejects '<' (from "< 1 Year"); column order matches self.columns
        dmatrix = xgb.DMatrix(Xt)
        contribs = self.model.get_booster().predict(dmatrix, pred_contribs=True)[:, :-1]

        results = []
        for i, p in enumerate(proba):
            results.append(
                {
                    "probability": round(float(p), 4),
                    "interested": bool(p >= self.threshold),
                    "threshold": round(self.threshold, 4),
                    "top_reasons": self._top_reasons(contribs[i], raw.iloc[i], top_k),
                }
            )
        return results

    def _top_reasons(self, row_contribs: np.ndarray, raw_row: pd.Series, top_k: int) -> list[dict]:
        # One-hot columns of the same feature are summed, so "Vehicle_Damage" gets one number
        per_feature: dict[str, float] = {}
        for name, value in zip(self.columns, row_contribs, strict=False):
            key = source_column(name)
            per_feature[key] = per_feature.get(key, 0.0) + float(value)

        ranked = sorted(per_feature.items(), key=lambda kv: abs(kv[1]), reverse=True)[:top_k]
        return [
            {"feature": f, "value": str(raw_row[f]), "shap_value": round(v, 4)}
            for f, v in ranked
        ]
