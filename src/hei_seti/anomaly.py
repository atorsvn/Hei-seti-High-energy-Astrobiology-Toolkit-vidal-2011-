"""Anomaly detection for Kardashev–Barrow feature space."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

LOGGER = logging.getLogger(__name__)

FEATURE_COLUMNS = ["flux", "hardness", "period", "bh_mass", "var_ratio", "K", "B"]


@dataclass(slots=True)
class AnomalyModel:
    """Wrapper around scikit-learn IsolationForest with structured logging."""

    contamination: float = 0.05
    random_state: int | None = None
    _model: IsolationForest | None = field(default=None, init=False, repr=False)

    def _prepare(self, df: pd.DataFrame) -> np.ndarray:
        missing = [column for column in FEATURE_COLUMNS if column not in df]
        if missing:
            raise KeyError(f"Missing feature columns: {missing}")
        matrix = df[FEATURE_COLUMNS].astype(float).to_numpy()
        return np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)

    def fit(self, df: pd.DataFrame) -> IsolationForest:
        matrix = self._prepare(df)
        LOGGER.info(
            "anomaly.fit.start",
            extra={"extra_data": {"rows": len(df), "contamination": self.contamination}},
        )
        self._model = IsolationForest(
            contamination=self.contamination,
            random_state=self.random_state,
        )
        self._model.fit(matrix)
        LOGGER.info(
            "anomaly.fit.finish",
            extra={"extra_data": {"estimators": len(self._model.estimators_)}}
        )
        return self._model

    def score(self, df: pd.DataFrame) -> pd.Series:
        if self._model is None:
            raise RuntimeError("Model has not been fit")
        matrix = self._prepare(df)
        raw_scores = -self._model.score_samples(matrix)
        LOGGER.info(
            "anomaly.score",
            extra={"extra_data": {"rows": len(df), "score_mean": float(np.mean(raw_scores))}},
        )
        return pd.Series(raw_scores, index=df.index, name="anomaly")

    def rank(self, df: pd.DataFrame, top: int = 50) -> pd.DataFrame:
        scores = self.score(df)
        ranked = df.copy()
        ranked["anomaly"] = scores
        ranked["rank"] = ranked["anomaly"].rank(ascending=False, method="first")
        LOGGER.info(
            "anomaly.rank",
            extra={"extra_data": {"rows": len(df), "top": top}},
        )
        return ranked.sort_values("anomaly", ascending=False).head(top)
