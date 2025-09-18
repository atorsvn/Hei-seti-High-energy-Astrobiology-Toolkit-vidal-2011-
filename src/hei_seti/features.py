"""Feature engineering from heterogeneous catalog columns."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class FeatureBuilder:
    """Selects and standardises astrophysical features across catalogues."""

    flux_cols: Iterable[str]
    hardness_cols: Iterable[str]
    period_cols: Iterable[str]
    bh_mass_cols: Iterable[str]

    def _first_valid(self, row: pd.Series, candidates: Iterable[str]) -> float:
        for column in candidates:
            if column in row and pd.notna(row[column]):
                return float(row[column])
        return float("nan")

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create a clean feature matrix from a raw dataframe."""

        LOGGER.info("features.start", extra={"extra_data": {"rows": len(df)}})
        features = pd.DataFrame(index=df.index)
        features["flux"] = df.apply(lambda r: self._first_valid(r, self.flux_cols), axis=1)
        features["hardness"] = df.apply(lambda r: self._first_valid(r, self.hardness_cols), axis=1)
        features["period"] = df.apply(lambda r: self._first_valid(r, self.period_cols), axis=1)
        features["bh_mass"] = df.apply(lambda r: self._first_valid(r, self.bh_mass_cols), axis=1)

        if {"flux_max", "flux_min"}.issubset(df.columns):
            ratio = df["flux_max"] / df["flux_min"].replace({0: np.nan})
            features["var_ratio"] = ratio.replace([np.inf, -np.inf], np.nan)
        else:
            features["var_ratio"] = np.nan

        summary = {
            column: float(np.nanmean(features[column].to_numpy()))
            for column in features.columns
        }
        LOGGER.info("features.summary", extra={"extra_data": summary})
        return features
