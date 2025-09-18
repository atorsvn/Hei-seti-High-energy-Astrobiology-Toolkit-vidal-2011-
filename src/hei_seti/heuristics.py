"""Heuristic mappings for Kardashev and Barrow metrics."""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .scales import BarrowLevel, KardashevRating

LOGGER = logging.getLogger(__name__)

ERG_CM2_S_TO_W_M2 = 1e-3  # Rough conversion factor for flux units
KPC_TO_METERS = 3.0856775814913673e19


@dataclass(slots=True)
class KBarrowCalculator:
    """Compute Kardashev and Barrow proxies from engineered features."""

    distance_col: str | None = None
    flux_unit: str = "erg cm-2 s-1"

    def estimate_power_watts(self, row: pd.Series) -> float:
        """Estimate power output using flux and optional distance."""

        flux = row.get("flux")
        if flux is None or math.isnan(float(flux)):
            return float("nan")

        flux_wm2 = float(flux)
        if self.flux_unit.lower().startswith("erg"):
            flux_wm2 *= ERG_CM2_S_TO_W_M2

        if self.distance_col and self.distance_col in row and not math.isnan(row[self.distance_col]):
            distance_m = float(row[self.distance_col]) * KPC_TO_METERS * 1e3  # kpc to m
            power = 4 * math.pi * (distance_m**2) * flux_wm2
        else:
            power = flux_wm2 * 1e20  # fallback scale factor when distance unknown
        return power

    def kardashev(self, row: pd.Series) -> float:
        power = self.estimate_power_watts(row)
        rating = KardashevRating(power).value()
        LOGGER.debug("kardashev", extra={"extra_data": {"power": power, "rating": rating}})
        return rating

    def barrow(self, row: pd.Series) -> int:
        mass = row.get("bh_mass")
        variability = row.get("var_ratio")
        hardness = row.get("hardness")

        level = BarrowLevel.BIII
        if pd.notna(mass) and float(mass) >= 10:
            level = BarrowLevel.BV
        if pd.notna(variability) and float(variability) > 100:
            level = BarrowLevel.BV
        if pd.notna(hardness) and float(hardness) > 5:
            level = BarrowLevel.BIV
        if pd.notna(mass) and float(mass) >= 20 and pd.notna(variability) and float(variability) > 200:
            level = BarrowLevel.BOMEGA

        LOGGER.debug(
            "barrow",
            extra={
                "extra_data": {
                    "mass": mass,
                    "variability": variability,
                    "hardness": hardness,
                    "level": int(level),
                }
            },
        )
        return int(level)

    def annotate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return dataframe with Kardashev and Barrow columns added."""

        result = df.copy()
        result["K"] = df.apply(self.kardashev, axis=1)
        result["B"] = df.apply(self.barrow, axis=1)
        LOGGER.info(
            "heuristics.annotate",
            extra={
                "extra_data": {
                    "rows": len(result),
                    "k_mean": float(np.nanmean(result["K"].to_numpy())),
                    "b_mean": float(np.nanmean(result["B"].to_numpy())),
                }
            },
        )
        return result
