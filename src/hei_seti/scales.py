"""Utilities for Kardashev and Barrow scales."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import IntEnum
from typing import Iterable

import numpy as np

LOGGER = logging.getLogger(__name__)


class BarrowLevel(IntEnum):
    """Enumerated Barrow levels following Vidal (2011)."""

    BI = 1
    BII = 2
    BIII = 3
    BIV = 4
    BV = 5
    BOMEGA = 6


@dataclass(slots=True)
class KardashevRating:
    """Continuous Kardashev rating using Sagan interpolation."""

    power_watts: float | None

    def value(self) -> float:
        """Return `(log10(P) - 6)/10` when P is positive, else NaN."""

        if self.power_watts is None or self.power_watts <= 0:
            LOGGER.debug("Invalid power_watts=%s for Kardashev calculation", self.power_watts)
            return float("nan")
        value = (np.log10(self.power_watts) - 6.0) / 10.0
        LOGGER.debug("Computed Kardashev rating %.3f from power %.3e", value, self.power_watts)
        return value


def normalize_barrow_levels(levels: Iterable[int | float | BarrowLevel]) -> list[int]:
    """Normalize a collection of barrow levels to integers.

    Values outside the defined range are clipped to `[BI, BOMEGA]`.
    """

    normalized: list[int] = []
    for level in levels:
        try:
            numeric = int(level)
        except (TypeError, ValueError):  # pragma: no cover - defensive
            LOGGER.debug("Encountered invalid Barrow level %s", level)
            numeric = BarrowLevel.BI
        numeric = max(BarrowLevel.BI, min(BarrowLevel.BOMEGA, numeric))
        normalized.append(int(numeric))
    LOGGER.debug("Normalized Barrow levels: %s", normalized)
    return normalized
