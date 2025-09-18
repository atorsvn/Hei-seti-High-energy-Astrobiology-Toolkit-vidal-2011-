import math

import pandas as pd

from hei_seti.heuristics import KBarrowCalculator


def test_kardashev_uses_flux_and_distance():
    calc = KBarrowCalculator(distance_col="distance", flux_unit="erg cm-2 s-1")
    row = pd.Series({"flux": 1e-9, "distance": 5})
    power = calc.estimate_power_watts(row)
    assert power > 0
    rating = calc.kardashev(row)
    assert not math.isnan(rating)


def test_barrow_levels_increase_with_mass_and_variability():
    calc = KBarrowCalculator()
    low = calc.barrow(pd.Series({"bh_mass": 5, "var_ratio": 10, "hardness": 1}))
    high = calc.barrow(pd.Series({"bh_mass": 25, "var_ratio": 400, "hardness": 8}))
    assert high > low
