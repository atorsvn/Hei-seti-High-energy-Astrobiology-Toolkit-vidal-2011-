import math

from hei_seti.scales import BarrowLevel, KardashevRating, normalize_barrow_levels


def test_kardashev_positive_power():
    rating = KardashevRating(1e16).value()
    assert math.isclose(rating, (math.log10(1e16) - 6) / 10)


def test_kardashev_non_positive_power_returns_nan():
    assert math.isnan(KardashevRating(-1).value())
    assert math.isnan(KardashevRating(None).value())


def test_normalize_barrow_levels_clips_values():
    values = normalize_barrow_levels([BarrowLevel.BII, 99, -1, "3"])
    assert values == [BarrowLevel.BII, BarrowLevel.BOMEGA, BarrowLevel.BI, BarrowLevel.BIII]
