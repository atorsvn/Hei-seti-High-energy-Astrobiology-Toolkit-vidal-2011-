import pandas as pd

from hei_seti.features import FeatureBuilder


def build_df():
    return pd.DataFrame(
        {
            "flux": [1.0, None],
            "fx": [None, 2.0],
            "flux_max": [4.0, 10.0],
            "flux_min": [1.0, 2.0],
            "hardness": [0.5, None],
            "hr1": [None, 1.2],
            "period": [10.0, None],
            "porb": [None, 5.0],
            "mbh": [7.0, None],
            "bhmass": [None, 12.0],
        }
    )


def test_feature_builder_selects_first_valid_values():
    df = build_df()
    builder = FeatureBuilder(
        flux_cols=["flux", "fx"],
        hardness_cols=["hardness", "hr1"],
        period_cols=["period", "porb"],
        bh_mass_cols=["mbh", "bhmass"],
    )
    features = builder.transform(df)
    assert features.loc[0, "flux"] == 1.0
    assert features.loc[1, "flux"] == 2.0
    assert features.loc[1, "hardness"] == 1.2
    assert features.loc[1, "period"] == 5.0
    assert features.loc[1, "bh_mass"] == 12.0
    assert features.loc[1, "var_ratio"] == 5.0
