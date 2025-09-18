from pathlib import Path

import pandas as pd

from hei_seti.pipeline import Pipeline


def sample_config(tmp_path: Path) -> dict:
    return {
        "features": {
            "flux_cols": ["flux", "fx"],
            "hardness_cols": ["hardness"],
            "period_cols": ["period"],
            "bh_mass_cols": ["bh_mass"],
        },
        "heuristics": {
            "distance_col": None,
            "flux_unit": "erg cm-2 s-1",
        },
        "anomaly": {
            "contamination": 0.2,
            "random_state": 0,
        },
    }


def raw_dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "flux": [1e-9, 2e-9, 3e-9, 4e-9],
            "hardness": [1.0, 2.0, 1.5, 3.5],
            "period": [10, 20, 30, 40],
            "bh_mass": [5, 8, 15, 30],
            "var_ratio": [5, 10, 200, 300],
            "name": ["A", "B", "C", "D"],
            "_source_table": ["t1"] * 4,
        }
    )


def test_pipeline_training_and_scoring(tmp_path):
    pipeline = Pipeline(config=sample_config(tmp_path))
    raw = raw_dataframe()
    features_path = tmp_path / "features.parquet"
    feats = pipeline.featurize(dataframe=raw, output=features_path)
    assert features_path.exists()
    model_path = tmp_path / "model.joblib"
    pipeline.train(features=feats, model_path=model_path)
    assert model_path.exists()
    output_path = tmp_path / "candidates.csv"
    scored = pipeline.score(model_path=model_path, features=feats, top=2, output=output_path)
    assert output_path.exists()
    assert len(scored) == 2
    assert {"anomaly", "rank"}.issubset(scored.columns)
