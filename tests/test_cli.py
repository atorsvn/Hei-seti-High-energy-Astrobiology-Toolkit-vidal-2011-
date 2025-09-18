from __future__ import annotations

from pathlib import Path

import pandas as pd

from hei_seti import cli


class StubPipeline:
    def __init__(self):
        self.fetch_args = None
        self.featurize_args = None
        self.train_args = None
        self.score_args = None

    def fetch(self, tables=None, output=None):
        self.fetch_args = (tables, output)
        return pd.DataFrame({"value": [1, 2]})

    def featurize(self, input_path=None, output=None, dataframe=None):
        self.featurize_args = (input_path, output)
        return pd.DataFrame({"K": [0.1, 0.2], "B": [1, 2]})

    def train(self, input_path=None, model_path=None, features=None):
        self.train_args = (input_path, model_path)
        return Path(model_path)

    def score(self, model_path=None, input_path=None, top=50, output=None, features=None):
        self.score_args = (model_path, input_path, top, output)
        return pd.DataFrame({"K": [0.1], "B": [2], "anomaly": [0.5]})


def test_cli_fetch(monkeypatch, tmp_path, capsys):
    stub = StubPipeline()
    monkeypatch.setattr(cli, "_load_pipeline", lambda _: stub)
    output = tmp_path / "raw.parquet"
    exit_code = cli.main(["--config", "configs/default.yaml", "fetch", "--output", str(output), "--tables", "a", "b"])
    assert exit_code == 0
    captured = capsys.readouterr().out
    assert "Fetched" in captured
    assert stub.fetch_args == (["a", "b"], str(output))


def test_cli_featurize_and_train(monkeypatch, tmp_path, capsys):
    stub = StubPipeline()
    monkeypatch.setattr(cli, "_load_pipeline", lambda _: stub)
    features_path = tmp_path / "features.parquet"
    model_path = tmp_path / "model.joblib"
    exit_code = cli.main(["--config", "configs/default.yaml", "featurize", "--input", "data/raw.parquet", "--output", str(features_path)])
    assert exit_code == 0
    exit_code = cli.main(["--config", "configs/default.yaml", "train", "--input", str(features_path), "--model", str(model_path)])
    assert exit_code == 0
    captured = capsys.readouterr().out
    assert "Model saved" in captured
    assert stub.train_args == (str(features_path), str(model_path))


def test_cli_score_and_plot(monkeypatch, tmp_path, capsys):
    stub = StubPipeline()
    monkeypatch.setattr(cli, "_load_pipeline", lambda _: stub)

    features_path = tmp_path / "features.parquet"
    candidates_path = tmp_path / "candidates.csv"
    plot_path = tmp_path / "kb.png"

    pd.DataFrame({"K": [0.1, 0.2], "B": [1, 2], "anomaly": [0.3, 0.4]}).to_parquet(features_path)
    pd.DataFrame({"K": [0.2], "B": [2], "anomaly": [0.4]}).to_csv(candidates_path, index=False)

    exit_code = cli.main([
        "--config",
        "configs/default.yaml",
        "score",
        "--model",
        str(tmp_path / "model.joblib"),
        "--input",
        str(features_path),
        "--output",
        str(candidates_path),
        "--top",
        "1",
    ])
    assert exit_code == 0

    exit_code = cli.main([
        "--config",
        "configs/default.yaml",
        "plot",
        "--input",
        str(features_path),
        "--candidates",
        str(candidates_path),
        "--output",
        str(plot_path),
    ])
    assert exit_code == 0
    assert plot_path.exists()
    captured = capsys.readouterr().out
    assert "Plot saved" in captured
