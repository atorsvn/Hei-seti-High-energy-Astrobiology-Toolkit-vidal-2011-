"""End-to-end orchestration for the HEI-SETI workflow."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd
import yaml
from joblib import dump, load

from .anomaly import AnomalyModel
from .data_sources import HeasarcFetcher
from .features import FeatureBuilder
from .heuristics import KBarrowCalculator
from .logging_conf import setup_logging

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class Pipeline:
    """High-level pipeline comprised of ingestion, feature, and anomaly stages."""

    config: dict

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Pipeline":
        path = Path(path)
        with path.open("r", encoding="utf-8") as stream:
            config = yaml.safe_load(stream)
        logging_config = config.get("logging", {}).get("config")
        setup_logging(logging_config)
        LOGGER.info("pipeline.init", extra={"extra_data": {"config": str(path)}})
        return cls(config=config)

    def fetch(self, tables: Iterable[str] | None = None, output: str | Path = "data/raw.parquet") -> pd.DataFrame:
        cfg = self.config.get("fetch", {})
        tables = list(tables or cfg.get("heasarc_tables", []))
        fetcher = HeasarcFetcher(maxrec=cfg.get("maxrec", 20000))
        dataframe = fetcher.fetch_many(tables)
        fetcher.persist_dataframe(dataframe, output)
        return dataframe

    def featurize(
        self,
        dataframe: pd.DataFrame | None = None,
        input_path: str | Path = "data/raw.parquet",
        output: str | Path = "data/features.parquet",
    ) -> pd.DataFrame:
        if dataframe is None:
            dataframe = pd.read_parquet(input_path)
        cfg = self.config.get("features", {})
        builder = FeatureBuilder(
            flux_cols=cfg.get("flux_cols", []),
            hardness_cols=cfg.get("hardness_cols", []),
            period_cols=cfg.get("period_cols", []),
            bh_mass_cols=cfg.get("bh_mass_cols", []),
        )
        features = builder.transform(dataframe)
        heur_cfg = self.config.get("heuristics", {})
        kb = KBarrowCalculator(
            distance_col=heur_cfg.get("distance_col"),
            flux_unit=heur_cfg.get("flux_unit", "erg cm-2 s-1"),
        )
        features = kb.annotate(features)
        features["name"] = dataframe.get("name", dataframe.get("src_name", dataframe.index))
        features["_source_table"] = dataframe.get("_source_table", "unknown")
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        features.to_parquet(output)
        LOGGER.info(
            "pipeline.featurize",
            extra={"extra_data": {"rows": len(features), "output": str(output)}},
        )
        return features

    def train(
        self,
        features: pd.DataFrame | None = None,
        input_path: str | Path = "data/features.parquet",
        model_path: str | Path = "models/iforest.joblib",
    ) -> Path:
        if features is None:
            features = pd.read_parquet(input_path)
        cfg = self.config.get("anomaly", {})
        model = AnomalyModel(
            contamination=cfg.get("contamination", 0.05),
            random_state=cfg.get("random_state"),
        )
        model.fit(features)
        model_path = Path(model_path)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        dump(model, model_path)
        LOGGER.info(
            "pipeline.train",
            extra={"extra_data": {"rows": len(features), "model_path": str(model_path)}},
        )
        return model_path

    def score(
        self,
        model_path: str | Path,
        features: pd.DataFrame | None = None,
        input_path: str | Path = "data/features.parquet",
        top: int = 50,
        output: str | Path | None = "results/candidates.csv",
    ) -> pd.DataFrame:
        if features is None:
            features = pd.read_parquet(input_path)
        model: AnomalyModel = load(model_path)
        scores = model.rank(features, top=top)
        if output is not None:
            output_path = Path(output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            scores.to_csv(output_path, index=False)
            LOGGER.info(
                "pipeline.score",
                extra={
                    "extra_data": {
                        "rows": len(features),
                        "top": top,
                        "output": str(output_path),
                    }
                },
            )
        return scores
