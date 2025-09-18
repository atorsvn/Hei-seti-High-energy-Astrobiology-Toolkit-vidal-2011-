"""Data ingestion utilities for HEASARC catalogs."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import pandas as pd

try:  # pragma: no cover - import guard for optional dependency
    from astroquery.heasarc import Heasarc
except ImportError as exc:  # pragma: no cover - surfaces during optional installs
    Heasarc = None  # type: ignore
    IMPORT_ERROR = exc
else:  # pragma: no cover - this block not executed during unit tests
    IMPORT_ERROR = None

LOGGER = logging.getLogger(__name__)


class HeasarcUnavailableError(RuntimeError):
    """Raised when astroquery/Heasarc is not available."""


@dataclass(slots=True)
class HeasarcFetcher:
    """Fetch catalogues from HEASARC with provenance-aware logging."""

    maxrec: int = 20000
    client: Heasarc | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if self.client is None:
            if Heasarc is None:
                raise HeasarcUnavailableError(
                    "astroquery.heasarc.Heasarc is unavailable. Install astroquery to use"
                    " network fetching."
                ) from IMPORT_ERROR
            self.client = Heasarc()
        LOGGER.debug("Initialized HeasarcFetcher maxrec=%s", self.maxrec)

    def query_table(self, table: str) -> pd.DataFrame:
        """Query a HEASARC table using TAP and return a pandas DataFrame."""

        LOGGER.info("fetch.start", extra={"extra_data": {"table": table}})
        query = f"SELECT * FROM {table}"
        result = self.client.query_tap(query, maxrec=self.maxrec)
        df = result.to_table().to_pandas()
        df["_source_table"] = table
        LOGGER.info(
            "fetch.finish",
            extra={"extra_data": {"table": table, "rows": len(df)}},
        )
        return df

    def fetch_many(self, tables: Iterable[str]) -> pd.DataFrame:
        """Fetch multiple tables and concatenate them with provenance metadata."""

        frames = []
        for table in tables:
            try:
                frames.append(self.query_table(table))
            except Exception as error:  # pragma: no cover - network error path
                LOGGER.warning(
                    "fetch.error",
                    extra={"extra_data": {"table": table, "error": str(error)}},
                )
        if not frames:
            raise RuntimeError("No tables were successfully fetched")
        combined = pd.concat(frames, ignore_index=True)
        LOGGER.info(
            "fetch.concat",
            extra={"extra_data": {"rows": len(combined), "tables": list(tables)}},
        )
        return combined

    @staticmethod
    def persist_dataframe(df: pd.DataFrame, path: str | Path) -> Path:
        """Persist the dataframe to parquet with logging."""

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path)
        LOGGER.info(
            "persist.finish", extra={"extra_data": {"path": str(path), "rows": len(df)}}
        )
        return path
