from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from hei_seti.data_sources import HeasarcFetcher


@dataclass
class DummyResult:
    table_name: str

    def to_table(self):
        class _Table:
            def __init__(self, table_name: str):
                self._table_name = table_name

            def to_pandas(self) -> pd.DataFrame:
                return pd.DataFrame({"name": ["src"], "flux": [1.0], "table": [self._table_name]})

        return _Table(self.table_name)


class DummyClient:
    def __init__(self):
        self.queries: list[str] = []

    def query_tap(self, query: str, maxrec: int):
        self.queries.append(query)
        table_name = query.split()[-1]
        return DummyResult(table_name)


def test_fetch_many_concatenates_tables(tmp_path):
    fetcher = HeasarcFetcher(maxrec=10, client=DummyClient())
    df = fetcher.fetch_many(["table1", "table2"])
    assert len(df) == 2
    assert set(df["_source_table"]) == {"table1", "table2"}
    out = tmp_path / "raw.parquet"
    path = fetcher.persist_dataframe(df, out)
    assert path.exists()
