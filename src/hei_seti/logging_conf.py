"""Logging helpers providing structured JSON output."""
from __future__ import annotations

import json
import logging
import logging.config
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml


class JsonFormatter(logging.Formatter):
    """Minimal JSON formatter to avoid external dependencies."""

    def format(self, record: logging.LogRecord) -> str:  # noqa: D401 - inherited docstring
        payload: dict[str, Any] = {
            "timestamp": datetime.utcfromtimestamp(record.created).isoformat() + "Z",
            "level": record.levelname,
            "name": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        if record.__dict__.get("extra_data"):
            payload.update(record.__dict__["extra_data"])
        return json.dumps(payload, ensure_ascii=False)


def setup_logging(config_path: str | Path | None = None) -> None:
    """Configure logging from a YAML file or fallback to sane defaults."""

    if config_path is not None and Path(config_path).exists():
        with Path(config_path).open("r", encoding="utf-8") as stream:
            config = yaml.safe_load(stream)
            logging.config.dictConfig(config)
            return

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
