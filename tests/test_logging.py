import json
import logging

from hei_seti import __version__
from hei_seti.logging_conf import JsonFormatter, setup_logging
from hei_seti.pipeline import Pipeline


def test_json_formatter_emits_expected_keys():
    formatter = JsonFormatter()
    record = logging.LogRecord("test", logging.INFO, __file__, 10, "hello", args=(), exc_info=None)
    payload = json.loads(formatter.format(record))
    assert payload["level"] == "INFO"
    assert payload["message"] == "hello"


def test_setup_logging_from_yaml(tmp_path):
    config_path = tmp_path / "logging.yaml"
    config_path.write_text(
        """
version: 1
formatters:
  json:
    (): hei_seti.logging_conf.JsonFormatter
handlers:
  console:
    class: logging.StreamHandler
    formatter: json
    level: INFO
    stream: ext://sys.stdout
root:
  level: INFO
  handlers: [console]
""",
        encoding="utf-8",
    )
    setup_logging(config_path)
    logger = logging.getLogger("hei_seti.tests")
    logger.info("configured")


def test_pipeline_from_yaml_uses_default_config():
    pipeline = Pipeline.from_yaml("configs/default.yaml")
    assert "fetch" in pipeline.config
    assert isinstance(__version__, str)


def test_setup_logging_basic_config():
    setup_logging(None)
    logger = logging.getLogger("hei_seti.sample")
    logger.info("message")
    assert logging.getLogger().handlers
