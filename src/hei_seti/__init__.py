"""High-Energy Astrobiology Toolkit."""

from importlib import metadata

from . import anomaly, data_sources, features, heuristics, pipeline, scales

__all__ = [
    "anomaly",
    "data_sources",
    "features",
    "heuristics",
    "pipeline",
    "scales",
    "__version__",
]


def __getattr__(name: str):
    if name == "__version__":
        try:
            return metadata.version("hei-seti")
        except metadata.PackageNotFoundError:  # pragma: no cover - during development
            return "0.0.0"
    raise AttributeError(name)
