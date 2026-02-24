from __future__ import annotations

import logging
from pathlib import Path


def setup_logging(log_file: str = "logs/pipeline.log", level: int = logging.INFO) -> None:
    root = logging.getLogger()
    if getattr(root, "_pipeline_logging_configured", False):
        return

    root.setLevel(level)
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    stream = logging.StreamHandler()
    stream.setLevel(level)
    stream.setFormatter(fmt)

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(level)
    file_handler.setFormatter(fmt)

    root.addHandler(stream)
    root.addHandler(file_handler)
    setattr(root, "_pipeline_logging_configured", True)
