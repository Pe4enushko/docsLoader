from __future__ import annotations

import logging
from pathlib import Path
from threading import Lock

from app.config import get_settings


LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
_PIPELINE_HANDLER_LOCK = Lock()
_PIPELINE_HANDLER_REGISTRY: set[tuple[str, str]] = set()
_CONFIGURED = False


def configure_logging(level: int | str | None = None) -> None:
    """Configure root logging with terminal output and optional file output.

    The function is idempotent: repeated calls update root level but do not
    duplicate handlers.
    """
    global _CONFIGURED
    settings = get_settings()
    resolved_level = _resolve_level(level if level is not None else settings.log_level)

    root_logger = logging.getLogger()
    if not _CONFIGURED:
        formatter = logging.Formatter(LOG_FORMAT)
        root_logger.handlers.clear()
        root_logger.setLevel(resolved_level)

        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        root_logger.addHandler(stream_handler)

        if settings.log_to_files:
            log_dir = Path(settings.log_dir)
            log_dir.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(log_dir / settings.app_log_file, encoding="utf-8")
            file_handler.setFormatter(formatter)
            root_logger.addHandler(file_handler)

        _CONFIGURED = True
        return

    root_logger.setLevel(resolved_level)


def get_logger(name: str) -> logging.Logger:
    configure_logging()
    return logging.getLogger(name)


def get_pipeline_logger(name: str, file_name: str) -> logging.Logger:
    """Return logger that also writes to dedicated pipeline log file."""
    configure_logging()
    logger = logging.getLogger(name)
    settings = get_settings()

    if not settings.log_to_files:
        return logger

    log_dir = Path(settings.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = (log_dir / file_name).resolve()
    registry_key = (name, str(log_path))

    with _PIPELINE_HANDLER_LOCK:
        if registry_key in _PIPELINE_HANDLER_REGISTRY:
            return logger

        formatter = logging.Formatter(LOG_FORMAT)
        handler = logging.FileHandler(log_path, encoding="utf-8")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        _PIPELINE_HANDLER_REGISTRY.add(registry_key)

    return logger


def _resolve_level(level: int | str) -> int:
    if isinstance(level, int):
        return level
    return int(getattr(logging, str(level).upper(), logging.INFO))
