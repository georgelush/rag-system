"""
rag/logging_config.py — Structured JSON logging for production.

Every log line is a single JSON object on stdout. Fields:
  ts        — ISO 8601 UTC timestamp
  level     — DEBUG / INFO / WARNING / ERROR / CRITICAL
  logger    — logger name  (e.g. "rag.query")
  msg       — log message
  ...       — any extra fields passed via extra={"key": value}

Usage:
    from rag.logging_config import setup_logging
    setup_logging(level="INFO")  # call once at startup

    import logging
    log = logging.getLogger(__name__)
    log.info("Query complete", extra={"latency_ms": 240, "tenant_id": "t1"})
"""

import json
import logging
import sys
from datetime import datetime, timezone
from typing import Any


class _JsonFormatter(logging.Formatter):
    _SKIP = frozenset({
        "msg", "args", "levelname", "levelno", "name",
        "pathname", "filename", "module", "exc_info", "exc_text",
        "stack_info", "lineno", "funcName", "created", "msecs",
        "relativeCreated", "thread", "threadName", "processName",
        "process", "message", "taskName",
    })

    def format(self, record: logging.LogRecord) -> str:
        record.message = record.getMessage()
        entry: dict[str, Any] = {
            "ts":     datetime.now(timezone.utc).isoformat(),
            "level":  record.levelname,
            "logger": record.name,
            "msg":    record.message,
        }
        if record.exc_info:
            entry["exc"] = self.formatException(record.exc_info)
        # Merge extra fields (skip internal LogRecord attrs)
        for key, val in record.__dict__.items():
            if key not in self._SKIP:
                entry[key] = val
        return json.dumps(entry, default=str)


def setup_logging(level: str = "INFO") -> None:
    """Configure root logger with JSON formatter. Call once at startup."""
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(_JsonFormatter())

    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    root.handlers.clear()
    root.addHandler(handler)

    # Suppress chatty third-party loggers
    for noisy in ("httpx", "httpcore", "openai", "qdrant_client", "uvicorn.access"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
