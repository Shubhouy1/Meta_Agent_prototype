"""Structured logging with a build_id on every record.

Log lines are key=value pairs so they stay readable in a terminal and are
easy to parse later:

    2026-10-01 12:00:00 INFO metaagent.pipeline build_id=3f2a event=stage_started stage=planning
"""

import contextvars
import logging
from contextlib import contextmanager

_build_id: contextvars.ContextVar[str] = contextvars.ContextVar("build_id", default="-")
_configured = False


class _BuildIdFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.build_id = _build_id.get()
        return True


def configure_logging(level: int = logging.INFO) -> None:
    """Install one handler on the 'metaagent' logger. Safe to call repeatedly."""
    global _configured
    if _configured:
        return
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(
        "%(asctime)s %(levelname)s %(name)s build_id=%(build_id)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))
    handler.addFilter(_BuildIdFilter())
    root = logging.getLogger("metaagent")
    root.addHandler(handler)
    root.setLevel(level)
    root.propagate = False
    _configured = True


@contextmanager
def build_context(build_id: str):
    """Attach build_id to every log record emitted inside the block."""
    token = _build_id.set(build_id)
    try:
        yield
    finally:
        _build_id.reset(token)


def _format_value(value) -> str:
    text = str(value)
    if not text or any(c.isspace() for c in text) or "=" in text:
        return '"' + text.replace('"', "'") + '"'
    return text


def log_event(logger: logging.Logger, event: str, level: int = logging.INFO, **fields) -> None:
    parts = [f"event={event}"] + [f"{k}={_format_value(v)}" for k, v in fields.items()]
    logger.log(level, " ".join(parts))
