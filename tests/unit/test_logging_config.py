"""Unit tests for the logging-setup module (Step 10, Part A)."""

from __future__ import annotations

import logging
from collections.abc import Iterator

import pytest

from memories.logging_config import configure_logging


@pytest.fixture
def _restore_memories_logger() -> Iterator[None]:
    """Snapshot and restore the ``memories`` logger's level/handlers/propagate.

    The ``memories`` logger is a process-global singleton; without this fixture a
    test that configures it would leak the level, a stray stderr handler, and the
    propagate flag into every subsequent test in the suite.
    """
    logger = logging.getLogger("memories")
    saved = (logger.level, list(logger.handlers), logger.propagate)
    yield
    logger.setLevel(saved[0])
    logger.handlers = saved[1]
    logger.propagate = saved[2]


def test_configure_logging_defaults_to_info(
    _restore_memories_logger: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("LOG_LEVEL", raising=False)
    configure_logging()
    assert logging.getLogger("memories").level == logging.INFO


def test_configure_logging_reads_log_level_env(
    _restore_memories_logger: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")
    configure_logging()
    assert logging.getLogger("memories").level == logging.DEBUG


def test_configure_logging_explicit_arg_overrides_env(
    _restore_memories_logger: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("LOG_LEVEL", "ERROR")
    configure_logging("WARNING")
    assert logging.getLogger("memories").level == logging.WARNING


def test_configure_logging_invalid_level_falls_back_to_info(
    _restore_memories_logger: None,
) -> None:
    configure_logging("NOTALEVEL")
    assert logging.getLogger("memories").level == logging.INFO


def test_configure_logging_is_case_insensitive(_restore_memories_logger: None) -> None:
    configure_logging("debug")
    assert logging.getLogger("memories").level == logging.DEBUG


def test_configure_logging_attaches_single_handler(_restore_memories_logger: None) -> None:
    logging.getLogger("memories").handlers = []
    configure_logging()
    handlers = logging.getLogger("memories").handlers
    assert len(handlers) == 1
    assert isinstance(handlers[0], logging.StreamHandler)


def test_configure_logging_is_idempotent(_restore_memories_logger: None) -> None:
    logging.getLogger("memories").handlers = []
    configure_logging()
    configure_logging()
    assert len(logging.getLogger("memories").handlers) == 1


def test_configure_logging_emits_at_configured_level(
    _restore_memories_logger: None, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.INFO, logger="memories"):
        configure_logging("INFO")
        logging.getLogger("memories.test").info("hello")
    assert "hello" in caplog.text
