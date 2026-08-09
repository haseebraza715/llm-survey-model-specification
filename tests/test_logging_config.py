"""Logging configuration: idempotency, reset, and JSON/human modes."""

from __future__ import annotations

import json
import sys
from io import StringIO

from llm_survey import logging_config


def test_configure_logging_is_idempotent(monkeypatch) -> None:
    monkeypatch.setattr(logging_config, "_CONFIGURED", False)
    logging_config.configure_logging()
    first = logging_config._CONFIGURED
    logging_config.configure_logging(force_json=True)
    # Second call is a no-op unless reset=True.
    assert first and logging_config._CONFIGURED is first


def test_configure_logging_reset_allows_reconfiguration(monkeypatch) -> None:
    monkeypatch.setattr(logging_config, "_CONFIGURED", False)
    logging_config.configure_logging()
    logging_config.configure_logging(force_json=True, reset=True)
    assert logging_config._CONFIGURED


def test_get_logger_auto_configured(monkeypatch) -> None:
    monkeypatch.setattr(logging_config, "_CONFIGURED", False)
    log = logging_config.get_logger("test.logger")
    assert hasattr(log, "info")
    assert hasattr(log, "error")
    assert hasattr(log, "exception")


def test_get_logger_fallback_adapter_without_structlog(monkeypatch) -> None:
    monkeypatch.setattr(logging_config, "structlog", None)
    monkeypatch.setattr(logging_config, "_CONFIGURED", False)
    log = logging_config.get_logger("fallback")
    log.info("event", key="value")
    assert hasattr(log, "info")


def test_json_renderer_produces_valid_json(monkeypatch) -> None:
    monkeypatch.setattr(logging_config, "_CONFIGURED", False)
    stderr = StringIO()
    monkeypatch.setattr(sys, "stderr", stderr)
    logging_config.configure_logging(force_json=True)
    log = logging_config.get_logger("json.test")
    log.info("some_event", n=3)
    # Every line on stderr must be parseable JSON.
    for line in stderr.getvalue().strip().splitlines():
        if not line:
            continue
        payload = json.loads(line)
        assert payload["event"] == "some_event"
        assert payload["n"] == 3
