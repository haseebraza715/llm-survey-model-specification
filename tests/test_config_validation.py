"""Settings validation: bounds and malformed env values."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from llm_survey.config import Settings


def test_temperature_out_of_range_rejected(monkeypatch) -> None:
    monkeypatch.setenv("LLM_TEMPERATURE", "1.5")
    with pytest.raises(ValidationError):
        Settings(_env_file=None)  # type: ignore[call-arg]


def test_temperature_boundary_values_accepted(monkeypatch) -> None:
    monkeypatch.setenv("LLM_TEMPERATURE", "1.0")
    assert Settings(_env_file=None).llm_temperature == 1.0  # type: ignore[call-arg]
    monkeypatch.setenv("LLM_TEMPERATURE", "0.0")
    assert Settings(_env_file=None).llm_temperature == 0.0  # type: ignore[call-arg]


def test_malformed_seed_rejected(monkeypatch) -> None:
    monkeypatch.setenv("LLM_SEED", "abc")
    with pytest.raises(ValidationError):
        Settings(_env_file=None)  # type: ignore[call-arg]


def test_negative_seed_rejected(monkeypatch) -> None:
    monkeypatch.setenv("LLM_SEED", "-5")
    with pytest.raises(ValidationError):
        Settings(_env_file=None)  # type: ignore[call-arg]


def test_completeness_threshold_out_of_range_rejected(monkeypatch) -> None:
    monkeypatch.setenv("COMPLETENESS_THRESHOLD", "1.2")
    with pytest.raises(ValidationError):
        Settings(_env_file=None)  # type: ignore[call-arg]


def test_negative_refinement_iterations_rejected(monkeypatch) -> None:
    monkeypatch.setenv("MAX_REFINEMENT_ITERATIONS", "-1")
    with pytest.raises(ValidationError):
        Settings(_env_file=None)  # type: ignore[call-arg]


def test_valid_env_values_pass(monkeypatch) -> None:
    monkeypatch.setenv("LLM_TEMPERATURE", "0.7")
    monkeypatch.setenv("LLM_SEED", "99")
    monkeypatch.setenv("MAX_REFINEMENT_ITERATIONS", "4")
    monkeypatch.setenv("COMPLETENESS_THRESHOLD", "0.9")
    s = Settings(_env_file=None)  # type: ignore[call-arg]
    assert (s.llm_temperature, s.seed, s.max_refinement_iterations, s.completeness_threshold) == (
        0.7,
        99,
        4,
        0.9,
    )
