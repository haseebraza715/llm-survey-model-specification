"""Mermaid export hardening: node ids and edge labels must stay mermaid-safe."""

from __future__ import annotations

from llm_survey.utils.export_reports import _mermaid_node_id, build_mermaid_diagram


def _model(relationships: list[dict]) -> dict:
    return {"relationships": relationships, "model_summary": "summary"}


def test_mermaid_node_id_sanitizes_special_characters() -> None:
    assert _mermaid_node_id('Workload ("demand")') == "Workload_demand"
    assert _mermaid_node_id("Stress!?") == "Stress"
    assert _mermaid_node_id("") == "unlabeled"
    assert _mermaid_node_id("   ") == "unlabeled"


def test_mermaid_diagram_escapes_quotes_and_pipes_in_labels() -> None:
    out = build_mermaid_diagram(
        _model(
            [
                {
                    "from_variable": 'Workload "heavy"',
                    "to_variable": "Stress",
                    "direction": "positive",
                    "confidence": 0.9,
                }
            ]
        )
    )
    assert 'Workload_heavy["Workload \'heavy\'"]' in out
    assert "positive, conf:0.90" in out
    assert out.count('"') % 2 == 0


def test_mermaid_diagram_replaces_pipe_in_label() -> None:
    out = build_mermaid_diagram(
        _model(
            [
                {
                    "from_variable": "A",
                    "to_variable": "B",
                    "direction": "conditional | weird",
                    "confidence": 0.5,
                }
            ]
        )
    )
    # The direction's pipe must not land inside the quoted label.
    assert "conditional / weird, conf:0.50" in out
    assert "| weird" not in out


def test_mermaid_diagram_skips_relationships_without_endpoints() -> None:
    out = build_mermaid_diagram(
        _model(
            [
                {"from_variable": "", "to_variable": "B", "direction": "positive", "confidence": 0.5},
                {"from_variable": "A", "to_variable": "", "direction": "positive", "confidence": 0.5},
            ]
        )
    )
    assert out == "graph LR"


def test_mermaid_diagram_is_deterministic() -> None:
    model = _model(
        [
            {"from_variable": "A", "to_variable": "B", "direction": "positive", "confidence": 0.9},
            {"from_variable": "B", "to_variable": "C", "direction": "negative", "confidence": 0.4},
        ]
    )
    assert build_mermaid_diagram(model) == build_mermaid_diagram(model)


def test_mermaid_diagram_reuses_same_node_id_for_duplicate_names() -> None:
    out = build_mermaid_diagram(
        _model(
            [
                {"from_variable": "A B", "to_variable": "C", "direction": "positive", "confidence": 0.9},
                {"from_variable": "A  B", "to_variable": "C", "direction": "negative", "confidence": 0.1},
            ]
        )
    )
    assert out.count('A_B["A B"]') == 1
    assert out.count("A_B -->") == 2
