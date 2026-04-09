from typing import Any, Dict

import instructor
import pytest

from llm_survey.rag_pipeline import RAGModelExtractor
from llm_survey.schemas.extraction import ChunkExtractionResult


class FakeCompletions:
    def create(self, **kwargs: Dict[str, Any]) -> ChunkExtractionResult:
        return ChunkExtractionResult(
            variables=[
                {
                    "name": "Workload",
                    "definition": "Task demand level.",
                    "type": "independent",
                    "example_quote": "I have too many deadlines.",
                },
                {
                    "name": "Stress",
                    "definition": "Experienced pressure.",
                    "type": "dependent",
                    "example_quote": "I feel overwhelmed.",
                },
            ],
            relationships=[
                {
                    "from_variable": "Workload",
                    "to_variable": "Stress",
                    "direction": "positive",
                    "mechanism": "Higher demand creates pressure.",
                    "supporting_quote": "Too many deadlines overwhelm me.",
                    "confidence": 0.91,
                }
            ],
            hypotheses=[
                {
                    "id": "H1",
                    "statement": "Workload positively affects stress.",
                    "supporting_quotes": ["I feel overwhelmed when deadlines pile up."],
                }
            ],
            moderators=[],
            gaps=[
                {
                    "description": "No coping strategy details",
                    "why_it_matters": "Limits intervention design",
                    "suggested_question": "What coping mechanisms are used?",
                }
            ],
            extraction_notes="Grounded in participant statements.",
        )


class FakeChat:
    def __init__(self) -> None:
        self.completions = FakeCompletions()


class FakeStructuredClient:
    def __init__(self) -> None:
        self.chat = FakeChat()


def test_extract_model_uses_typed_schema(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(instructor, "from_openai", lambda *args, **kwargs: FakeStructuredClient())

    extractor = RAGModelExtractor(
        openai_api_key="test-key",
        enable_literature_retrieval=False,
    )
    extractor.processed_chunks = [
        {
            "id": "c1",
            "text": "I feel overwhelmed when workload increases.",
            "metadata": {"speaker_id": "r1", "language": "en"},
            "original_index": 0,
        }
    ]

    results = extractor.extract_models_from_all_chunks(use_rag=False, save_results=False)

    assert len(results) == 1
    assert results[0]["success"] is True
    assert results[0]["model"]["relationships"][0]["direction"] == "positive"
    assert results[0]["model"]["gaps"][0]["suggested_question"]
