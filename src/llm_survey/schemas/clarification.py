from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class AnswerSource(str, Enum):
    RESEARCHER = "researcher"
    LITERATURE = "literature"
    EITHER = "either"


class ClarificationPriority(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class ClarificationQuestion(BaseModel):
    question_id: str
    question_text: str
    target_gap: str
    priority: ClarificationPriority
    answer_source: AnswerSource
    context_for_researcher: str


class ClarificationAnswer(BaseModel):
    question_id: str
    answer_source: AnswerSource
    answer_text: str
    supporting_references: list[str] = Field(default_factory=list)


class ClarificationPlan(BaseModel):
    questions: list[ClarificationQuestion] = Field(default_factory=list)
    estimated_new_data_needed: bool
    can_proceed_with_literature: bool
    auto_answers: list[ClarificationAnswer] = Field(default_factory=list)
