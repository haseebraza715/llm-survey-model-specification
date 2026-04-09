from __future__ import annotations

from enum import Enum
from typing import List

from pydantic import BaseModel, Field


class GapType(str, Enum):
    MISSING_VARIABLE = "missing_variable"
    MISSING_MECHANISM = "missing_mechanism"
    AMBIGUOUS_DIRECTION = "ambiguous_direction"
    NO_MEASUREMENT = "no_measurement"


class GapPriority(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class CrossChunkGap(BaseModel):
    gap_type: GapType
    description: str
    affected_hypotheses: List[str] = Field(default_factory=list)
    frequency: int = Field(ge=1)
    priority: GapPriority
    suggested_follow_up: str


class CrossChunkGapReport(BaseModel):
    gaps: List[CrossChunkGap] = Field(default_factory=list)
    overall_model_completeness: float = Field(ge=0.0, le=1.0)
    model_testability_score: float = Field(ge=0.0, le=1.0)
    priority_gaps: List[str] = Field(default_factory=list)
