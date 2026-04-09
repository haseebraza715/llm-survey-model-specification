from __future__ import annotations

from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field


class VariableType(str, Enum):
    INDEPENDENT = "independent"
    DEPENDENT = "dependent"
    MEDIATOR = "mediator"
    MODERATOR = "moderator"
    CONTEXTUAL = "contextual"


class RelationshipDirection(str, Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    UNCLEAR = "unclear"
    CONDITIONAL = "conditional"


class Variable(BaseModel):
    name: str = Field(description="Short, clear variable name.")
    definition: str = Field(description="Grounded definition from the text.")
    type: VariableType
    example_quote: str = Field(description="Direct supporting quote from the chunk.")


class Relationship(BaseModel):
    from_variable: str
    to_variable: str
    direction: RelationshipDirection
    mechanism: str = Field(description="How/why the relationship occurs.")
    supporting_quote: str
    confidence: float = Field(ge=0.0, le=1.0)


class Hypothesis(BaseModel):
    id: str = Field(description="Hypothesis id (e.g., H1).")
    statement: str = Field(description="Testable hypothesis statement.")
    supporting_quotes: List[str] = Field(default_factory=list)


class DetectedGap(BaseModel):
    description: str
    why_it_matters: str
    suggested_question: str


class ChunkExtractionResult(BaseModel):
    variables: List[Variable] = Field(default_factory=list)
    relationships: List[Relationship] = Field(default_factory=list)
    hypotheses: List[Hypothesis] = Field(default_factory=list)
    moderators: List[Variable] = Field(default_factory=list)
    gaps: List[DetectedGap] = Field(default_factory=list)
    extraction_notes: Optional[str] = None
