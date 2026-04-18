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


class EvidenceStrength(str, Enum):
    """How directly the chunk text supports the construct."""

    DIRECT = "direct"
    INFERRED = "inferred"
    WEAK = "weak"


class Variable(BaseModel):
    name: str = Field(description="Short, clear variable name.")
    definition: str = Field(description="Grounded definition from the text.")
    type: VariableType
    example_quote: str = Field(description="Direct supporting quote from the chunk.")
    source_chunk_ids: List[str] = Field(
        default_factory=list,
        description="Chunk id(s) this extraction row was grounded on.",
    )
    evidence_strength: EvidenceStrength = Field(
        default=EvidenceStrength.DIRECT,
        description="direct: explicit in text; inferred: not literally stated; weak: thin or single ambiguous line.",
    )


class Relationship(BaseModel):
    from_variable: str
    to_variable: str
    direction: RelationshipDirection
    mechanism: str = Field(description="How/why the relationship occurs.")
    supporting_quote: str
    confidence: float = Field(ge=0.0, le=1.0)
    source_chunk_ids: List[str] = Field(
        default_factory=list,
        description="Chunk id(s) supporting this relationship.",
    )
    evidence_strength: EvidenceStrength = Field(
        default=EvidenceStrength.DIRECT,
        description="direct / inferred / weak per Variable.evidence_strength semantics.",
    )


class Hypothesis(BaseModel):
    id: str = Field(description="Hypothesis id (e.g., H1).")
    statement: str = Field(description="Testable hypothesis statement.")
    supporting_quotes: List[str] = Field(default_factory=list)
    source_chunk_ids: List[str] = Field(
        default_factory=list,
        description="Chunk id(s) for cited quotes.",
    )
    evidence_strength: EvidenceStrength = Field(
        default=EvidenceStrength.DIRECT,
        description="direct / inferred / weak — use weak when only one thin quote supports.",
    )


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
