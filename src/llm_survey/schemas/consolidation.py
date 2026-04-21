from __future__ import annotations

from enum import Enum
from typing import List

from pydantic import BaseModel, Field

from llm_survey.schemas.extraction import EvidenceStrength, RelationshipDirection, VariableType


class ConsensusStrength(str, Enum):
    STRONG = "strong"
    MODERATE = "moderate"
    WEAK = "weak"
    CONTESTED = "contested"
    NOVEL = "novel"


class ContradictionType(str, Enum):
    DIRECTION_CONFLICT = "direction_conflict"
    PRESENCE_CONFLICT = "presence_conflict"
    MODERATOR_MISSING = "moderator_missing"
    SUBGROUP_DIFFERENCE = "subgroup_difference"


class ResolutionStatus(str, Enum):
    RESOLVED = "resolved"
    PARTIALLY_RESOLVED = "partially_resolved"
    UNRESOLVED = "unresolved"


class PaperStance(str, Enum):
    SUPPORTS = "supports"
    CONTRADICTS = "contradicts"
    PARTIAL = "partial"


class MergedVariable(BaseModel):
    name: str
    aliases: List[str] = Field(default_factory=list)
    definition: str
    type: VariableType
    chunk_frequency: int = Field(ge=0, default=0)
    confidence: float = Field(ge=0.0, le=1.0)
    source_chunk_ids: List[str] = Field(default_factory=list)
    supporting_quotes: List[str] = Field(default_factory=list)
    evidence_strength: EvidenceStrength = Field(default=EvidenceStrength.DIRECT)


class ConsolidatedRelationship(BaseModel):
    from_variable: str
    to_variable: str
    direction: RelationshipDirection
    mechanism: str
    confidence: float = Field(ge=0.0, le=1.0)
    support_count: int = Field(ge=0, default=0)
    support_fraction: float = Field(ge=0.0, le=1.0, default=0.0)
    source_chunk_ids: List[str] = Field(default_factory=list)
    supporting_quotes: List[str] = Field(default_factory=list)
    contradicting_quotes: List[str] = Field(default_factory=list)
    evidence_strength: EvidenceStrength = Field(default=EvidenceStrength.DIRECT)


class ScoredHypothesis(BaseModel):
    id: str
    statement: str
    confidence: float = Field(ge=0.0, le=1.0)
    support_count: int = Field(ge=0, default=0)
    support_fraction: float = Field(ge=0.0, le=1.0, default=0.0)
    source_chunk_ids: List[str] = Field(default_factory=list)
    supporting_quotes: List[str] = Field(default_factory=list)
    contradicting_quotes: List[str] = Field(default_factory=list)
    linked_relationships: List[str] = Field(default_factory=list)
    from_variable: str = ""
    to_variable: str = ""
    direction: RelationshipDirection = Field(default=RelationshipDirection.UNCLEAR)
    evidence_strength: EvidenceStrength = Field(default=EvidenceStrength.DIRECT)
    consensus_strength: ConsensusStrength = Field(default=ConsensusStrength.WEAK)
    literature_support_score: float = Field(ge=0.0, le=1.0, default=0.0)
    novelty_flag: bool = False
    researcher_notes: str = ""


class ModeratorSpec(BaseModel):
    name: str
    target_relationship: str = ""
    rationale: str = ""
    source_chunk_ids: List[str] = Field(default_factory=list)
    supporting_quotes: List[str] = Field(default_factory=list)


class Contradiction(BaseModel):
    relationship: str
    conflict_type: ContradictionType
    version_a: str
    version_b: str
    resolution_attempted: str
    resolution_status: ResolutionStatus
    resolution_explanation: str
    requires_researcher_input: bool = False


class PaperReference(BaseModel):
    paper_id: str
    title: str
    authors: List[str] = Field(default_factory=list)
    year: int | None = None
    citation_count: int = 0
    relevant_excerpt: str = ""
    stance: PaperStance = Field(default=PaperStance.PARTIAL)


class HypothesisValidation(BaseModel):
    hypothesis_id: str
    hypothesis_statement: str
    supporting_papers: List[PaperReference] = Field(default_factory=list)
    contradicting_papers: List[PaperReference] = Field(default_factory=list)
    partial_papers: List[PaperReference] = Field(default_factory=list)
    literature_support_score: float = Field(ge=0.0, le=1.0, default=0.0)
    consensus_strength: ConsensusStrength = Field(default=ConsensusStrength.WEAK)
    novelty_flag: bool = False


class ConflictReport(BaseModel):
    contradictions: List[Contradiction] = Field(default_factory=list)
    resolved_count: int = 0
    partially_resolved_count: int = 0
    unresolved_count: int = 0


class LiteratureValidationReport(BaseModel):
    validations: List[HypothesisValidation] = Field(default_factory=list)
    strong_support_count: int = 0
    contested_count: int = 0
    novelty_count: int = 0


class ConsolidatedModel(BaseModel):
    variables: List[MergedVariable] = Field(default_factory=list)
    relationships: List[ConsolidatedRelationship] = Field(default_factory=list)
    hypotheses: List[ScoredHypothesis] = Field(default_factory=list)
    moderators: List[ModeratorSpec] = Field(default_factory=list)
    contradictions: List[Contradiction] = Field(default_factory=list)
    model_summary: str = ""
    research_questions: List[str] = Field(default_factory=list)
    overall_confidence: float = Field(ge=0.0, le=1.0, default=0.0)
