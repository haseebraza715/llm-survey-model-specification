from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any, Dict, Iterable, List, Mapping, Sequence

from llm_survey.schemas.consolidation import (
    ConflictReport,
    ConsolidatedModel,
    ConsolidatedRelationship,
    ConsensusStrength,
    Contradiction,
    ContradictionType,
    HypothesisValidation,
    LiteratureValidationReport,
    MergedVariable,
    ModeratorSpec,
    PaperReference,
    PaperStance,
    ResolutionStatus,
    ScoredHypothesis,
)
from llm_survey.schemas.extraction import EvidenceStrength, RelationshipDirection, VariableType


_STOP_TOKENS = {
    "and",
    "at",
    "by",
    "for",
    "from",
    "in",
    "of",
    "on",
    "or",
    "the",
    "to",
    "with",
}

_NEGATIVE_CUES = (
    "negative",
    "decrease",
    "decreases",
    "decreased",
    "reduce",
    "reduces",
    "reduced",
    "lower",
    "lowers",
    "lowered",
    "undermine",
    "undermines",
    "worsen",
    "worsens",
    "worsened",
    "harm",
    "harms",
)

_POSITIVE_CUES = (
    "positive",
    "increase",
    "increases",
    "increased",
    "higher",
    "improve",
    "improves",
    "improved",
    "boost",
    "boosts",
    "enhance",
    "enhances",
    "predicts",
    "associated with",
)


def _normalize_whitespace(text: str) -> str:
    return " ".join(str(text or "").split()).strip()


def _tokenize(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9]+", _normalize_whitespace(text).lower())
        if token not in _STOP_TOKENS
    }


def _variable_similarity(a: str, b: str) -> float:
    a_norm = _normalize_whitespace(a).lower()
    b_norm = _normalize_whitespace(b).lower()
    if not a_norm or not b_norm:
        return 0.0
    if a_norm == b_norm:
        return 1.0
    a_tokens = _tokenize(a_norm)
    b_tokens = _tokenize(b_norm)
    if not a_tokens or not b_tokens:
        return SequenceMatcher(None, a_norm, b_norm).ratio()
    jaccard = len(a_tokens & b_tokens) / len(a_tokens | b_tokens)
    seq = SequenceMatcher(None, a_norm, b_norm).ratio()
    if a_norm in b_norm or b_norm in a_norm:
        return max(jaccard, seq, 0.9)
    return max(jaccard, seq * 0.9)


def _dedupe_keep_order(values: Iterable[str]) -> List[str]:
    seen: set[str] = set()
    output: List[str] = []
    for value in values:
        cleaned = _normalize_whitespace(value)
        key = cleaned.lower()
        if not cleaned or key in seen:
            continue
        seen.add(key)
        output.append(cleaned)
    return output


def _safe_direction(value: str) -> RelationshipDirection:
    try:
        return RelationshipDirection(str(value).lower())
    except ValueError:
        return RelationshipDirection.UNCLEAR


def _safe_variable_type(value: str) -> VariableType:
    try:
        return VariableType(str(value).lower())
    except ValueError:
        return VariableType.CONTEXTUAL


def _safe_evidence_strength(value: str) -> EvidenceStrength:
    try:
        return EvidenceStrength(str(value).lower())
    except ValueError:
        return EvidenceStrength.WEAK


def _pick_majority_counter(values: Sequence[str], default: str) -> str:
    filtered = [_normalize_whitespace(v) for v in values if _normalize_whitespace(v)]
    if not filtered:
        return default
    counts = Counter(filtered)
    return max(counts, key=lambda item: (counts[item], len(item)))


@dataclass
class _VariableMention:
    name: str
    definition: str
    var_type: VariableType
    quote: str
    chunk_id: str
    evidence_strength: EvidenceStrength


@dataclass
class _RelationshipMention:
    from_variable: str
    to_variable: str
    direction: RelationshipDirection
    mechanism: str
    quote: str
    confidence: float
    chunk_id: str
    evidence_strength: EvidenceStrength


@dataclass
class _HypothesisMention:
    statement: str
    hypothesis_id: str
    quote: str
    chunk_id: str
    evidence_strength: EvidenceStrength
    from_variable: str = ""
    to_variable: str = ""
    direction: RelationshipDirection = RelationshipDirection.UNCLEAR


@dataclass
class _ModeratorMention:
    name: str
    quote: str
    chunk_id: str
    target_relationship: str = ""
    rationale: str = ""


class ModelConsolidator:
    """Merge chunk-level extractions into a coherent consolidated model."""

    def consolidate(
        self,
        extraction_results: List[Dict[str, Any]],
        gap_report: Mapping[str, Any] | None = None,
        clarification_plan: Mapping[str, Any] | None = None,
    ) -> ConsolidatedModel:
        successful = [r for r in extraction_results if r.get("success") and isinstance(r.get("model"), dict)]
        total_chunks = max(1, len(successful))

        variable_mentions = self._collect_variable_mentions(successful)
        grouped_variables = self._cluster_variables(variable_mentions)
        merged_variables, alias_to_canonical = self._merge_variables(grouped_variables, total_chunks=total_chunks)

        relationship_mentions = self._collect_relationship_mentions(successful, alias_to_canonical)
        consolidated_relationships = self._merge_relationships(relationship_mentions, total_chunks=total_chunks)
        self._attach_contradicting_quotes(consolidated_relationships)

        hypothesis_mentions = self._collect_hypothesis_mentions(successful, alias_to_canonical, consolidated_relationships)
        scored_hypotheses = self._merge_hypotheses(
            hypothesis_mentions,
            consolidated_relationships,
            total_chunks=total_chunks,
        )

        moderators = self._merge_moderators(successful, alias_to_canonical)
        model_summary = self._build_model_summary(merged_variables, consolidated_relationships, scored_hypotheses)
        research_questions = self._build_research_questions(gap_report, clarification_plan, consolidated_relationships)
        overall_confidence = self._overall_confidence(consolidated_relationships, scored_hypotheses)

        return ConsolidatedModel(
            variables=merged_variables,
            relationships=consolidated_relationships,
            hypotheses=scored_hypotheses,
            moderators=moderators,
            contradictions=[],
            model_summary=model_summary,
            research_questions=research_questions,
            overall_confidence=overall_confidence,
        )

    def _collect_variable_mentions(self, successful: Sequence[Dict[str, Any]]) -> List[_VariableMention]:
        mentions: List[_VariableMention] = []
        for row in successful:
            chunk_id = str(row.get("chunk_id", ""))
            model = row.get("model") or {}
            for key in ("variables", "moderators"):
                for variable in model.get(key) or []:
                    name = _normalize_whitespace(variable.get("name", ""))
                    if not name:
                        continue
                    mentions.append(
                        _VariableMention(
                            name=name,
                            definition=_normalize_whitespace(variable.get("definition", "")) or f"{name} as described in participant text.",
                            var_type=_safe_variable_type(variable.get("type", VariableType.CONTEXTUAL.value)),
                            quote=_normalize_whitespace(variable.get("example_quote", "")),
                            chunk_id=chunk_id,
                            evidence_strength=_safe_evidence_strength(variable.get("evidence_strength", EvidenceStrength.WEAK.value)),
                        )
                    )
            for relationship in model.get("relationships") or []:
                for endpoint in ("from_variable", "to_variable"):
                    name = _normalize_whitespace(relationship.get(endpoint, ""))
                    if not name:
                        continue
                    mentions.append(
                        _VariableMention(
                            name=name,
                            definition=f"{name} inferred from relationships across chunks.",
                            var_type=VariableType.CONTEXTUAL,
                            quote=_normalize_whitespace(relationship.get("supporting_quote", "")),
                            chunk_id=chunk_id,
                            evidence_strength=_safe_evidence_strength(relationship.get("evidence_strength", EvidenceStrength.WEAK.value)),
                        )
                    )
        return mentions

    def _cluster_variables(self, mentions: Sequence[_VariableMention]) -> List[List[_VariableMention]]:
        clusters: List[List[_VariableMention]] = []
        for mention in mentions:
            placed = False
            for cluster in clusters:
                score = max(_variable_similarity(mention.name, existing.name) for existing in cluster)
                if score >= 0.86:
                    cluster.append(mention)
                    placed = True
                    break
            if not placed:
                clusters.append([mention])
        return clusters

    def _merge_variables(
        self,
        clusters: Sequence[Sequence[_VariableMention]],
        *,
        total_chunks: int,
    ) -> tuple[List[MergedVariable], Dict[str, str]]:
        merged: List[MergedVariable] = []
        alias_to_canonical: Dict[str, str] = {}

        for cluster in clusters:
            names = [_normalize_whitespace(v.name) for v in cluster]
            counts = Counter(names)
            canonical_name = max(counts, key=lambda item: (counts[item], len(item)))
            definitions = _dedupe_keep_order(v.definition for v in cluster if v.definition)
            definition = definitions[0] if definitions else f"{canonical_name} derived from chunk-level extractions."
            if len(definitions) > 1:
                definition = "; ".join(definitions[:3])
            type_value = _safe_variable_type(
                _pick_majority_counter([v.var_type.value for v in cluster if v.var_type != VariableType.CONTEXTUAL], VariableType.CONTEXTUAL.value)
            )
            evidence = _safe_evidence_strength(
                _pick_majority_counter([v.evidence_strength.value for v in cluster], EvidenceStrength.WEAK.value)
            )
            source_chunk_ids = sorted({v.chunk_id for v in cluster if v.chunk_id})
            chunk_frequency = len(source_chunk_ids)
            confidence = round(min(0.98, 0.45 + (chunk_frequency / max(1, total_chunks)) * 0.55), 3)
            aliases = [name for name in _dedupe_keep_order(names) if name.lower() != canonical_name.lower()]

            merged_variable = MergedVariable(
                name=canonical_name,
                aliases=aliases,
                definition=definition,
                type=type_value,
                chunk_frequency=chunk_frequency,
                confidence=confidence,
                source_chunk_ids=source_chunk_ids,
                supporting_quotes=_dedupe_keep_order(v.quote for v in cluster if v.quote)[:8],
                evidence_strength=evidence,
            )
            merged.append(merged_variable)
            for alias in [canonical_name, *aliases]:
                alias_to_canonical[alias.lower()] = canonical_name

        merged.sort(key=lambda item: (-item.chunk_frequency, item.name.lower()))
        return merged, alias_to_canonical

    def _collect_relationship_mentions(
        self,
        successful: Sequence[Dict[str, Any]],
        alias_to_canonical: Mapping[str, str],
    ) -> List[_RelationshipMention]:
        mentions: List[_RelationshipMention] = []
        for row in successful:
            chunk_id = str(row.get("chunk_id", ""))
            for relationship in (row.get("model") or {}).get("relationships") or []:
                from_variable = self._canonical_name(str(relationship.get("from_variable", "")), alias_to_canonical)
                to_variable = self._canonical_name(str(relationship.get("to_variable", "")), alias_to_canonical)
                if not from_variable or not to_variable:
                    continue
                mentions.append(
                    _RelationshipMention(
                        from_variable=from_variable,
                        to_variable=to_variable,
                        direction=_safe_direction(relationship.get("direction", RelationshipDirection.UNCLEAR.value)),
                        mechanism=_normalize_whitespace(relationship.get("mechanism", "")),
                        quote=_normalize_whitespace(relationship.get("supporting_quote", "")),
                        confidence=float(relationship.get("confidence", 0.5) or 0.5),
                        chunk_id=chunk_id,
                        evidence_strength=_safe_evidence_strength(relationship.get("evidence_strength", EvidenceStrength.WEAK.value)),
                    )
                )
        return mentions

    def _merge_relationships(
        self,
        mentions: Sequence[_RelationshipMention],
        *,
        total_chunks: int,
    ) -> List[ConsolidatedRelationship]:
        buckets: dict[tuple[str, str, str], list[_RelationshipMention]] = defaultdict(list)
        for mention in mentions:
            buckets[(mention.from_variable, mention.to_variable, mention.direction.value)].append(mention)

        relationships: List[ConsolidatedRelationship] = []
        for (from_variable, to_variable, direction), rows in buckets.items():
            source_chunk_ids = sorted({row.chunk_id for row in rows if row.chunk_id})
            support_count = len(source_chunk_ids)
            support_fraction = support_count / max(1, total_chunks)
            avg_individual_conf = sum(row.confidence for row in rows) / max(1, len(rows))
            confidence = round(min(0.99, (support_fraction * 0.6) + (avg_individual_conf * 0.4)), 3)
            mechanism = _pick_majority_counter([row.mechanism for row in rows if row.mechanism], "Mechanism needs researcher review.")
            evidence = _safe_evidence_strength(
                _pick_majority_counter([row.evidence_strength.value for row in rows], EvidenceStrength.WEAK.value)
            )
            relationships.append(
                ConsolidatedRelationship(
                    from_variable=from_variable,
                    to_variable=to_variable,
                    direction=_safe_direction(direction),
                    mechanism=mechanism,
                    confidence=confidence,
                    support_count=support_count,
                    support_fraction=round(support_fraction, 3),
                    source_chunk_ids=source_chunk_ids,
                    supporting_quotes=_dedupe_keep_order(row.quote for row in rows if row.quote)[:10],
                    contradicting_quotes=[],
                    evidence_strength=evidence,
                )
            )

        relationships.sort(key=lambda item: (-item.confidence, item.from_variable.lower(), item.to_variable.lower()))
        return relationships

    def _attach_contradicting_quotes(self, relationships: List[ConsolidatedRelationship]) -> None:
        by_pair: dict[tuple[str, str], list[ConsolidatedRelationship]] = defaultdict(list)
        for relationship in relationships:
            by_pair[(relationship.from_variable, relationship.to_variable)].append(relationship)

        for variants in by_pair.values():
            if len(variants) < 2:
                continue
            for relationship in variants:
                opposing_quotes: List[str] = []
                for other in variants:
                    if other is relationship:
                        continue
                    opposing_quotes.extend(other.supporting_quotes)
                relationship.contradicting_quotes = _dedupe_keep_order(opposing_quotes)[:8]

    def _collect_hypothesis_mentions(
        self,
        successful: Sequence[Dict[str, Any]],
        alias_to_canonical: Mapping[str, str],
        consolidated_relationships: Sequence[ConsolidatedRelationship],
    ) -> List[_HypothesisMention]:
        mentions: List[_HypothesisMention] = []
        relationship_pairs = {
            (relationship.from_variable, relationship.to_variable): relationship.direction
            for relationship in consolidated_relationships
        }
        for row in successful:
            chunk_id = str(row.get("chunk_id", ""))
            model = row.get("model") or {}
            for hypothesis in model.get("hypotheses") or []:
                statement = _normalize_whitespace(hypothesis.get("statement", ""))
                if not statement:
                    continue
                quotes = hypothesis.get("supporting_quotes") or []
                canonical_from = ""
                canonical_to = ""
                direction = _safe_direction(hypothesis.get("direction", RelationshipDirection.UNCLEAR.value))
                for from_variable, to_variable in relationship_pairs:
                    if from_variable.lower() in statement.lower() and to_variable.lower() in statement.lower():
                        canonical_from = from_variable
                        canonical_to = to_variable
                        direction = relationship_pairs[(from_variable, to_variable)]
                        break
                mentions.append(
                    _HypothesisMention(
                        statement=statement,
                        hypothesis_id=_normalize_whitespace(hypothesis.get("id", "")) or "H?",
                        quote=_normalize_whitespace(quotes[0] if quotes else ""),
                        chunk_id=chunk_id,
                        evidence_strength=_safe_evidence_strength(hypothesis.get("evidence_strength", EvidenceStrength.WEAK.value)),
                        from_variable=self._canonical_name(canonical_from, alias_to_canonical),
                        to_variable=self._canonical_name(canonical_to, alias_to_canonical),
                        direction=direction,
                    )
                )

        for idx, relationship in enumerate(consolidated_relationships, start=1):
            statement = f"{relationship.from_variable} has a {relationship.direction.value} effect on {relationship.to_variable}."
            mentions.append(
                _HypothesisMention(
                    statement=statement,
                    hypothesis_id=f"H{idx}",
                    quote=relationship.supporting_quotes[0] if relationship.supporting_quotes else "",
                    chunk_id=relationship.source_chunk_ids[0] if relationship.source_chunk_ids else "",
                    evidence_strength=relationship.evidence_strength,
                    from_variable=relationship.from_variable,
                    to_variable=relationship.to_variable,
                    direction=relationship.direction,
                )
            )

        return mentions

    def _merge_hypotheses(
        self,
        mentions: Sequence[_HypothesisMention],
        consolidated_relationships: Sequence[ConsolidatedRelationship],
        *,
        total_chunks: int,
    ) -> List[ScoredHypothesis]:
        linked_relationships = {
            f"{relationship.from_variable} -> {relationship.to_variable} ({relationship.direction.value})": relationship
            for relationship in consolidated_relationships
        }
        buckets: dict[str, list[_HypothesisMention]] = defaultdict(list)
        for mention in mentions:
            key = _normalize_whitespace(mention.statement).lower()
            if key:
                buckets[key].append(mention)

        hypotheses: List[ScoredHypothesis] = []
        for idx, rows in enumerate(buckets.values(), start=1):
            best = max(rows, key=lambda item: len(item.statement))
            source_chunk_ids = sorted({row.chunk_id for row in rows if row.chunk_id})
            support_count = len(source_chunk_ids)
            support_fraction = support_count / max(1, total_chunks)
            relationship_keys = [
                key
                for key, relationship in linked_relationships.items()
                if relationship.from_variable == best.from_variable
                and relationship.to_variable == best.to_variable
                and relationship.direction == best.direction
            ]
            related_confidences = [linked_relationships[key].confidence for key in relationship_keys]
            base_confidence = sum(related_confidences) / max(1, len(related_confidences)) if related_confidences else 0.55
            contradiction_penalty = 0.15 if any(linked_relationships[key].contradicting_quotes for key in relationship_keys) else 0.0
            confidence = round(max(0.0, min(0.99, (support_fraction * 0.5) + (base_confidence * 0.5) - contradiction_penalty)), 3)
            contradicting_quotes: List[str] = []
            for key in relationship_keys:
                contradicting_quotes.extend(linked_relationships[key].contradicting_quotes)
            consensus_strength = ConsensusStrength.MODERATE if confidence >= 0.7 else ConsensusStrength.WEAK
            hypotheses.append(
                ScoredHypothesis(
                    id=f"H{idx}",
                    statement=best.statement,
                    confidence=confidence,
                    support_count=support_count,
                    support_fraction=round(support_fraction, 3),
                    source_chunk_ids=source_chunk_ids,
                    supporting_quotes=_dedupe_keep_order(row.quote for row in rows if row.quote)[:8],
                    contradicting_quotes=_dedupe_keep_order(contradicting_quotes)[:8],
                    linked_relationships=relationship_keys,
                    from_variable=best.from_variable,
                    to_variable=best.to_variable,
                    direction=best.direction,
                    evidence_strength=_safe_evidence_strength(
                        _pick_majority_counter([row.evidence_strength.value for row in rows], EvidenceStrength.WEAK.value)
                    ),
                    consensus_strength=consensus_strength,
                )
            )

        hypotheses.sort(key=lambda item: (-item.confidence, item.id))
        return hypotheses

    def _merge_moderators(
        self,
        successful: Sequence[Dict[str, Any]],
        alias_to_canonical: Mapping[str, str],
    ) -> List[ModeratorSpec]:
        buckets: dict[tuple[str, str], _ModeratorMention] = {}
        for row in successful:
            chunk_id = str(row.get("chunk_id", ""))
            model = row.get("model") or {}
            relationships = model.get("relationships") or []
            target_relationship = ""
            if relationships:
                rel = relationships[0]
                target_relationship = (
                    f"{self._canonical_name(rel.get('from_variable', ''), alias_to_canonical)} -> "
                    f"{self._canonical_name(rel.get('to_variable', ''), alias_to_canonical)}"
                )

            for moderator in model.get("moderators") or []:
                name = self._canonical_name(str(moderator.get("name", "")), alias_to_canonical)
                if not name:
                    continue
                key = (name.lower(), target_relationship.lower())
                existing = buckets.get(key)
                quote = _normalize_whitespace(moderator.get("example_quote", ""))
                rationale = _normalize_whitespace(moderator.get("definition", "")) or "Moderator inferred from chunk context."
                if existing is None:
                    buckets[key] = _ModeratorMention(
                        name=name,
                        quote=quote,
                        chunk_id=chunk_id,
                        target_relationship=target_relationship,
                        rationale=rationale,
                    )
                else:
                    if quote and quote.lower() not in existing.quote.lower():
                        existing.quote = existing.quote + " | " + quote if existing.quote else quote
                    existing.rationale = existing.rationale or rationale

        moderators = [
            ModeratorSpec(
                name=mention.name,
                target_relationship=mention.target_relationship,
                rationale=mention.rationale,
                source_chunk_ids=[mention.chunk_id] if mention.chunk_id else [],
                supporting_quotes=_dedupe_keep_order(mention.quote.split(" | ")) if mention.quote else [],
            )
            for mention in buckets.values()
        ]
        moderators.sort(key=lambda item: (item.name.lower(), item.target_relationship.lower()))
        return moderators

    def _build_model_summary(
        self,
        variables: Sequence[MergedVariable],
        relationships: Sequence[ConsolidatedRelationship],
        hypotheses: Sequence[ScoredHypothesis],
    ) -> str:
        if not relationships:
            return "No consolidated relationships were strong enough to summarize yet. Review extraction quality and unresolved gaps before interpreting the model."
        top_relationships = relationships[:2]
        rel_summaries = [
            f"{rel.from_variable} shows a {rel.direction.value} link to {rel.to_variable} (confidence {rel.confidence:.2f})"
            for rel in top_relationships
        ]
        variable_summary = f"{len(variables)} consolidated variables and {len(relationships)} cross-chunk relationships were merged"
        hypothesis_summary = (
            f"{len(hypotheses)} scored hypotheses are ready for review"
            if hypotheses
            else "no scored hypotheses were retained after consolidation"
        )
        return f"{variable_summary}. {'; '.join(rel_summaries)}. {hypothesis_summary}."

    def _build_research_questions(
        self,
        gap_report: Mapping[str, Any] | None,
        clarification_plan: Mapping[str, Any] | None,
        relationships: Sequence[ConsolidatedRelationship],
    ) -> List[str]:
        questions = [
            str(question.get("question_text", "")).strip()
            for question in (clarification_plan or {}).get("questions", [])[:5]
            if str(question.get("question_text", "")).strip()
        ]
        if questions:
            return _dedupe_keep_order(questions)[:5]

        derived = [
            f"What boundary conditions explain the {relationship.direction.value} effect from {relationship.from_variable} to {relationship.to_variable}?"
            for relationship in relationships
            if relationship.direction in {RelationshipDirection.CONDITIONAL, RelationshipDirection.UNCLEAR}
        ]
        if derived:
            return _dedupe_keep_order(derived)[:5]

        fallback = [
            str(description).strip()
            for description in (gap_report or {}).get("priority_gaps", [])[:5]
            if str(description).strip()
        ]
        return fallback

    @staticmethod
    def _overall_confidence(
        relationships: Sequence[ConsolidatedRelationship],
        hypotheses: Sequence[ScoredHypothesis],
    ) -> float:
        scores = [rel.confidence for rel in relationships] + [hyp.confidence for hyp in hypotheses]
        if not scores:
            return 0.0
        return round(sum(scores) / len(scores), 3)

    @staticmethod
    def _canonical_name(name: str, alias_to_canonical: Mapping[str, str]) -> str:
        cleaned = _normalize_whitespace(name)
        if not cleaned:
            return ""
        return alias_to_canonical.get(cleaned.lower(), cleaned)


class ConflictDetector:
    """Detect contradictions and attempt lightweight deterministic resolution."""

    _IGNORE_METADATA_KEYS = {
        "chunk_id",
        "content_hash",
        "language",
        "original_index",
        "sentence_count",
        "sentiment",
        "source_type",
        "subjectivity",
        "timestamp",
        "word_count",
    }

    def detect(
        self,
        consolidated_model: ConsolidatedModel,
        extraction_results: Sequence[Dict[str, Any]],
        literature_store: Any | None = None,
    ) -> ConflictReport:
        contradictions: List[Contradiction] = []
        relationship_lookup: dict[tuple[str, str], list[ConsolidatedRelationship]] = defaultdict(list)
        for relationship in consolidated_model.relationships:
            relationship_lookup[(relationship.from_variable, relationship.to_variable)].append(relationship)

        chunk_metadata_by_id = {
            str(row.get("chunk_id", "")): dict(row.get("chunk_metadata") or {})
            for row in extraction_results
            if row.get("chunk_id")
        }

        for (from_variable, to_variable), variants in relationship_lookup.items():
            if len(variants) < 2:
                continue
            variants = sorted(variants, key=lambda item: item.confidence, reverse=True)
            version_a = variants[0]
            version_b = variants[1]
            if version_a.direction == version_b.direction:
                continue

            attempted = "subgroup metadata, literature scan, confidence differential"
            status = ResolutionStatus.UNRESOLVED
            conflict_type = ContradictionType.DIRECTION_CONFLICT
            explanation = "Opposing relationship directions were detected and no deterministic resolution was strong enough."

            subgroup_resolution = self._resolve_by_subgroup(version_a, version_b, chunk_metadata_by_id)
            if subgroup_resolution is not None:
                status = ResolutionStatus.RESOLVED
                conflict_type = ContradictionType.SUBGROUP_DIFFERENCE
                explanation = subgroup_resolution
            else:
                literature_resolution = self._resolve_by_literature(version_a, version_b, literature_store)
                if literature_resolution is not None:
                    status = ResolutionStatus.PARTIALLY_RESOLVED
                    explanation = literature_resolution
                else:
                    differential = version_a.confidence - version_b.confidence
                    if differential >= 0.2:
                        status = ResolutionStatus.PARTIALLY_RESOLVED
                        explanation = (
                            f"Kept {version_a.direction.value} as the leading direction because its confidence "
                            f"({version_a.confidence:.2f}) exceeded the alternative ({version_b.confidence:.2f})."
                        )

            contradictions.append(
                Contradiction(
                    relationship=f"{from_variable} -> {to_variable}",
                    conflict_type=conflict_type,
                    version_a=(
                        f"{version_a.direction.value} effect ({version_a.support_count} chunks, "
                        f"conf: {version_a.confidence:.2f})"
                    ),
                    version_b=(
                        f"{version_b.direction.value} effect ({version_b.support_count} chunks, "
                        f"conf: {version_b.confidence:.2f})"
                    ),
                    resolution_attempted=attempted,
                    resolution_status=status,
                    resolution_explanation=explanation,
                    requires_researcher_input=status == ResolutionStatus.UNRESOLVED,
                )
            )

        resolved_count = sum(1 for c in contradictions if c.resolution_status == ResolutionStatus.RESOLVED)
        partially_resolved_count = sum(1 for c in contradictions if c.resolution_status == ResolutionStatus.PARTIALLY_RESOLVED)
        unresolved_count = sum(1 for c in contradictions if c.resolution_status == ResolutionStatus.UNRESOLVED)
        return ConflictReport(
            contradictions=contradictions,
            resolved_count=resolved_count,
            partially_resolved_count=partially_resolved_count,
            unresolved_count=unresolved_count,
        )

    def _resolve_by_subgroup(
        self,
        version_a: ConsolidatedRelationship,
        version_b: ConsolidatedRelationship,
        chunk_metadata_by_id: Mapping[str, Mapping[str, Any]],
    ) -> str | None:
        if not version_a.source_chunk_ids or not version_b.source_chunk_ids:
            return None
        for key in sorted(
            {
                *chunk_metadata_by_id.get(version_a.source_chunk_ids[0], {}).keys(),
                *chunk_metadata_by_id.get(version_b.source_chunk_ids[0], {}).keys(),
            }
        ):
            if key in self._IGNORE_METADATA_KEYS:
                continue
            values_a = {
                _normalize_whitespace(chunk_metadata_by_id.get(chunk_id, {}).get(key, ""))
                for chunk_id in version_a.source_chunk_ids
            }
            values_b = {
                _normalize_whitespace(chunk_metadata_by_id.get(chunk_id, {}).get(key, ""))
                for chunk_id in version_b.source_chunk_ids
            }
            values_a.discard("")
            values_b.discard("")
            if values_a and values_b and values_a.isdisjoint(values_b):
                return (
                    f"Opposing directions line up with different `{key}` values: "
                    f"{sorted(values_a)} vs {sorted(values_b)}. Treat this relationship as subgroup-conditional."
                )
        return None

    def _resolve_by_literature(
        self,
        version_a: ConsolidatedRelationship,
        version_b: ConsolidatedRelationship,
        literature_store: Any | None,
    ) -> str | None:
        if literature_store is None:
            return None
        try:
            matches = literature_store.query(
                f"{version_a.from_variable} {version_a.to_variable} {version_a.direction.value}",
                k=4,
            )
        except (OSError, RuntimeError, ValueError, TypeError, KeyError, AttributeError):
            return None
        if not matches:
            return None

        score_a = 0.0
        score_b = 0.0
        for match in matches:
            text = _normalize_whitespace(match.get("text", "")).lower()
            citation_count = int((match.get("metadata") or {}).get("citation_count", 0) or 0)
            weight = math.log(citation_count + 1) + 1.0
            if self._text_matches_direction(text, version_a.direction):
                score_a += weight
            if self._text_matches_direction(text, version_b.direction):
                score_b += weight

        if score_a == score_b:
            return None
        favored = version_a if score_a > score_b else version_b
        return (
            f"Related literature leaned toward a {favored.direction.value} effect, so that direction remains primary "
            "and the alternative is retained as contesting evidence."
        )

    @staticmethod
    def _text_matches_direction(text: str, direction: RelationshipDirection) -> bool:
        if direction == RelationshipDirection.POSITIVE:
            return any(cue in text for cue in _POSITIVE_CUES)
        if direction == RelationshipDirection.NEGATIVE:
            return any(cue in text for cue in _NEGATIVE_CUES)
        return "conditional" in text or "moderat" in text


class LiteratureValidator:
    """Validate consolidated hypotheses against the literature store."""

    def validate(
        self,
        hypotheses: Sequence[ScoredHypothesis],
        literature_store: Any | None,
    ) -> LiteratureValidationReport:
        validations: List[HypothesisValidation] = []
        if literature_store is None:
            return LiteratureValidationReport(validations=[], strong_support_count=0, contested_count=0, novelty_count=0)

        for hypothesis in hypotheses:
            validations.append(self._validate_one(hypothesis, literature_store))

        strong_support_count = sum(1 for v in validations if v.consensus_strength == ConsensusStrength.STRONG)
        contested_count = sum(1 for v in validations if v.consensus_strength == ConsensusStrength.CONTESTED)
        novelty_count = sum(1 for v in validations if v.novelty_flag)
        return LiteratureValidationReport(
            validations=validations,
            strong_support_count=strong_support_count,
            contested_count=contested_count,
            novelty_count=novelty_count,
        )

    def _validate_one(self, hypothesis: ScoredHypothesis, literature_store: Any) -> HypothesisValidation:
        query = hypothesis.statement
        if hypothesis.from_variable and hypothesis.to_variable:
            query = f"{hypothesis.from_variable} {hypothesis.to_variable} {hypothesis.direction.value}"

        try:
            matches = literature_store.query(query, k=5)
        except (OSError, RuntimeError, ValueError, TypeError, KeyError, AttributeError):
            matches = []

        supporting: List[PaperReference] = []
        contradicting: List[PaperReference] = []
        partial: List[PaperReference] = []

        for match in matches:
            paper = self._to_paper_reference(match, hypothesis)
            if paper.stance.value == "supports":
                supporting.append(paper)
            elif paper.stance.value == "contradicts":
                contradicting.append(paper)
            else:
                partial.append(paper)

        support_score = self._calculate_literature_score(supporting, contradicting)
        novelty_flag = not supporting and not contradicting and hypothesis.confidence >= 0.7
        consensus_strength = self._consensus_strength(
            support_score=support_score,
            supporting=supporting,
            contradicting=contradicting,
            novelty_flag=novelty_flag,
        )
        return HypothesisValidation(
            hypothesis_id=hypothesis.id,
            hypothesis_statement=hypothesis.statement,
            supporting_papers=supporting,
            contradicting_papers=contradicting,
            partial_papers=partial,
            literature_support_score=support_score,
            consensus_strength=consensus_strength,
            novelty_flag=novelty_flag,
        )

    def _to_paper_reference(self, match: Mapping[str, Any], hypothesis: ScoredHypothesis) -> PaperReference:
        metadata = dict(match.get("metadata") or {})
        title = _normalize_whitespace(metadata.get("title", "Untitled")) or "Untitled"
        text = _normalize_whitespace(match.get("text", ""))
        stance = self._classify_stance(text.lower(), hypothesis)
        excerpt = self._relevant_excerpt(text, stance.value)
        authors_raw = metadata.get("authors", "")
        authors = [author.strip() for author in str(authors_raw).split(",") if author.strip()]
        return PaperReference(
            paper_id=str(metadata.get("paper_id", title)),
            title=title,
            authors=authors,
            year=int(metadata["year"]) if metadata.get("year") not in (None, "") else None,
            citation_count=int(metadata.get("citation_count", 0) or 0),
            relevant_excerpt=excerpt,
            stance=stance,
        )

    def _classify_stance(self, text: str, hypothesis: ScoredHypothesis):
        positive = any(cue in text for cue in _POSITIVE_CUES)
        negative = any(cue in text for cue in _NEGATIVE_CUES)
        if hypothesis.direction == RelationshipDirection.POSITIVE:
            if positive and not negative:
                return PaperStance.SUPPORTS
            if negative and not positive:
                return PaperStance.CONTRADICTS
        elif hypothesis.direction == RelationshipDirection.NEGATIVE:
            if negative and not positive:
                return PaperStance.SUPPORTS
            if positive and not negative:
                return PaperStance.CONTRADICTS
        return PaperStance.PARTIAL

    @staticmethod
    def _relevant_excerpt(text: str, stance: str) -> str:
        sentences = [segment.strip() for segment in re.split(r"(?<=[.!?])\s+", text) if segment.strip()]
        if not sentences:
            return ""
        if stance == "partial":
            return sentences[0]
        for sentence in sentences:
            lowered = sentence.lower()
            if stance == "supports" and any(cue in lowered for cue in _POSITIVE_CUES + _NEGATIVE_CUES):
                return sentence
            if stance == "contradicts" and any(cue in lowered for cue in _POSITIVE_CUES + _NEGATIVE_CUES):
                return sentence
        return sentences[0]

    @staticmethod
    def _calculate_literature_score(
        supporting: Sequence[PaperReference],
        contradicting: Sequence[PaperReference],
    ) -> float:
        if not supporting and not contradicting:
            return 0.0
        support_weight = sum(math.log(p.citation_count + 1) + 1.0 for p in supporting)
        contra_weight = sum(math.log(p.citation_count + 1) + 1.0 for p in contradicting)
        total = support_weight + contra_weight
        if total <= 0:
            return 0.5
        return round(support_weight / total, 3)

    @staticmethod
    def _consensus_strength(
        *,
        support_score: float,
        supporting: Sequence[PaperReference],
        contradicting: Sequence[PaperReference],
        novelty_flag: bool,
    ) -> ConsensusStrength:
        if novelty_flag:
            return ConsensusStrength.NOVEL
        if supporting and contradicting and abs(support_score - 0.5) <= 0.15:
            return ConsensusStrength.CONTESTED
        if support_score >= 0.75 and len(supporting) >= 2:
            return ConsensusStrength.STRONG
        if support_score >= 0.6 and supporting:
            return ConsensusStrength.MODERATE
        return ConsensusStrength.WEAK
