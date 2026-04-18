from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Set, Tuple

from llm_survey.schemas.gap import CrossChunkGap, CrossChunkGapReport, GapPriority, GapType


@dataclass
class _GapAccumulator:
    gap_type: GapType
    description: str
    frequency: int
    affected_hypotheses: Set[str]
    suggested_follow_up: str


class CrossChunkGapDetector:
    """Deterministic cross-chunk gap detection and scoring."""

    _GAP_WEIGHTS = {
        GapType.MISSING_VARIABLE: 1.0,
        GapType.NO_MEASUREMENT: 0.9,
        GapType.MISSING_MECHANISM: 0.8,
        GapType.AMBIGUOUS_DIRECTION: 0.7,
    }

    _TESTABILITY_WEIGHTS = {
        GapType.NO_MEASUREMENT: 1.0,
        GapType.AMBIGUOUS_DIRECTION: 0.85,
        GapType.MISSING_VARIABLE: 0.8,
        GapType.MISSING_MECHANISM: 0.7,
    }

    def detect(self, extraction_results: List[Dict[str, Any]]) -> CrossChunkGapReport:
        successful = [r for r in extraction_results if r.get("success") and isinstance(r.get("model"), dict)]
        total_chunks = len(successful)
        if total_chunks == 0:
            return CrossChunkGapReport(
                gaps=[],
                structural_coverage_score=0.0,
                model_testability_score=0.0,
                priority_gaps=[],
            )

        global_variable_names = self._collect_global_variable_names(successful)

        buckets: Dict[Tuple[str, str], _GapAccumulator] = {}

        for result in successful:
            model = result.get("model") or {}
            hypotheses = [str(h.get("id", "")).strip() for h in model.get("hypotheses", []) if h.get("id")]
            local_variables = {
                str(v.get("name", "")).strip().lower()
                for v in model.get("variables", [])
                if v.get("name")
            }

            self._ingest_explicit_gaps(buckets, model.get("gaps", []), hypotheses)
            self._ingest_relationship_gaps(
                buckets=buckets,
                relationships=model.get("relationships", []),
                local_variables=local_variables,
                global_variables=global_variable_names,
                hypotheses=hypotheses,
            )
            self._ingest_hypothesis_gaps(buckets, model.get("hypotheses", []))

        gaps = [self._to_schema_gap(acc, total_chunks=total_chunks) for acc in buckets.values()]
        gaps.sort(key=lambda g: (-g.frequency, g.priority.value, g.gap_type.value, g.description))

        structural_coverage_score = self._score_completeness(gaps, total_chunks=total_chunks)
        model_testability_score = self._score_testability(gaps, total_chunks=total_chunks)

        priority_gaps = [gap.description for gap in gaps if gap.priority == GapPriority.HIGH][:3]
        if len(priority_gaps) < 3:
            for gap in gaps:
                if gap.description not in priority_gaps:
                    priority_gaps.append(gap.description)
                if len(priority_gaps) == 3:
                    break

        return CrossChunkGapReport(
            gaps=gaps,
            structural_coverage_score=structural_coverage_score,
            model_testability_score=model_testability_score,
            priority_gaps=priority_gaps,
        )

    def _collect_global_variable_names(self, successful_results: List[Dict[str, Any]]) -> Set[str]:
        names: Set[str] = set()
        for result in successful_results:
            model = result.get("model") or {}
            for variable in model.get("variables", []):
                name = str(variable.get("name", "")).strip().lower()
                if name:
                    names.add(name)
        return names

    def _ingest_explicit_gaps(
        self,
        buckets: Dict[Tuple[str, str], _GapAccumulator],
        gaps: Iterable[Dict[str, Any]],
        hypotheses: List[str],
    ) -> None:
        seen_keys: Set[Tuple[str, str]] = set()
        for gap in gaps or []:
            description = str(gap.get("description", "")).strip()
            why = str(gap.get("why_it_matters", "")).strip()
            follow_up = str(gap.get("suggested_question", "")).strip()
            if not description:
                continue

            gap_type = self._infer_gap_type(f"{description} {why} {follow_up}".lower())
            key = (gap_type.value, self._normalize_text(description))
            if key in seen_keys:
                continue
            seen_keys.add(key)

            self._add_gap(
                buckets=buckets,
                gap_type=gap_type,
                description=description,
                follow_up=follow_up or self._default_follow_up(gap_type, description),
                hypotheses=hypotheses,
            )

    def _ingest_relationship_gaps(
        self,
        buckets: Dict[Tuple[str, str], _GapAccumulator],
        relationships: Iterable[Dict[str, Any]],
        local_variables: Set[str],
        global_variables: Set[str],
        hypotheses: List[str],
    ) -> None:
        for rel in relationships or []:
            from_var = str(rel.get("from_variable", "")).strip().lower()
            to_var = str(rel.get("to_variable", "")).strip().lower()
            direction = str(rel.get("direction", "")).strip().lower()
            mechanism = str(rel.get("mechanism", "")).strip()

            missing_vars = [v for v in (from_var, to_var) if v and v not in local_variables and v not in global_variables]
            if missing_vars:
                description = (
                    "Relationship references variables that are not defined in extracted variable lists: "
                    + ", ".join(sorted(set(missing_vars)))
                )
                self._add_gap(
                    buckets=buckets,
                    gap_type=GapType.MISSING_VARIABLE,
                    description=description,
                    follow_up="Which exact constructs define these referenced variables and how should they be measured?",
                    hypotheses=hypotheses,
                )

            if direction in {"", "unclear", "conditional"}:
                description = "Relationship direction is unclear or conditional, reducing interpretability."
                self._add_gap(
                    buckets=buckets,
                    gap_type=GapType.AMBIGUOUS_DIRECTION,
                    description=description,
                    follow_up="What is the expected direction of effect and under which specific boundary conditions?",
                    hypotheses=hypotheses,
                )

            if len(mechanism) < 12:
                description = "Relationship mechanism is missing or underspecified for one or more extracted links."
                self._add_gap(
                    buckets=buckets,
                    gap_type=GapType.MISSING_MECHANISM,
                    description=description,
                    follow_up="What mechanism explains how the source variable influences the target variable?",
                    hypotheses=hypotheses,
                )

    def _ingest_hypothesis_gaps(
        self,
        buckets: Dict[Tuple[str, str], _GapAccumulator],
        hypotheses: Iterable[Dict[str, Any]],
    ) -> None:
        for hyp in hypotheses or []:
            hyp_id = str(hyp.get("id", "")).strip() or "unknown_hypothesis"
            supporting_quotes = hyp.get("supporting_quotes")
            has_quotes = isinstance(supporting_quotes, list) and len([q for q in supporting_quotes if str(q).strip()]) > 0
            if has_quotes:
                continue

            self._add_gap(
                buckets=buckets,
                gap_type=GapType.NO_MEASUREMENT,
                description="Hypotheses are missing supporting evidence quotes, limiting testability.",
                follow_up="What concrete evidence or observable indicators support each hypothesis?",
                hypotheses=[hyp_id],
            )

    def _add_gap(
        self,
        buckets: Dict[Tuple[str, str], _GapAccumulator],
        gap_type: GapType,
        description: str,
        follow_up: str,
        hypotheses: List[str],
    ) -> None:
        normalized_description = self._normalize_text(description)
        key = (gap_type.value, normalized_description)
        if key not in buckets:
            buckets[key] = _GapAccumulator(
                gap_type=gap_type,
                description=description.strip(),
                frequency=0,
                affected_hypotheses=set(),
                suggested_follow_up=follow_up.strip(),
            )

        buckets[key].frequency += 1
        buckets[key].affected_hypotheses.update([h for h in hypotheses if h])

    def _to_schema_gap(self, acc: _GapAccumulator, total_chunks: int) -> CrossChunkGap:
        priority = self._priority_for(acc.gap_type, frequency=acc.frequency, total_chunks=total_chunks)
        return CrossChunkGap(
            gap_type=acc.gap_type,
            description=acc.description,
            affected_hypotheses=sorted(acc.affected_hypotheses),
            frequency=acc.frequency,
            priority=priority,
            suggested_follow_up=acc.suggested_follow_up,
        )

    def _priority_for(self, gap_type: GapType, frequency: int, total_chunks: int) -> GapPriority:
        normalized_frequency = frequency / max(1, total_chunks)
        severity_boost = {
            GapType.MISSING_VARIABLE: 0.35,
            GapType.NO_MEASUREMENT: 0.30,
            GapType.MISSING_MECHANISM: 0.20,
            GapType.AMBIGUOUS_DIRECTION: 0.15,
        }[gap_type]
        score = normalized_frequency + severity_boost

        if score >= 1.0:
            return GapPriority.HIGH
        if score >= 0.55:
            return GapPriority.MEDIUM
        return GapPriority.LOW

    def _score_completeness(self, gaps: List[CrossChunkGap], total_chunks: int) -> float:
        if total_chunks <= 0:
            return 0.0

        weighted_sum = 0.0
        for gap in gaps:
            weighted_sum += self._GAP_WEIGHTS[gap.gap_type] * min(gap.frequency, total_chunks)

        max_possible = max(1.0, total_chunks * 4.0)
        score = 1.0 - min(1.0, weighted_sum / max_possible)
        return round(max(0.0, min(1.0, score)), 3)

    def _score_testability(self, gaps: List[CrossChunkGap], total_chunks: int) -> float:
        if total_chunks <= 0:
            return 0.0

        weighted_sum = 0.0
        for gap in gaps:
            weighted_sum += self._TESTABILITY_WEIGHTS[gap.gap_type] * min(gap.frequency, total_chunks)

        max_possible = max(1.0, total_chunks * 4.0)
        score = 1.0 - min(1.0, weighted_sum / max_possible)
        return round(max(0.0, min(1.0, score)), 3)

    @staticmethod
    def _infer_gap_type(text: str) -> GapType:
        if any(token in text for token in ("measure", "measurement", "operational", "indicator")):
            return GapType.NO_MEASUREMENT
        if any(token in text for token in ("direction", "unclear", "ambiguous", "conditional")):
            return GapType.AMBIGUOUS_DIRECTION
        if any(token in text for token in ("variable", "construct", "missing factor")):
            return GapType.MISSING_VARIABLE
        return GapType.MISSING_MECHANISM

    @staticmethod
    def _default_follow_up(gap_type: GapType, description: str) -> str:
        if gap_type == GapType.NO_MEASUREMENT:
            return "What measurable indicators can be used to operationalize this element?"
        if gap_type == GapType.AMBIGUOUS_DIRECTION:
            return "What is the expected direction of effect and when might it change?"
        if gap_type == GapType.MISSING_VARIABLE:
            return "Which missing variable should be explicitly added and defined?"
        return "What mechanism details are needed to explain this relationship clearly?"

    @staticmethod
    def _normalize_text(text: str) -> str:
        return " ".join(text.strip().lower().split())
