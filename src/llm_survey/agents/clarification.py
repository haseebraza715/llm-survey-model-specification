from __future__ import annotations

from typing import Any, Dict, List

from llm_survey.schemas.clarification import (
    AnswerSource,
    ClarificationAnswer,
    ClarificationPlan,
    ClarificationPriority,
    ClarificationQuestion,
)


class ClarificationAgent:
    """Build clarification plan from cross-chunk gaps and literature context."""

    _PRIORITY_ORDER = {
        ClarificationPriority.HIGH.value: 0,
        ClarificationPriority.MEDIUM.value: 1,
        ClarificationPriority.LOW.value: 2,
    }

    def build_plan(
        self,
        gap_report: Dict[str, Any],
        literature_store: Any | None = None,
        auto_answer_top_k: int = 3,
    ) -> ClarificationPlan:
        raw_gaps = list(gap_report.get("gaps", []))
        questions: List[ClarificationQuestion] = []
        auto_answers: List[ClarificationAnswer] = []

        for idx, gap in enumerate(raw_gaps, start=1):
            description = str(gap.get("description", "")).strip()
            if not description:
                continue

            priority = self._safe_priority(gap.get("priority", "medium"))
            question_text = self._normalize_question_text(
                str(gap.get("suggested_follow_up", "")).strip()
                or f"How should we address this gap: {description}?"
            )

            context_for_researcher = self._build_context(gap)
            answer_source = self._route_answer_source(gap, literature_store, question_text)

            question = ClarificationQuestion(
                question_id=f"Q{idx}",
                question_text=question_text,
                target_gap=description,
                priority=priority,
                answer_source=answer_source,
                context_for_researcher=context_for_researcher,
            )
            questions.append(question)

            if answer_source in {AnswerSource.LITERATURE, AnswerSource.EITHER} and literature_store is not None:
                auto_answer = self.auto_answer_from_literature(
                    question_id=question.question_id,
                    question_text=question.question_text,
                    literature_store=literature_store,
                    top_k=auto_answer_top_k,
                )
                if auto_answer is not None:
                    auto_answers.append(auto_answer)

        questions.sort(key=lambda q: (self._PRIORITY_ORDER[q.priority.value], q.question_id))

        estimated_new_data_needed = any(q.answer_source in {AnswerSource.RESEARCHER, AnswerSource.EITHER} for q in questions)
        can_proceed_with_literature = self._can_proceed_with_literature(questions, auto_answers)

        return ClarificationPlan(
            questions=questions,
            estimated_new_data_needed=estimated_new_data_needed,
            can_proceed_with_literature=can_proceed_with_literature,
            auto_answers=auto_answers,
        )

    def auto_answer_from_literature(
        self,
        question_id: str,
        question_text: str,
        literature_store: Any,
        top_k: int = 3,
    ) -> ClarificationAnswer | None:
        """Synthesize a concise answer from retrieved literature snippets."""
        try:
            matches = literature_store.query(question_text, k=max(1, top_k))
        except (OSError, RuntimeError, ValueError, TypeError, KeyError, AttributeError):
            return None

        if not matches:
            return None

        supporting_references: List[str] = []
        distilled_points: List[str] = []

        for match in matches[:top_k]:
            metadata = match.get("metadata", {}) or {}
            title = str(metadata.get("title", "Untitled")).strip() or "Untitled"
            source = str(metadata.get("source", "literature")).strip() or "literature"
            year = metadata.get("year")

            ref = f"[{source}] {title}" + (f" ({year})" if year else "")
            supporting_references.append(ref)

            text = str(match.get("text", "")).strip()
            if text:
                first_sentence = text.split(".")[0].strip()
                if first_sentence:
                    distilled_points.append(first_sentence)

        if not distilled_points:
            answer_text = "Relevant literature exists, but no concise abstract evidence was extracted."
        else:
            unique_points = self._dedupe_keep_order(distilled_points)
            answer_text = "; ".join(unique_points[:3]) + "."

        return ClarificationAnswer(
            question_id=question_id,
            answer_source=AnswerSource.LITERATURE,
            answer_text=answer_text,
            supporting_references=self._dedupe_keep_order(supporting_references),
        )

    def _route_answer_source(self, gap: Dict[str, Any], literature_store: Any | None, question_text: str) -> AnswerSource:
        gap_type = str(gap.get("gap_type", "")).strip().lower()
        priority = str(gap.get("priority", "medium")).strip().lower()

        literature_available = False
        if literature_store is not None:
            try:
                literature_available = len(literature_store.query(question_text, k=1)) > 0
            except (OSError, RuntimeError, ValueError, TypeError, KeyError, AttributeError):
                literature_available = False

        if gap_type in {"no_measurement", "missing_mechanism"}:
            return AnswerSource.LITERATURE if literature_available else AnswerSource.EITHER

        if gap_type == "ambiguous_direction":
            if priority == "high":
                return AnswerSource.RESEARCHER
            return AnswerSource.EITHER if literature_available else AnswerSource.RESEARCHER

        if gap_type == "missing_variable":
            return AnswerSource.EITHER if literature_available else AnswerSource.RESEARCHER

        return AnswerSource.EITHER if literature_available else AnswerSource.RESEARCHER

    @staticmethod
    def _build_context(gap: Dict[str, Any]) -> str:
        affected = gap.get("affected_hypotheses") or []
        affected_text = ", ".join(str(h) for h in affected if str(h).strip())
        base = str(gap.get("description", "")).strip()
        if affected_text:
            return f"{base} This impacts hypotheses: {affected_text}."
        return base or "This gap reduces model completeness and testability."

    @staticmethod
    def _safe_priority(value: str) -> ClarificationPriority:
        try:
            return ClarificationPriority(str(value).lower())
        except ValueError:
            return ClarificationPriority.MEDIUM

    @staticmethod
    def _normalize_question_text(text: str) -> str:
        text = " ".join(text.split()).strip()
        if not text:
            return "What additional information is needed to resolve this gap?"
        if text.endswith("?"):
            return text
        return f"{text}?"

    @staticmethod
    def _can_proceed_with_literature(
        questions: List[ClarificationQuestion],
        auto_answers: List[ClarificationAnswer],
    ) -> bool:
        if not questions:
            return True

        answerable_ids = {a.question_id for a in auto_answers if a.answer_text.strip()}

        for q in questions:
            if q.answer_source == AnswerSource.RESEARCHER:
                return False
            if q.answer_source in {AnswerSource.LITERATURE, AnswerSource.EITHER} and q.question_id not in answerable_ids:
                return False

        return True

    @staticmethod
    def _dedupe_keep_order(values: List[str]) -> List[str]:
        seen = set()
        output: List[str] = []
        for value in values:
            key = value.strip().lower()
            if not key or key in seen:
                continue
            seen.add(key)
            output.append(value.strip())
        return output
