"""SFT on traces — learns which model handles which query type best."""

from __future__ import annotations

from typing import Any, Dict, List

from openjarvis.core.registry import LearningRegistry
from openjarvis.learning._stubs import IntelligenceLearningPolicy


@LearningRegistry.register("sft")
class SFTPolicy(IntelligenceLearningPolicy):
    """SFT on traces: learns which model handles which query type best.

    Reads successful traces, groups by query class, and produces an
    updated query_class -> model mapping.
    """

    def __init__(self, *, min_samples: int = 5) -> None:
        self._min_samples = min_samples
        self._policy_map: Dict[str, str] = {}

    def update(self, trace_store: Any, **kwargs: object) -> Dict[str, Any]:
        """Analyze trace outcomes and update the policy map."""
        try:
            traces = trace_store.list_traces()
        except Exception:
            return {"updated": False, "reason": "Could not access trace store"}

        # Group traces by query class and model
        class_model_scores: Dict[str, Dict[str, List[float]]] = {}
        for trace in traces:
            query_class = self._classify_query(trace.query)
            model = trace.model or "unknown"
            outcome_score = 1.0 if trace.outcome == "success" else 0.0
            feedback = trace.feedback if trace.feedback is not None else 0.5

            composite = 0.6 * outcome_score + 0.4 * feedback

            if query_class not in class_model_scores:
                class_model_scores[query_class] = {}
            if model not in class_model_scores[query_class]:
                class_model_scores[query_class][model] = []
            class_model_scores[query_class][model].append(composite)

        # Update policy map: best model per query class
        changes = {}
        for qclass, model_scores in class_model_scores.items():
            best_model = None
            best_score = -1.0
            for model, scores in model_scores.items():
                if len(scores) >= self._min_samples:
                    avg = sum(scores) / len(scores)
                    if avg > best_score:
                        best_score = avg
                        best_model = model
            if best_model and best_model != self._policy_map.get(qclass):
                self._policy_map[qclass] = best_model
                changes[qclass] = best_model

        return {
            "updated": bool(changes),
            "changes": changes,
            "policy_map": dict(self._policy_map),
        }

    @staticmethod
    def _classify_query(query: str) -> str:
        """Classify a query into a category."""
        q = query.lower()
        if any(kw in q for kw in ("def ", "class ", "import ", "```", "function")):
            return "code"
        math_kws = ("solve", "integral", "equation", "derivative", "proof")
        if any(kw in q for kw in math_kws):
            return "math"
        if len(query.split()) < 10:
            return "short"
        if len(query.split()) > 100:
            return "long"
        return "general"

    @property
    def policy_map(self) -> Dict[str, str]:
        return dict(self._policy_map)


__all__ = ["SFTPolicy"]
