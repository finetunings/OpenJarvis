"""Intelligence pillar ABCs — model routing and query analysis."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from openjarvis.core.types import RoutingContext


class RouterPolicy(ABC):
    """Abstract interface for model selection policies."""

    @abstractmethod
    def select_model(self, context: "RoutingContext") -> str:
        """Select the best model key for the given routing context."""


class QueryAnalyzer(ABC):
    """Abstract interface for analyzing queries into routing contexts."""

    @abstractmethod
    def analyze(self, query: str, **kwargs: object) -> "RoutingContext":
        """Analyze a query and return a RoutingContext."""


__all__ = ["QueryAnalyzer", "RouterPolicy"]
