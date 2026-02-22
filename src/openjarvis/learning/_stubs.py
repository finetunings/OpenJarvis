"""Learning pillar ABCs and re-exports for backward compatibility."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, ClassVar, Dict

from openjarvis.core.registry import LearningRegistry  # noqa: F401

# Re-export from canonical locations for backward compatibility
from openjarvis.core.types import RoutingContext  # noqa: F401
from openjarvis.intelligence._stubs import RouterPolicy  # noqa: F401

if TYPE_CHECKING:
    pass


class RewardFunction(ABC):
    """Compute a scalar reward for a routing decision."""

    @abstractmethod
    def compute(
        self,
        context: "RoutingContext",
        model_key: str,
        response: str,
        **kwargs: object,
    ) -> float:
        """Return reward in [0, 1]."""


class LearningPolicy(ABC):
    """Base for all learning policies. Targets one or more pillars."""

    target: ClassVar[str] = ""  # "intelligence" | "agent" | "tools"

    @abstractmethod
    def update(self, trace_store: Any, **kwargs: object) -> Dict[str, Any]:
        """Analyze traces and return update actions."""


class IntelligenceLearningPolicy(LearningPolicy):
    """Updates intelligence (model routing) from traces."""

    target: ClassVar[str] = "intelligence"


class AgentLearningPolicy(LearningPolicy):
    """Updates agent logic from traces."""

    target: ClassVar[str] = "agent"


class ToolLearningPolicy(LearningPolicy):
    """Updates tool usage (ICL examples, skills) from traces."""

    target: ClassVar[str] = "tools"


__all__ = [
    "AgentLearningPolicy",
    "IntelligenceLearningPolicy",
    "LearningPolicy",
    "LearningRegistry",
    "RewardFunction",
    "RouterPolicy",
    "RoutingContext",
    "ToolLearningPolicy",
]
