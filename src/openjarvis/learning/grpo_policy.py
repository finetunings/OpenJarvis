"""GRPO-based router policy — stub for Phase 5."""

from __future__ import annotations

from typing import Any

from openjarvis.core.registry import RouterPolicyRegistry
from openjarvis.core.types import RoutingContext
from openjarvis.intelligence._stubs import RouterPolicy


class GRPORouterPolicy(RouterPolicy):
    """Placeholder for GRPO-trained router policy (Phase 5).

    Raises ``NotImplementedError`` until training infrastructure is ready.
    """

    def __init__(self, **kwargs: Any) -> None:
        self._kwargs = kwargs

    def select_model(self, context: RoutingContext) -> str:
        raise NotImplementedError(
            "GRPORouterPolicy is not yet implemented. "
            "GRPO training will be available in Phase 5."
        )


def ensure_registered() -> None:
    """Register GRPORouterPolicy if not already present."""
    if not RouterPolicyRegistry.contains("grpo"):
        RouterPolicyRegistry.register_value("grpo", GRPORouterPolicy)


ensure_registered()

__all__ = ["GRPORouterPolicy"]
