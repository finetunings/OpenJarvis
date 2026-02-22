"""Learning pillar — router policy and reward functions."""

from __future__ import annotations

from openjarvis.learning._stubs import RewardFunction, RouterPolicy, RoutingContext
from openjarvis.learning.heuristic_reward import HeuristicRewardFunction


def ensure_registered() -> None:
    """Ensure all learning policies are registered in RouterPolicyRegistry.

    Imported lazily to avoid circular imports with the intelligence pillar.
    """
    from openjarvis.learning.heuristic_policy import (
        ensure_registered as _reg_heuristic,
    )

    _reg_heuristic()

    try:
        from openjarvis.learning.grpo_policy import (
            ensure_registered as _reg_grpo,
        )

        _reg_grpo()
    except ImportError:
        pass

    from openjarvis.learning.trace_policy import (
        ensure_registered as _reg_trace,
    )

    _reg_trace()

    try:
        import openjarvis.learning.sft_policy  # noqa: F401
    except ImportError:
        pass

    try:
        import openjarvis.learning.agent_advisor  # noqa: F401
    except ImportError:
        pass

    try:
        import openjarvis.learning.icl_updater  # noqa: F401
    except ImportError:
        pass


__all__ = [
    "HeuristicRewardFunction",
    "RewardFunction",
    "RouterPolicy",
    "RoutingContext",
    "ensure_registered",
]
