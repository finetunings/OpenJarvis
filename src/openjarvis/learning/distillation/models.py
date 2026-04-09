"""Pydantic models and enums for the distillation subsystem.

This module defines the typed vocabulary used by the diagnose, plan, execute,
and record phases. Three model families:

- Enums: pillar / risk tier / op / trigger kind / autonomy mode / session status
- Edit + LearningPlan + FailureCluster: the teacher's frozen output
- LearningSession + EditOutcome + BenchmarkSnapshot: the durable session record

See spec §4 for the data model rationale.
"""

from __future__ import annotations

from enum import Enum

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class EditPillar(str, Enum):
    """Which OpenJarvis pillar an edit targets."""

    INTELLIGENCE = "intelligence"
    AGENT = "agent"
    TOOLS = "tools"
    ENGINE = "engine"  # designed-for; no v1 appliers ship


class EditRiskTier(str, Enum):
    """How an edit gets applied: auto, review queue, or manual-only."""

    AUTO = "auto"
    REVIEW = "review"
    MANUAL = "manual"


class EditOp(str, Enum):
    """The set of typed operations a teacher can propose.

    Each op corresponds to one EditApplier in v1 (or a refusing stub for
    deferred ops). The teacher cannot invent new ops — only choose from this
    set.
    """

    # Intelligence
    SET_MODEL_FOR_QUERY_CLASS = "set_model_for_query_class"
    SET_MODEL_PARAM = "set_model_param"

    # Agent
    PATCH_SYSTEM_PROMPT = "patch_system_prompt"
    REPLACE_SYSTEM_PROMPT = "replace_system_prompt"
    SET_AGENT_CLASS = "set_agent_class"
    SET_AGENT_PARAM = "set_agent_param"
    EDIT_FEW_SHOT_EXEMPLARS = "edit_few_shot_exemplars"

    # Tools
    ADD_TOOL_TO_AGENT = "add_tool_to_agent"
    REMOVE_TOOL_FROM_AGENT = "remove_tool_from_agent"
    EDIT_TOOL_DESCRIPTION = "edit_tool_description"

    # v2 placeholder — planner can emit, executor refuses with NotImplementedError
    LORA_FINETUNE = "lora_finetune"


class TriggerKind(str, Enum):
    """What kicked off a learning session."""

    SCHEDULED = "scheduled"
    CLUSTER = "cluster"
    USER_FLAG = "user_flag"
    ON_DEMAND = "on_demand"


class AutonomyMode(str, Enum):
    """How aggressively the orchestrator applies edits without review."""

    AUTO = "auto"  # all tiers auto-apply, ignore review tier
    TIERED = "tiered"  # default: respect per-edit risk tier
    MANUAL = "manual"  # everything goes to review queue (dry-run mode)


class SessionStatus(str, Enum):
    """Lifecycle states for a LearningSession.

    See spec §7.7 for the transition rules.
    """

    INITIATED = "initiated"
    DIAGNOSING = "diagnosing"
    PLANNING = "planning"
    EXECUTING = "executing"
    AWAITING_REVIEW = "awaiting_review"
    COMPLETED = "completed"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"


# ---------------------------------------------------------------------------
# Edit — atomic unit of change
# ---------------------------------------------------------------------------


from typing import Any  # noqa: E402  (kept with model definitions)

from pydantic import BaseModel, Field  # noqa: E402


class Edit(BaseModel):
    """One atomic edit to the OpenJarvis harness.

    Emitted by the LearningPlanner, consumed by an EditApplier. The teacher
    proposes the op, target, payload, rationale, and references; the planner
    overwrites ``risk_tier`` deterministically from the (pillar, op) lookup
    table — the teacher cannot pick its own tier.

    See spec §4.1.
    """

    id: str = Field(
        ...,
        description="UUID for this edit; also used as a footer in git commits.",
    )
    pillar: EditPillar = Field(..., description="Which OpenJarvis pillar is touched.")
    op: EditOp = Field(..., description="The typed operation to perform.")
    target: str = Field(
        ...,
        description="Dotted path to the target, e.g. 'agents.simple.system_prompt'.",
    )
    payload: dict[str, Any] = Field(
        default_factory=dict,
        description="Op-specific arguments. Schema depends on op.",
    )
    rationale: str = Field(
        ...,
        description="Teacher's natural-language reason for this edit.",
    )
    expected_improvement: str = Field(
        ...,
        description="Which failure cluster id this edit is intended to address.",
    )
    risk_tier: EditRiskTier = Field(
        ...,
        description="Set by the planner from a (pillar, op) lookup table.",
    )
    references: list[str] = Field(
        default_factory=list,
        description="Trace ids or benchmark task ids that justify this edit.",
    )
