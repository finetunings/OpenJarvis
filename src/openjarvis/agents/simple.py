"""SimpleAgent — single-turn query-to-response agent (no tool calling)."""

from __future__ import annotations

from typing import Any, Optional

from openjarvis.agents._stubs import AgentContext, AgentResult, BaseAgent
from openjarvis.core.events import EventBus, EventType
from openjarvis.core.registry import AgentRegistry
from openjarvis.core.types import Message, Role
from openjarvis.engine._stubs import InferenceEngine


@AgentRegistry.register("simple")
class SimpleAgent(BaseAgent):
    """Single-turn agent: query → model → response.  No tool calling."""

    agent_id = "simple"

    def __init__(
        self,
        engine: InferenceEngine,
        model: str,
        *,
        bus: Optional[EventBus] = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> None:
        self._engine = engine
        self._model = model
        self._bus = bus
        self._temperature = temperature
        self._max_tokens = max_tokens

    def run(
        self,
        input: str,
        context: Optional[AgentContext] = None,
        **kwargs: Any,
    ) -> AgentResult:
        """Single-turn: build messages, call engine, return result."""
        bus = self._bus

        # Emit turn start
        if bus:
            bus.publish(EventType.AGENT_TURN_START, {
                "agent": self.agent_id,
                "input": input,
            })

        # Build messages from context conversation + user input
        messages: list[Message] = []
        if context and context.conversation.messages:
            messages.extend(context.conversation.messages)
        messages.append(Message(role=Role.USER, content=input))

        # Generate — telemetry is handled by InstrumentedEngine wrapper
        result = self._engine.generate(
            messages,
            model=self._model,
            temperature=self._temperature,
            max_tokens=self._max_tokens,
        )

        content = result.get("content", "")

        # Emit turn end
        if bus:
            bus.publish(EventType.AGENT_TURN_END, {
                "agent": self.agent_id,
                "content_length": len(content),
            })

        return AgentResult(content=content, turns=1)


__all__ = ["SimpleAgent"]
