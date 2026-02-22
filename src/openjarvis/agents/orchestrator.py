"""OrchestratorAgent — multi-turn agent with tool-calling loop."""

from __future__ import annotations

from typing import Any, List, Optional

from openjarvis.agents._stubs import AgentContext, AgentResult, BaseAgent
from openjarvis.core.events import EventBus, EventType
from openjarvis.core.registry import AgentRegistry
from openjarvis.core.types import Message, Role, ToolCall, ToolResult
from openjarvis.engine._stubs import InferenceEngine
from openjarvis.tools._stubs import BaseTool, ToolExecutor


@AgentRegistry.register("orchestrator")
class OrchestratorAgent(BaseAgent):
    """Multi-turn agent that routes between tools and the LLM.

    Implements a tool-calling loop:
    1. Send messages with tool definitions to the engine.
    2. If the response contains tool_calls, execute them and loop.
    3. If no tool_calls, return the final answer.
    4. Stop after ``max_turns`` iterations.
    """

    agent_id = "orchestrator"

    def __init__(
        self,
        engine: InferenceEngine,
        model: str,
        *,
        tools: Optional[List[BaseTool]] = None,
        bus: Optional[EventBus] = None,
        max_turns: int = 10,
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> None:
        self._engine = engine
        self._model = model
        self._tools = tools or []
        self._executor = ToolExecutor(self._tools, bus=bus)
        self._bus = bus
        self._max_turns = max_turns
        self._temperature = temperature
        self._max_tokens = max_tokens

    def run(
        self,
        input: str,
        context: Optional[AgentContext] = None,
        **kwargs: Any,
    ) -> AgentResult:
        bus = self._bus

        # Emit agent start
        if bus:
            bus.publish(EventType.AGENT_TURN_START, {
                "agent": self.agent_id,
                "input": input,
            })

        # Build initial messages
        messages: list[Message] = []
        if context and context.conversation.messages:
            messages.extend(context.conversation.messages)
        messages.append(Message(role=Role.USER, content=input))

        # Get OpenAI-format tool definitions
        openai_tools = self._executor.get_openai_tools() if self._tools else []

        all_tool_results: list[ToolResult] = []
        turns = 0

        for _turn in range(self._max_turns):
            turns += 1

            # Build generate kwargs
            gen_kwargs: dict[str, Any] = {}
            if openai_tools:
                gen_kwargs["tools"] = openai_tools

            result = self._engine.generate(
                messages,
                model=self._model,
                temperature=self._temperature,
                max_tokens=self._max_tokens,
                **gen_kwargs,
            )

            content = result.get("content", "")
            raw_tool_calls = result.get("tool_calls", [])

            # No tool calls → final answer
            if not raw_tool_calls:
                if bus:
                    bus.publish(EventType.AGENT_TURN_END, {
                        "agent": self.agent_id,
                        "turns": turns,
                        "content_length": len(content),
                    })
                return AgentResult(
                    content=content,
                    tool_results=all_tool_results,
                    turns=turns,
                )

            # Build ToolCall objects from raw dicts
            tool_calls = [
                ToolCall(
                    id=tc.get("id", f"call_{i}"),
                    name=tc.get("name", ""),
                    arguments=tc.get("arguments", "{}"),
                )
                for i, tc in enumerate(raw_tool_calls)
            ]

            # Append assistant message with tool calls
            messages.append(Message(
                role=Role.ASSISTANT,
                content=content,
                tool_calls=tool_calls,
            ))

            # Execute each tool and append results
            for tc in tool_calls:
                tool_result = self._executor.execute(tc)
                all_tool_results.append(tool_result)

                # Append tool response message
                # Serialize arguments for the content
                messages.append(Message(
                    role=Role.TOOL,
                    content=tool_result.content,
                    tool_call_id=tc.id,
                    name=tc.name,
                ))

        # Max turns exceeded
        if bus:
            bus.publish(EventType.AGENT_TURN_END, {
                "agent": self.agent_id,
                "turns": turns,
                "max_turns_exceeded": True,
            })

        # Try to provide last content or a warning
        final_content = content if content else (
            "Maximum turns reached without a final answer."
        )
        return AgentResult(
            content=final_content,
            tool_results=all_tool_results,
            turns=turns,
            metadata={"max_turns_exceeded": True},
        )


__all__ = ["OrchestratorAgent"]
