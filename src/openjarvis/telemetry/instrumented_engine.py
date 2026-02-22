"""Transparent telemetry wrapper for inference engines."""

from __future__ import annotations

import time
from typing import Any, Dict, List, Sequence

from openjarvis.core.events import EventBus, EventType
from openjarvis.core.types import Message, TelemetryRecord
from openjarvis.engine._stubs import InferenceEngine


class InstrumentedEngine(InferenceEngine):
    """Transparent wrapper that records telemetry around engine calls.

    Agents call ``engine.generate()`` normally -- they don't know
    about telemetry.  The wrapper publishes ``INFERENCE_START``,
    ``INFERENCE_END``, and ``TELEMETRY_RECORD`` events on the bus.
    """

    engine_id = "instrumented"

    def __init__(self, engine: InferenceEngine, bus: EventBus) -> None:
        self._inner = engine
        self._bus = bus

    def generate(
        self,
        messages: Sequence[Message],
        *,
        model: str = "",
        temperature: float = 0.7,
        max_tokens: int = 1024,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Generate with telemetry recording."""
        self._bus.publish(EventType.INFERENCE_START, {
            "model": model, "message_count": len(messages),
        })

        t0 = time.time()
        result = self._inner.generate(
            messages, model=model, temperature=temperature,
            max_tokens=max_tokens, **kwargs,
        )
        latency = time.time() - t0

        usage = result.get("usage", {})
        record = TelemetryRecord(
            timestamp=t0,
            model_id=model,
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
            latency_seconds=latency,
            engine=getattr(self._inner, "engine_id", "unknown"),
        )

        self._bus.publish(EventType.INFERENCE_END, {
            "model": model, "latency": latency, "usage": usage,
        })
        self._bus.publish(EventType.TELEMETRY_RECORD, {"record": record})

        return result

    async def stream(
        self,
        messages: Sequence[Message],
        *,
        model: str = "",
        temperature: float = 0.7,
        max_tokens: int = 1024,
        **kwargs: Any,
    ) -> Any:
        """Stream with deferred telemetry recording."""
        self._bus.publish(EventType.INFERENCE_START, {
            "model": model, "message_count": len(messages),
        })
        t0 = time.time()
        async for token in self._inner.stream(
            messages, model=model, temperature=temperature,
            max_tokens=max_tokens, **kwargs,
        ):
            yield token
        latency = time.time() - t0
        self._bus.publish(EventType.INFERENCE_END, {
            "model": model, "latency": latency,
        })

    def list_models(self) -> List[str]:
        return self._inner.list_models()

    def health(self) -> bool:
        return self._inner.health()


__all__ = ["InstrumentedEngine"]
