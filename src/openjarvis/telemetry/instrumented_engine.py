"""Transparent telemetry wrapper for inference engines."""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Sequence

from openjarvis.core.events import EventBus, EventType
from openjarvis.core.types import Message, TelemetryRecord
from openjarvis.engine._stubs import InferenceEngine
from openjarvis.telemetry.gpu_monitor import GpuMonitor, GpuSample


class InstrumentedEngine(InferenceEngine):
    """Transparent wrapper that records telemetry around engine calls.

    Agents call ``engine.generate()`` normally -- they don't know
    about telemetry.  The wrapper publishes ``INFERENCE_START``,
    ``INFERENCE_END``, and ``TELEMETRY_RECORD`` events on the bus.
    """

    engine_id = "instrumented"

    def __init__(
        self,
        engine: InferenceEngine,
        bus: EventBus,
        gpu_monitor: Optional[Any] = None,
    ) -> None:
        self._inner = engine
        self._bus = bus
        self._gpu_monitor = gpu_monitor

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

        gpu_sample: Optional[GpuSample] = None
        t0 = time.time()

        if self._gpu_monitor is not None:
            with self._gpu_monitor.sample() as gpu_sample:
                result = self._inner.generate(
                    messages, model=model, temperature=temperature,
                    max_tokens=max_tokens, **kwargs,
                )
        else:
            result = self._inner.generate(
                messages, model=model, temperature=temperature,
                max_tokens=max_tokens, **kwargs,
            )

        latency = time.time() - t0

        usage = result.get("usage", {})
        completion_tokens = usage.get("completion_tokens", 0)
        ttft = result.get("ttft", 0.0)
        throughput = completion_tokens / latency if latency > 0 else 0.0

        # GPU metrics from sample
        energy_joules = 0.0
        power_watts = 0.0
        gpu_utilization_pct = 0.0
        gpu_memory_used_gb = 0.0
        gpu_temperature_c = 0.0
        prefill_latency = 0.0

        if gpu_sample is not None:
            energy_joules = gpu_sample.energy_joules
            power_watts = gpu_sample.mean_power_watts
            gpu_utilization_pct = gpu_sample.mean_utilization_pct
            gpu_memory_used_gb = gpu_sample.peak_memory_used_gb
            gpu_temperature_c = gpu_sample.mean_temperature_c

        if ttft > 0:
            prefill_latency = ttft

        record = TelemetryRecord(
            timestamp=t0,
            model_id=model,
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=completion_tokens,
            latency_seconds=latency,
            ttft=ttft,
            throughput_tok_per_sec=throughput,
            energy_joules=energy_joules,
            power_watts=power_watts,
            gpu_utilization_pct=gpu_utilization_pct,
            gpu_memory_used_gb=gpu_memory_used_gb,
            gpu_temperature_c=gpu_temperature_c,
            prefill_latency_seconds=prefill_latency,
            engine=getattr(self._inner, "engine_id", "unknown"),
        )

        event_data = {
            "model": model,
            "latency": latency,
            "usage": usage,
            "ttft": ttft,
            "throughput_tok_per_sec": throughput,
            "energy_joules": energy_joules,
            "power_watts": power_watts,
            "gpu_utilization_pct": gpu_utilization_pct,
            "gpu_memory_used_gb": gpu_memory_used_gb,
            "gpu_temperature_c": gpu_temperature_c,
            "prefill_latency_seconds": prefill_latency,
        }

        self._bus.publish(EventType.INFERENCE_END, event_data)
        self._bus.publish(EventType.TELEMETRY_RECORD, {"record": record})

        # Inject telemetry dict into result for downstream consumers (eval backend)
        result["_telemetry"] = {
            "latency": latency,
            "ttft": ttft,
            "throughput_tok_per_sec": throughput,
            "energy_joules": energy_joules,
            "power_watts": power_watts,
            "gpu_utilization_pct": gpu_utilization_pct,
            "gpu_memory_used_gb": gpu_memory_used_gb,
            "gpu_temperature_c": gpu_temperature_c,
            "prefill_latency_seconds": prefill_latency,
        }

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

    def close(self) -> None:
        self._inner.close()


__all__ = ["InstrumentedEngine"]
