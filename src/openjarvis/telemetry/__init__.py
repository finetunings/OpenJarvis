"""Telemetry — SQLite-backed inference recording and instrumented wrappers."""

from __future__ import annotations

from openjarvis.telemetry.aggregator import (
    AggregatedStats,
    EngineStats,
    ModelStats,
    TelemetryAggregator,
)
from openjarvis.telemetry.store import TelemetryStore
from openjarvis.telemetry.wrapper import instrumented_generate

try:
    from openjarvis.telemetry.gpu_monitor import GpuHardwareSpec, GpuMonitor, GpuSample, GpuSnapshot
except ImportError:
    pass

try:
    from openjarvis.telemetry.efficiency import EfficiencyMetrics, compute_efficiency
except ImportError:
    pass

try:
    from openjarvis.telemetry.vllm_metrics import VLLMMetrics, VLLMMetricsScraper
except ImportError:
    pass

__all__ = [
    "AggregatedStats",
    "EfficiencyMetrics",
    "EngineStats",
    "GpuHardwareSpec",
    "GpuMonitor",
    "GpuSample",
    "GpuSnapshot",
    "ModelStats",
    "TelemetryAggregator",
    "TelemetryStore",
    "VLLMMetrics",
    "VLLMMetricsScraper",
    "compute_efficiency",
    "instrumented_generate",
]
