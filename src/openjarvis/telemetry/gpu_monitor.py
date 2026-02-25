"""GPU monitoring via pynvml — background poller for GPU metrics."""

from __future__ import annotations

import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Dict, Generator, List, Optional

try:
    import pynvml

    _PYNVML_AVAILABLE = True
except ImportError:
    _PYNVML_AVAILABLE = False


# ---------------------------------------------------------------------------
# Hardware spec database
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GpuHardwareSpec:
    """Peak theoretical capabilities for a known GPU model."""

    tflops_fp16: float
    bandwidth_gb_s: float
    tdp_watts: float


GPU_SPECS: Dict[str, GpuHardwareSpec] = {
    "A100-SXM": GpuHardwareSpec(tflops_fp16=312, bandwidth_gb_s=2039, tdp_watts=400),
    "A100-PCIE": GpuHardwareSpec(tflops_fp16=312, bandwidth_gb_s=2039, tdp_watts=300),
    "H100-SXM": GpuHardwareSpec(tflops_fp16=990, bandwidth_gb_s=3350, tdp_watts=700),
    "H100-PCIE": GpuHardwareSpec(tflops_fp16=756, bandwidth_gb_s=2000, tdp_watts=350),
    "L40S": GpuHardwareSpec(tflops_fp16=366, bandwidth_gb_s=864, tdp_watts=350),
    "A10": GpuHardwareSpec(tflops_fp16=125, bandwidth_gb_s=600, tdp_watts=150),
    "RTX 4090": GpuHardwareSpec(tflops_fp16=165, bandwidth_gb_s=1008, tdp_watts=450),
    "RTX 3090": GpuHardwareSpec(tflops_fp16=71, bandwidth_gb_s=936, tdp_watts=350),
}


def lookup_gpu_spec(name: str) -> Optional[GpuHardwareSpec]:
    """Return the :class:`GpuHardwareSpec` for *name*, or ``None`` if unknown.

    Matches are case-insensitive substring lookups against the keys in
    :data:`GPU_SPECS`.
    """
    upper = name.upper()
    for key, spec in GPU_SPECS.items():
        if key.upper() in upper:
            return spec
    return None


# ---------------------------------------------------------------------------
# Snapshot & aggregated sample
# ---------------------------------------------------------------------------


@dataclass
class GpuSnapshot:
    """A single point-in-time reading from one GPU device."""

    power_watts: float
    utilization_pct: float
    memory_used_gb: float
    temperature_c: float
    device_id: int = 0


@dataclass
class GpuSample:
    """Aggregated GPU metrics over an inference bracket."""

    energy_joules: float = 0.0
    mean_power_watts: float = 0.0
    peak_power_watts: float = 0.0
    mean_utilization_pct: float = 0.0
    peak_utilization_pct: float = 0.0
    mean_memory_used_gb: float = 0.0
    peak_memory_used_gb: float = 0.0
    mean_temperature_c: float = 0.0
    peak_temperature_c: float = 0.0
    duration_seconds: float = 0.0
    num_snapshots: int = 0


# ---------------------------------------------------------------------------
# Monitor
# ---------------------------------------------------------------------------


class GpuMonitor:
    """Background GPU poller using pynvml.

    Usage::

        mon = GpuMonitor(poll_interval_ms=50)
        with mon.sample() as result:
            # ... run inference ...
            pass
        print(result.energy_joules)
        mon.close()
    """

    def __init__(self, poll_interval_ms: int = 50) -> None:
        self._poll_interval_s = poll_interval_ms / 1000.0
        self._handles: List = []
        self._device_count = 0
        self._initialized = False

        if _PYNVML_AVAILABLE:
            try:
                pynvml.nvmlInit()
                self._device_count = pynvml.nvmlDeviceGetCount()
                self._handles = [
                    pynvml.nvmlDeviceGetHandleByIndex(i)
                    for i in range(self._device_count)
                ]
                self._initialized = True
            except Exception:
                self._initialized = False

    @staticmethod
    def available() -> bool:
        """Return ``True`` if pynvml is importable and can be initialized."""
        if not _PYNVML_AVAILABLE:
            return False
        try:
            pynvml.nvmlInit()
            pynvml.nvmlShutdown()
            return True
        except Exception:
            return False

    # -- polling thread internals ---------------------------------------------

    def _poll_once(self) -> List[GpuSnapshot]:
        """Read current metrics from all GPU devices."""
        snapshots: List[GpuSnapshot] = []
        for idx, handle in enumerate(self._handles):
            try:
                power_mw = pynvml.nvmlDeviceGetPowerUsage(handle)
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                temp = pynvml.nvmlDeviceGetTemperature(
                    handle, pynvml.NVML_TEMPERATURE_GPU
                )
                snapshots.append(
                    GpuSnapshot(
                        power_watts=power_mw / 1000.0,
                        utilization_pct=float(util.gpu),
                        memory_used_gb=mem_info.used / (1024**3),
                        temperature_c=float(temp),
                        device_id=idx,
                    )
                )
            except Exception:
                pass
        return snapshots

    def _polling_loop(
        self,
        snapshots_out: List[List[GpuSnapshot]],
        timestamps_out: List[float],
        lock: threading.Lock,
        stop_event: threading.Event,
    ) -> None:
        """Background thread: poll GPUs until *stop_event* is set."""
        while not stop_event.is_set():
            reading = self._poll_once()
            if reading:
                now = time.monotonic()
                with lock:
                    snapshots_out.append(reading)
                    timestamps_out.append(now)
            stop_event.wait(self._poll_interval_s)

    # -- aggregation -----------------------------------------------------------

    @staticmethod
    def _aggregate(
        all_snapshots: List[List[GpuSnapshot]],
        timestamps: List[float],
        wall_duration: float,
    ) -> GpuSample:
        """Build a :class:`GpuSample` from collected snapshots.

        Energy is computed via trapezoidal integration of total power
        (summed across all devices) over the timestamp series.
        """
        if not all_snapshots:
            return GpuSample(duration_seconds=wall_duration)

        # Flatten per-tick aggregates (sum power across devices per tick)
        tick_powers: List[float] = []
        tick_utils: List[float] = []
        tick_mems: List[float] = []
        tick_temps: List[float] = []

        for tick_snaps in all_snapshots:
            total_power = sum(s.power_watts for s in tick_snaps)
            mean_util = (
                sum(s.utilization_pct for s in tick_snaps) / len(tick_snaps)
            )
            total_mem = sum(s.memory_used_gb for s in tick_snaps)
            mean_temp = (
                sum(s.temperature_c for s in tick_snaps) / len(tick_snaps)
            )
            tick_powers.append(total_power)
            tick_utils.append(mean_util)
            tick_mems.append(total_mem)
            tick_temps.append(mean_temp)

        n = len(tick_powers)

        # Trapezoidal integration for energy
        energy = 0.0
        for i in range(1, len(timestamps)):
            dt = timestamps[i] - timestamps[i - 1]
            energy += 0.5 * (tick_powers[i - 1] + tick_powers[i]) * dt

        return GpuSample(
            energy_joules=energy,
            mean_power_watts=sum(tick_powers) / n,
            peak_power_watts=max(tick_powers),
            mean_utilization_pct=sum(tick_utils) / n,
            peak_utilization_pct=max(tick_utils),
            mean_memory_used_gb=sum(tick_mems) / n,
            peak_memory_used_gb=max(tick_mems),
            mean_temperature_c=sum(tick_temps) / n,
            peak_temperature_c=max(tick_temps),
            duration_seconds=wall_duration,
            num_snapshots=n,
        )

    # -- public API -----------------------------------------------------------

    @contextmanager
    def sample(self) -> Generator[GpuSample, None, None]:
        """Context manager that polls GPUs during the block, then populates the sample.

        If pynvml is unavailable or no devices are found, yields an empty
        :class:`GpuSample` without starting a background thread.
        """
        result = GpuSample()

        if not self._initialized or self._device_count == 0:
            t_start = time.monotonic()
            yield result
            result.duration_seconds = time.monotonic() - t_start
            return

        snapshots: List[List[GpuSnapshot]] = []
        timestamps: List[float] = []
        lock = threading.Lock()
        stop_event = threading.Event()

        thread = threading.Thread(
            target=self._polling_loop,
            args=(snapshots, timestamps, lock, stop_event),
            daemon=True,
        )

        t_start = time.monotonic()
        thread.start()
        try:
            yield result
        finally:
            stop_event.set()
            thread.join(timeout=2.0)
            wall = time.monotonic() - t_start

            with lock:
                snap_copy = list(snapshots)
                ts_copy = list(timestamps)

            aggregated = self._aggregate(snap_copy, ts_copy, wall)

            # Copy aggregated values into the yielded result object
            result.energy_joules = aggregated.energy_joules
            result.mean_power_watts = aggregated.mean_power_watts
            result.peak_power_watts = aggregated.peak_power_watts
            result.mean_utilization_pct = aggregated.mean_utilization_pct
            result.peak_utilization_pct = aggregated.peak_utilization_pct
            result.mean_memory_used_gb = aggregated.mean_memory_used_gb
            result.peak_memory_used_gb = aggregated.peak_memory_used_gb
            result.mean_temperature_c = aggregated.mean_temperature_c
            result.peak_temperature_c = aggregated.peak_temperature_c
            result.duration_seconds = aggregated.duration_seconds
            result.num_snapshots = aggregated.num_snapshots

    def close(self) -> None:
        """Shut down pynvml if it was initialized."""
        if self._initialized:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass
            self._initialized = False


__all__ = [
    "GpuHardwareSpec",
    "GpuSnapshot",
    "GpuSample",
    "GpuMonitor",
    "GPU_SPECS",
    "lookup_gpu_spec",
]
