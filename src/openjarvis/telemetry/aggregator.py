"""Read-only telemetry aggregation — query stored inference records."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass(slots=True)
class ModelStats:
    """Aggregated statistics for a single model."""

    model_id: str = ""
    call_count: int = 0
    total_tokens: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_latency: float = 0.0
    avg_latency: float = 0.0
    total_cost: float = 0.0
    avg_ttft: float = 0.0
    total_energy_joules: float = 0.0
    avg_gpu_utilization_pct: float = 0.0
    avg_throughput_tok_per_sec: float = 0.0


@dataclass(slots=True)
class EngineStats:
    """Aggregated statistics for a single engine backend."""

    engine: str = ""
    call_count: int = 0
    total_tokens: int = 0
    total_latency: float = 0.0
    avg_latency: float = 0.0
    total_cost: float = 0.0
    avg_ttft: float = 0.0
    total_energy_joules: float = 0.0
    avg_gpu_utilization_pct: float = 0.0
    avg_throughput_tok_per_sec: float = 0.0


@dataclass(slots=True)
class AggregatedStats:
    """Top-level summary combining per-model and per-engine stats."""

    total_calls: int = 0
    total_tokens: int = 0
    total_cost: float = 0.0
    total_latency: float = 0.0
    per_model: List[ModelStats] = field(default_factory=list)
    per_engine: List[EngineStats] = field(default_factory=list)


class TelemetryAggregator:
    """Read-only query layer over the telemetry SQLite database."""

    def __init__(self, db_path: str | Path) -> None:
        self._db_path = str(db_path)
        self._conn = sqlite3.connect(self._db_path)
        self._conn.row_factory = sqlite3.Row

    @staticmethod
    def _time_filter(
        since: Optional[float] = None,
        until: Optional[float] = None,
    ) -> tuple[str, list[Any]]:
        """Build a WHERE clause fragment for time-range filtering."""
        clauses: list[str] = []
        params: list[Any] = []
        if since is not None:
            clauses.append("timestamp >= ?")
            params.append(since)
        if until is not None:
            clauses.append("timestamp <= ?")
            params.append(until)
        if clauses:
            return " WHERE " + " AND ".join(clauses), params
        return "", params

    def per_model_stats(
        self,
        *,
        since: Optional[float] = None,
        until: Optional[float] = None,
    ) -> List[ModelStats]:
        where, params = self._time_filter(since, until)
        sql = (
            "SELECT model_id,"
            " COUNT(*) AS call_count,"
            " SUM(total_tokens) AS total_tokens,"
            " SUM(prompt_tokens) AS prompt_tokens,"
            " SUM(completion_tokens) AS completion_tokens,"
            " SUM(latency_seconds) AS total_latency,"
            " AVG(latency_seconds) AS avg_latency,"
            " SUM(cost_usd) AS total_cost,"
            " AVG(ttft) AS avg_ttft,"
            " SUM(energy_joules) AS total_energy_joules,"
            " AVG(gpu_utilization_pct) AS avg_gpu_utilization_pct,"
            " AVG(throughput_tok_per_sec) AS avg_throughput_tok_per_sec"
            f" FROM telemetry{where}"
            " GROUP BY model_id ORDER BY call_count DESC"
        )
        rows = self._conn.execute(sql, params).fetchall()
        return [
            ModelStats(
                model_id=r["model_id"],
                call_count=r["call_count"],
                total_tokens=r["total_tokens"] or 0,
                prompt_tokens=r["prompt_tokens"] or 0,
                completion_tokens=r["completion_tokens"] or 0,
                total_latency=r["total_latency"] or 0.0,
                avg_latency=r["avg_latency"] or 0.0,
                total_cost=r["total_cost"] or 0.0,
                avg_ttft=r["avg_ttft"] or 0.0,
                total_energy_joules=r["total_energy_joules"] or 0.0,
                avg_gpu_utilization_pct=r["avg_gpu_utilization_pct"] or 0.0,
                avg_throughput_tok_per_sec=r["avg_throughput_tok_per_sec"] or 0.0,
            )
            for r in rows
        ]

    def per_engine_stats(
        self,
        *,
        since: Optional[float] = None,
        until: Optional[float] = None,
    ) -> List[EngineStats]:
        where, params = self._time_filter(since, until)
        sql = (
            "SELECT engine,"
            " COUNT(*) AS call_count,"
            " SUM(total_tokens) AS total_tokens,"
            " SUM(latency_seconds) AS total_latency,"
            " AVG(latency_seconds) AS avg_latency,"
            " SUM(cost_usd) AS total_cost,"
            " AVG(ttft) AS avg_ttft,"
            " SUM(energy_joules) AS total_energy_joules,"
            " AVG(gpu_utilization_pct) AS avg_gpu_utilization_pct,"
            " AVG(throughput_tok_per_sec) AS avg_throughput_tok_per_sec"
            f" FROM telemetry{where}"
            " GROUP BY engine ORDER BY call_count DESC"
        )
        rows = self._conn.execute(sql, params).fetchall()
        return [
            EngineStats(
                engine=r["engine"],
                call_count=r["call_count"],
                total_tokens=r["total_tokens"] or 0,
                total_latency=r["total_latency"] or 0.0,
                avg_latency=r["avg_latency"] or 0.0,
                total_cost=r["total_cost"] or 0.0,
                avg_ttft=r["avg_ttft"] or 0.0,
                total_energy_joules=r["total_energy_joules"] or 0.0,
                avg_gpu_utilization_pct=r["avg_gpu_utilization_pct"] or 0.0,
                avg_throughput_tok_per_sec=r["avg_throughput_tok_per_sec"] or 0.0,
            )
            for r in rows
        ]

    def top_models(
        self,
        n: int = 5,
        *,
        since: Optional[float] = None,
    ) -> List[ModelStats]:
        stats = self.per_model_stats(since=since)
        return stats[:n]

    def summary(
        self,
        *,
        since: Optional[float] = None,
        until: Optional[float] = None,
    ) -> AggregatedStats:
        model_stats = self.per_model_stats(since=since, until=until)
        engine_stats = self.per_engine_stats(since=since, until=until)
        return AggregatedStats(
            total_calls=sum(m.call_count for m in model_stats),
            total_tokens=sum(m.total_tokens for m in model_stats),
            total_cost=sum(m.total_cost for m in model_stats),
            total_latency=sum(m.total_latency for m in model_stats),
            per_model=model_stats,
            per_engine=engine_stats,
        )

    def export_records(
        self,
        *,
        since: Optional[float] = None,
        until: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        where, params = self._time_filter(since, until)
        sql = f"SELECT * FROM telemetry{where} ORDER BY timestamp"
        rows = self._conn.execute(sql, params).fetchall()
        return [dict(r) for r in rows]

    def record_count(self) -> int:
        row = self._conn.execute("SELECT COUNT(*) FROM telemetry").fetchone()
        return row[0] if row else 0

    def clear(self) -> int:
        count = self.record_count()
        self._conn.execute("DELETE FROM telemetry")
        self._conn.commit()
        return count

    def close(self) -> None:
        self._conn.close()


__all__ = [
    "AggregatedStats",
    "EngineStats",
    "ModelStats",
    "TelemetryAggregator",
]
