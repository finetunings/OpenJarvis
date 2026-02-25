"""SQLite-backed telemetry storage."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from openjarvis.core.events import Event, EventBus, EventType
from openjarvis.core.types import TelemetryRecord

_CREATE_TABLE = """\
CREATE TABLE IF NOT EXISTS telemetry (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp       REAL    NOT NULL,
    model_id        TEXT    NOT NULL,
    engine          TEXT    NOT NULL DEFAULT '',
    agent           TEXT    NOT NULL DEFAULT '',
    prompt_tokens   INTEGER NOT NULL DEFAULT 0,
    completion_tokens INTEGER NOT NULL DEFAULT 0,
    total_tokens    INTEGER NOT NULL DEFAULT 0,
    latency_seconds REAL    NOT NULL DEFAULT 0.0,
    ttft            REAL    NOT NULL DEFAULT 0.0,
    cost_usd        REAL    NOT NULL DEFAULT 0.0,
    energy_joules   REAL    NOT NULL DEFAULT 0.0,
    power_watts     REAL    NOT NULL DEFAULT 0.0,
    gpu_utilization_pct  REAL NOT NULL DEFAULT 0.0,
    gpu_memory_used_gb   REAL NOT NULL DEFAULT 0.0,
    gpu_temperature_c    REAL NOT NULL DEFAULT 0.0,
    throughput_tok_per_sec REAL NOT NULL DEFAULT 0.0,
    prefill_latency_seconds REAL NOT NULL DEFAULT 0.0,
    decode_latency_seconds  REAL NOT NULL DEFAULT 0.0,
    metadata        TEXT    NOT NULL DEFAULT '{}'
);
"""

_INSERT = """\
INSERT INTO telemetry (
    timestamp, model_id, engine, agent,
    prompt_tokens, completion_tokens, total_tokens,
    latency_seconds, ttft, cost_usd, energy_joules, power_watts,
    gpu_utilization_pct, gpu_memory_used_gb, gpu_temperature_c,
    throughput_tok_per_sec, prefill_latency_seconds, decode_latency_seconds,
    metadata
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""

_MIGRATE_COLUMNS = [
    ("gpu_utilization_pct", "REAL NOT NULL DEFAULT 0.0"),
    ("gpu_memory_used_gb", "REAL NOT NULL DEFAULT 0.0"),
    ("gpu_temperature_c", "REAL NOT NULL DEFAULT 0.0"),
    ("throughput_tok_per_sec", "REAL NOT NULL DEFAULT 0.0"),
    ("prefill_latency_seconds", "REAL NOT NULL DEFAULT 0.0"),
    ("decode_latency_seconds", "REAL NOT NULL DEFAULT 0.0"),
]


class TelemetryStore:
    """Append-only SQLite store for inference telemetry records."""

    def __init__(self, db_path: str | Path) -> None:
        self._db_path = str(db_path)
        self._conn = sqlite3.connect(self._db_path, check_same_thread=False)
        self._conn.execute(_CREATE_TABLE)
        self._conn.commit()
        self._migrate_schema()

    def _migrate_schema(self) -> None:
        """Add new columns to existing databases (idempotent)."""
        for col_name, col_def in _MIGRATE_COLUMNS:
            try:
                self._conn.execute(
                    f"ALTER TABLE telemetry ADD COLUMN {col_name} {col_def}",
                )
            except sqlite3.OperationalError:
                pass  # Column already exists
        self._conn.commit()

    def record(self, rec: TelemetryRecord) -> None:
        """Persist a single telemetry record."""
        self._conn.execute(
            _INSERT,
            (
                rec.timestamp,
                rec.model_id,
                rec.engine,
                rec.agent,
                rec.prompt_tokens,
                rec.completion_tokens,
                rec.total_tokens,
                rec.latency_seconds,
                rec.ttft,
                rec.cost_usd,
                rec.energy_joules,
                rec.power_watts,
                rec.gpu_utilization_pct,
                rec.gpu_memory_used_gb,
                rec.gpu_temperature_c,
                rec.throughput_tok_per_sec,
                rec.prefill_latency_seconds,
                rec.decode_latency_seconds,
                json.dumps(rec.metadata),
            ),
        )
        self._conn.commit()

    def subscribe_to_bus(self, bus: EventBus) -> None:
        """Subscribe to ``TELEMETRY_RECORD`` events on *bus*."""
        bus.subscribe(EventType.TELEMETRY_RECORD, self._on_event)

    def _on_event(self, event: Event) -> None:
        rec = event.data.get("record")
        if isinstance(rec, TelemetryRecord):
            self.record(rec)

    def close(self) -> None:
        """Close the underlying SQLite connection."""
        self._conn.close()

    # -- helpers for querying (used by tests) --------------------------------

    def _fetchall(self, sql: str = "SELECT * FROM telemetry") -> list:
        return self._conn.execute(sql).fetchall()


__all__ = ["TelemetryStore"]
