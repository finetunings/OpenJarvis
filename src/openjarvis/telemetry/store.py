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
    metadata        TEXT    NOT NULL DEFAULT '{}'
);
"""

_INSERT = """\
INSERT INTO telemetry (
    timestamp, model_id, engine, agent,
    prompt_tokens, completion_tokens, total_tokens,
    latency_seconds, ttft, cost_usd, energy_joules, power_watts, metadata
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""


class TelemetryStore:
    """Append-only SQLite store for inference telemetry records."""

    def __init__(self, db_path: str | Path) -> None:
        self._db_path = str(db_path)
        self._conn = sqlite3.connect(self._db_path, check_same_thread=False)
        self._conn.execute(_CREATE_TABLE)
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
