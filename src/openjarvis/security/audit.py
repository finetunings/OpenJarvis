"""Audit logger — persist security events to SQLite."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import List, Optional, Union

from openjarvis.core.config import DEFAULT_CONFIG_DIR
from openjarvis.core.events import Event, EventBus, EventType
from openjarvis.security.types import (
    ScanFinding,
    SecurityEvent,
    SecurityEventType,
    ThreatLevel,
)


class AuditLogger:
    """Append-only SQLite audit log for security events.

    Parameters
    ----------
    db_path:
        Path to the SQLite database file.
    bus:
        Optional event bus — if provided, subscribes to security events
        (``SECURITY_SCAN``, ``SECURITY_ALERT``, ``SECURITY_BLOCK``).
    """

    def __init__(
        self,
        db_path: Union[str, Path] = DEFAULT_CONFIG_DIR / "audit.db",
        bus: Optional[EventBus] = None,
    ) -> None:
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._db_path))
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS security_events (
                id          INTEGER PRIMARY KEY,
                timestamp   REAL,
                event_type  TEXT,
                findings_json TEXT,
                content_preview TEXT,
                action_taken TEXT
            )
            """
        )
        self._conn.commit()

        if bus is not None:
            bus.subscribe(EventType.SECURITY_SCAN, self._on_event)
            bus.subscribe(EventType.SECURITY_ALERT, self._on_event)
            bus.subscribe(EventType.SECURITY_BLOCK, self._on_event)

    # -- public API ----------------------------------------------------------

    def log(self, event: SecurityEvent) -> None:
        """Insert a security event into the audit log."""
        findings_json = json.dumps([
            {
                "pattern_name": f.pattern_name,
                "matched_text": f.matched_text,
                "threat_level": f.threat_level.value,
                "start": f.start,
                "end": f.end,
                "description": f.description,
            }
            for f in event.findings
        ])
        self._conn.execute(
            """
            INSERT INTO security_events
                (timestamp, event_type, findings_json, content_preview, action_taken)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                event.timestamp,
                event.event_type.value,
                findings_json,
                event.content_preview,
                event.action_taken,
            ),
        )
        self._conn.commit()

    def query(
        self,
        *,
        event_type: Optional[str] = None,
        since: Optional[float] = None,
        limit: int = 100,
    ) -> List[SecurityEvent]:
        """Query logged security events with optional filters."""
        sql = (
            "SELECT timestamp, event_type, findings_json,"
            " content_preview, action_taken"
            " FROM security_events WHERE 1=1"
        )
        params: list = []

        if event_type is not None:
            sql += " AND event_type = ?"
            params.append(event_type)
        if since is not None:
            sql += " AND timestamp >= ?"
            params.append(since)

        sql += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)

        rows = self._conn.execute(sql, params).fetchall()
        events: List[SecurityEvent] = []
        for row in rows:
            ts, etype, findings_json, preview, action = row
            findings_raw = json.loads(findings_json) if findings_json else []
            findings = [
                ScanFinding(
                    pattern_name=f["pattern_name"],
                    matched_text=f["matched_text"],
                    threat_level=ThreatLevel(f["threat_level"]),
                    start=f["start"],
                    end=f["end"],
                    description=f.get("description", ""),
                )
                for f in findings_raw
            ]
            events.append(
                SecurityEvent(
                    event_type=SecurityEventType(etype),
                    timestamp=ts,
                    findings=findings,
                    content_preview=preview or "",
                    action_taken=action or "",
                )
            )
        return events

    def count(self) -> int:
        """Return the total number of logged security events."""
        row = self._conn.execute(
            "SELECT COUNT(*) FROM security_events"
        ).fetchone()
        return row[0] if row else 0

    def close(self) -> None:
        """Close the SQLite connection."""
        self._conn.close()

    # -- EventBus handler ----------------------------------------------------

    def _on_event(self, event: Event) -> None:
        """Handle an event from the EventBus and log it."""
        data = event.data
        # Map EventType to SecurityEventType
        mapping = {
            EventType.SECURITY_SCAN: SecurityEventType.SECRET_DETECTED,
            EventType.SECURITY_ALERT: SecurityEventType.SECRET_DETECTED,
            EventType.SECURITY_BLOCK: SecurityEventType.SECRET_DETECTED,
        }
        event_type = mapping.get(event.event_type, SecurityEventType.SECRET_DETECTED)

        # Extract findings from event data if present
        findings: List[ScanFinding] = []
        for f in data.get("findings", []):
            findings.append(
                ScanFinding(
                    pattern_name=f.get("pattern", ""),
                    matched_text="",
                    threat_level=ThreatLevel(f.get("threat", "low")),
                    start=0,
                    end=0,
                    description=f.get("description", ""),
                )
            )

        sec_event = SecurityEvent(
            event_type=event_type,
            timestamp=event.timestamp,
            findings=findings,
            content_preview=data.get("content_preview", ""),
            action_taken=data.get("mode", ""),
        )
        self.log(sec_event)


__all__ = ["AuditLogger"]
