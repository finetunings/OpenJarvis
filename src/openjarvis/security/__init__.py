"""Security guardrails — scanners, engine wrapper, and audit logging."""

from openjarvis.security._stubs import BaseScanner
from openjarvis.security.audit import AuditLogger
from openjarvis.security.file_policy import (
    DEFAULT_SENSITIVE_PATTERNS,
    filter_sensitive_paths,
    is_sensitive_file,
)
from openjarvis.security.guardrails import GuardrailsEngine, SecurityBlockError
from openjarvis.security.scanner import PIIScanner, SecretScanner
from openjarvis.security.types import (
    RedactionMode,
    ScanFinding,
    ScanResult,
    SecurityEvent,
    SecurityEventType,
    ThreatLevel,
)

__all__ = [
    "AuditLogger",
    "BaseScanner",
    "DEFAULT_SENSITIVE_PATTERNS",
    "GuardrailsEngine",
    "PIIScanner",
    "RedactionMode",
    "ScanFinding",
    "ScanResult",
    "SecretScanner",
    "SecurityBlockError",
    "SecurityEvent",
    "SecurityEventType",
    "ThreatLevel",
    "filter_sensitive_paths",
    "is_sensitive_file",
]
