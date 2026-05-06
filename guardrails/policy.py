"""Dynamic risk policy for clinical generation guardrails.

The key idea for the trainee: not every clinical task can tolerate the same
uncertainty. A lightweight administrative summary can pass with a lower
confidence than a medication or therapy suggestion.
"""

from __future__ import annotations

from enum import Enum


class RiskLevel(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


# These thresholds are deliberately visible and simple in Phase 1. Later, Agile
# can tune them with evaluation data without changing feature code.
DEFAULT_THRESHOLDS: dict[RiskLevel, float] = {
    RiskLevel.LOW: 0.60,
    RiskLevel.MEDIUM: 0.75,
    RiskLevel.HIGH: 0.85,
    RiskLevel.CRITICAL: 0.95,
}


TASK_RISK_LEVELS: dict[str, RiskLevel] = {
    "patient_status": RiskLevel.MEDIUM,
    "patient_summary": RiskLevel.HIGH,
    "discharge_draft": RiskLevel.HIGH,
    "medication_or_therapy": RiskLevel.CRITICAL,
    "administrative_summary": RiskLevel.LOW,
    "unknown": RiskLevel.CRITICAL,
}


def get_risk_level(task_type: str) -> RiskLevel:
    """Return the configured risk level, defaulting unknown tasks to CRITICAL."""
    return TASK_RISK_LEVELS.get(task_type, RiskLevel.CRITICAL)


def get_threshold(task_type: str) -> float:
    """Return the confidence threshold for a task's risk level."""
    return DEFAULT_THRESHOLDS[get_risk_level(task_type)]
