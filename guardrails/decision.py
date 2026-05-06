"""Decision logic for accepting, reviewing, or rejecting LLM outputs."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from guardrails.policy import get_risk_level, get_threshold


class GuardrailDecision(str, Enum):
    ACCEPT = "ACCEPT"
    REQUIRES_REVIEW = "REQUIRES_REVIEW"
    REJECT = "REJECT"


class GuardrailResult(BaseModel):
    task_type: str
    risk_level: str
    confidence: float | None
    threshold: float
    decision: str
    reasons: list[str]
    validation_issues: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


def evaluate_guardrail(
    task_type: str,
    confidence: float | None,
    validation_issues: list[str] | None = None,
    metadata: dict | None = None,
) -> GuardrailResult:
    """Evaluate a generated clinical output against the risk policy.

    This function does not decide whether the clinical content is true. It only
    combines structural validation status, an uncertainty confidence score, and
    a task-specific threshold into an auditable decision.
    """
    issues = validation_issues or []
    risk_level = get_risk_level(task_type)
    threshold = get_threshold(task_type)

    if issues:
        decision = GuardrailDecision.REJECT
        reasons = ["Validation issues present."]
    elif confidence is None:
        decision = GuardrailDecision.REQUIRES_REVIEW
        reasons = ["Confidence unavailable."]
    elif confidence < 0.50:
        decision = GuardrailDecision.REJECT
        reasons = ["Confidence below hard rejection floor."]
    elif confidence < threshold:
        decision = GuardrailDecision.REQUIRES_REVIEW
        reasons = ["Confidence below task threshold."]
    else:
        decision = GuardrailDecision.ACCEPT
        reasons = ["Confidence meets task threshold."]

    return GuardrailResult(
        task_type=task_type,
        risk_level=risk_level.value,
        confidence=confidence,
        threshold=threshold,
        decision=decision.value,
        reasons=reasons,
        validation_issues=issues,
        metadata=metadata or {},
    )
