from guardrails.decision import evaluate_guardrail


def test_validation_issues_reject():
    result = evaluate_guardrail(
        task_type="patient_summary",
        confidence=0.99,
        validation_issues=["missing patient_id"],
    )
    assert result.decision == "REJECT"
    assert "Validation issues present." in result.reasons


def test_missing_confidence_requires_review():
    result = evaluate_guardrail(task_type="patient_summary", confidence=None)
    assert result.decision == "REQUIRES_REVIEW"


def test_low_confidence_rejects_below_hard_floor():
    result = evaluate_guardrail(task_type="patient_summary", confidence=0.40)
    assert result.decision == "REJECT"
    assert "Confidence below hard rejection floor." in result.reasons


def test_high_risk_confidence_below_threshold_requires_review():
    result = evaluate_guardrail(task_type="patient_summary", confidence=0.70)
    assert result.risk_level == "HIGH"
    assert result.threshold == 0.85
    assert result.decision == "REQUIRES_REVIEW"


def test_high_risk_confidence_above_threshold_accepts():
    result = evaluate_guardrail(task_type="patient_summary", confidence=0.90)
    assert result.decision == "ACCEPT"
    assert "Confidence meets task threshold." in result.reasons


def test_critical_risk_confidence_below_threshold_requires_review():
    result = evaluate_guardrail(task_type="medication_or_therapy", confidence=0.90)
    assert result.risk_level == "CRITICAL"
    assert result.threshold == 0.95
    assert result.decision == "REQUIRES_REVIEW"
