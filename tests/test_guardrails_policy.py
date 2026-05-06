from guardrails.policy import RiskLevel, get_risk_level, get_threshold


def test_patient_summary_maps_to_high():
    assert get_risk_level("patient_summary") == RiskLevel.HIGH
    assert get_threshold("patient_summary") == 0.85


def test_medication_or_therapy_maps_to_critical():
    assert get_risk_level("medication_or_therapy") == RiskLevel.CRITICAL
    assert get_threshold("medication_or_therapy") == 0.95


def test_unknown_task_maps_safely_to_critical():
    assert get_risk_level("not_configured") == RiskLevel.CRITICAL
    assert get_threshold("not_configured") == 0.95
