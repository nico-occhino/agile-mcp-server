from guardrails.input_guardrail import (
    InputGuardrailDecision,
    InputRiskScore,
    decide_from_scores,
)


def _score(**overrides) -> InputRiskScore:
    values = {
        "domain_relevance": 0.90,
        "intent_clarity": 0.90,
        "injection_risk": 0.10,
        "sensitive_data_risk": 0.10,
        "requested_action": "patient_status",
        "explanation": "Clean clinical lookup.",
    }
    values.update(overrides)
    return InputRiskScore(**values)


def test_high_injection_risk_blocks():
    decision, reasons = decide_from_scores(_score(injection_risk=0.90))

    assert decision == InputGuardrailDecision.BLOCK
    assert reasons == ["High injection risk."]


def test_high_sensitive_data_risk_requires_review():
    decision, reasons = decide_from_scores(_score(sensitive_data_risk=0.95))

    assert decision == InputGuardrailDecision.REQUIRES_REVIEW
    assert reasons == ["High sensitive-data risk."]


def test_low_domain_relevance_clarifies():
    decision, reasons = decide_from_scores(_score(domain_relevance=0.20))

    assert decision == InputGuardrailDecision.CLARIFY
    assert reasons == ["Low domain relevance."]


def test_low_intent_clarity_clarifies():
    decision, reasons = decide_from_scores(_score(intent_clarity=0.40))

    assert decision == InputGuardrailDecision.CLARIFY
    assert reasons == ["Low intent clarity."]


def test_clean_high_scores_allow():
    decision, reasons = decide_from_scores(_score())

    assert decision == InputGuardrailDecision.ALLOW
    assert reasons == ["Input passed dynamic risk policy."]
