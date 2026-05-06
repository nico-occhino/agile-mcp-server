from rag.context_builder import build_rag_context


def test_build_rag_context_returns_chunks_and_text():
    context = build_rag_context("discharge summary allergy follow-up", top_k=3)

    assert context["query"] == "discharge summary allergy follow-up"
    assert context["chunks"]
    assert context["context_text"]
    assert "Source:" in context["context_text"]
    assert "Chunk:" in context["context_text"]
    assert {"chunk_id", "source", "score"}.issubset(context["chunks"][0])


def test_discharge_draft_includes_rag_context_without_real_llm(monkeypatch):
    from features import patient_summary

    class FakeUncertainResult:
        confidence = 1.0

        def to_dict(self):
            return {
                "result": "Bozza di dimissione simulata.",
                "confidence": 1.0,
                "confidence_level": "HIGH",
                "rationale": "test double",
            }

    monkeypatch.setattr(
        patient_summary,
        "call_llm_structured",
        lambda **kwargs: patient_summary.ClinicalFlags(
            primary_diagnosis_description="diagnosi test",
            active_conditions=[],
            key_medications=[],
            known_allergies=["Arachidi"],
            follow_up_required=False,
        ),
    )
    monkeypatch.setattr(
        patient_summary,
        "call_llm_n_times",
        lambda **kwargs: ["Bozza di dimissione simulata."],
    )
    monkeypatch.setattr(
        patient_summary,
        "build_uncertain_result",
        lambda samples, mode: FakeUncertainResult(),
    )

    result = patient_summary.get_patient_discharge_draft("45")

    assert result["found"] is True
    assert "rag_context" in result
    assert result["rag_context"]["query"]
    assert result["rag_context"]["sources"]
    assert {"chunk_id", "source", "score", "metadata"}.issubset(
        result["rag_context"]["sources"][0]
    )


def test_discharge_draft_returns_controlled_error_on_structured_failure(monkeypatch):
    from features import patient_summary

    def raise_structured_failure(**kwargs):
        raise ValueError("mock structured validation failure")

    monkeypatch.setattr(
        patient_summary,
        "call_llm_structured",
        raise_structured_failure,
    )

    result = patient_summary.get_patient_discharge_draft("45")

    assert result["found"] is True
    assert result["patient_id"] == "45"
    assert result["patient_name"] == "Mario Rossi"
    assert "Structured clinical flag extraction failed" in result["error"]
    assert "mock structured validation failure" in result["details"]
    assert result["rag_context"]["sources"]
