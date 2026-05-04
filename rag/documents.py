"""In-code Phase 1 domain corpus for lightweight RAG demos and tests."""

from __future__ import annotations

from retrieval.base import DocumentChunk


PHASE1_DOMAIN_CORPUS: list[DocumentChunk] = [
    DocumentChunk(
        chunk_id="current_admission_rule",
        source="docs/domain_glossary.md",
        text=(
            "Nel modello dati di ricovero, dateEnd uguale a None indica che "
            "il paziente e attualmente ricoverato. Se dateEnd contiene una "
            "data, il ricovero e concluso e il paziente risulta dimesso. "
            "English keywords: current admission, admitted, discharged."
        ),
        metadata={
            "doc_type": "glossary",
            "language": "it",
            "domain": "admissions",
        },
    ),
    DocumentChunk(
        chunk_id="allergy_safety_rule",
        source="docs/safety_rules.md",
        text=(
            "Le allergie note sono informazioni safety-critical. Se presenti "
            "nei dati paziente, devono sempre essere evidenziate nei riassunti "
            "clinici e nelle bozze di dimissione. English keywords: allergy "
            "safety discharge."
        ),
        metadata={
            "doc_type": "safety_rule",
            "language": "it",
            "safety_critical": "true",
        },
    ),
    DocumentChunk(
        chunk_id="discharge_summary_template",
        source="docs/discharge_template.md",
        text=(
            "Una bozza di dimissione deve includere identita del paziente, "
            "motivo del ricovero, diagnosi principale, decorso clinico, "
            "allergie, terapie e follow-up quando presenti nei dati. English "
            "keywords: discharge summary template allergy medications follow-up."
        ),
        metadata={
            "doc_type": "template",
            "language": "it",
            "task": "discharge_draft",
        },
    ),
    DocumentChunk(
        chunk_id="diagnosis_code_rule",
        source="docs/domain_glossary.md",
        text=(
            "Il mock Phase 1 usa codici diagnosi numerici ministeriali "
            "italiani ICD-9-CM. Non sono codici ICD-10, anche se alcune "
            "funzioni storiche usano nomi di parametro compatibili. English "
            "keywords: diagnosis codes."
        ),
        metadata={
            "doc_type": "glossary",
            "language": "it",
            "domain": "diagnosis_codes",
        },
    ),
    DocumentChunk(
        chunk_id="llm_uncertainty_limit",
        source="docs/uncertainty_limits.md",
        text=(
            "MC sampling confidence measures semantic self-consistency across "
            "model outputs. It does not prove clinical truth or correctness; "
            "clinicians must verify safety-critical content."
        ),
        metadata={
            "doc_type": "safety_rule",
            "language": "en",
            "domain": "uncertainty",
        },
    ),
]
