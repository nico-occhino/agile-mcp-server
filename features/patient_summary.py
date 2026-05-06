"""
features/patient_summary.py
----------------------------
LLM-based patient summary with uncertainty estimation.

Workflow:
  1. Retrieve record from mock store (event + patient blocks)
  2. Format into a prompt — allergy always included if present
  3. Call LLM N times at temperature > 0 (stochastic sampling)
  4. Estimate confidence via semantic similarity (cosine on embeddings)
  5. Return best sample + UncertainResult
"""

from __future__ import annotations
import time
from data.repository import get_repository
from guardrails.decision import evaluate_guardrail
from guardrails.evaluation import measure_latency_ms
from rag.context_builder import build_rag_context

def _get_repo():
    return get_repository()
from workflow.llm_client import call_llm, call_llm_n_times, call_llm_structured
from workflow.uncertainty import build_uncertain_result, UncertainResult
from pydantic import BaseModel


SUMMARY_SYSTEM = """
You are a clinical assistant supporting hospital staff. You will receive
structured patient data and must produce a concise clinical summary for
a physician.

Rules:
- Write 3–5 sentences maximum.
- Use clinical terminology appropriate for a physician audience.
- State the primary diagnosis, current status, and reason for admission.
- If an allergy is listed, ALWAYS mention it in the summary.
- Do NOT invent or infer information not present in the data.
- If a field is missing, omit it — do not say "not available".
- Respond in Italian.
"""

SUMMARY_USER_TEMPLATE = """
Patient data:
{patient_data}

Write a concise clinical summary for the attending physician.
"""


class ClinicalFlags(BaseModel):
    """Structured clinical flags extracted for discharge draft generation."""
    primary_diagnosis_description: str
    active_conditions: list[str]
    key_medications: list[str]
    known_allergies: list[str]      # extracted from patient.allergy — safety-critical
    follow_up_required: bool
    follow_up_notes: str = ""


def get_patient_summary(patient_id: str) -> dict:
    """
    Generate a concise clinical summary for a patient using an LLM.

    Calls the LLM N times independently and returns the most consistent
    response together with a semantic confidence score (0–1).
    LOW confidence means the model was uncertain — clinician must verify.

    Args:
        patient_id: numeric hospital identifier (e.g. "45")
    """
    record = _get_repo().get_patient(patient_id)
    if record is None:
        return {
            "found": False,
            "error": f"No patient found with ID '{patient_id}'.",
        }

    demo = record["patient"]
    event = record.get("event")
    patient_data_str = _format_patient_for_prompt(demo, event)

    generation_start = time.perf_counter()
    samples = call_llm_n_times(
        system=SUMMARY_SYSTEM,
        user=SUMMARY_USER_TEMPLATE.format(patient_data=patient_data_str),
        temperature=0.7,
    )

    uncertain_result: UncertainResult = build_uncertain_result(
        samples=samples,
        mode="freetext",
    )
    generation_end = time.perf_counter()
    guardrail = evaluate_guardrail(
        task_type="patient_summary",
        confidence=uncertain_result.confidence,
        metadata={
            "latency_ms": measure_latency_ms(generation_start, generation_end),
            "uncertainty_samples": len(samples),
        },
    )

    return {
        "found": True,
        "patient_id": patient_id,
        "patient_name": f"{demo['name']} {demo['surname']}",
        "allergy": demo.get("allergy"),
        "guardrail": guardrail.model_dump(),
        **uncertain_result.to_dict(),
    }


def get_patient_discharge_draft(patient_id: str) -> dict:
    """
    Generate a draft discharge letter section for a patient.

    Multi-step internal workflow:
      Step A  Extract structured clinical flags (deterministic, temp=0)
      Step B  Generate discharge narrative from flags (N samples, temp=0.6)
      Step C  Estimate uncertainty on the draft

    Args:
        patient_id: numeric hospital identifier (e.g. "45")
    """
    record = _get_repo().get_patient(patient_id)
    if record is None:
        return {
            "found": False,
            "error": f"No patient found with ID '{patient_id}'.",
        }

    demo = record["patient"]
    event = record.get("event")

    if event is None:
        return {
            "found": True,
            "patient_id": patient_id,
            "patient_name": f"{demo['name']} {demo['surname']}",
            "error": "No admission record found. Cannot generate discharge letter.",
        }

    patient_data_str = _format_patient_for_prompt(demo, event)
    rag_context = build_rag_context(
        "discharge summary template allergy safety diagnosis follow-up",
        top_k=3,
        filters={"language": "it"},
    )

    # Step A: structured extraction
    try:
        flags: ClinicalFlags = call_llm_structured(
        system=(
            "You are a clinical data extractor. Extract key clinical flags from "
            "the patient data as structured JSON. Be conservative — only include "
            "information explicitly present in the data."
        ),
        user=f"Patient data:\n{patient_data_str}",
        schema=ClinicalFlags,
        temperature=0.0,
    )

    except Exception as exc:
        guardrail = evaluate_guardrail(
            task_type="discharge_draft",
            confidence=None,
            validation_issues=["Structured clinical flag extraction failed."],
            metadata={"stage": "clinical_flags_extraction"},
        )
        return {
            "found": True,
            "patient_id": patient_id,
            "patient_name": f"{demo['name']} {demo['surname']}",
            "error": "Structured clinical flag extraction failed. Cannot generate discharge letter.",
            "details": str(exc),
            "rag_context": _rag_context_metadata(rag_context),
            "guardrail": guardrail.model_dump(),
        }

    discharge_system = """
    You are a clinical documentation specialist. Write a professional discharge
    letter section for a hospital physician. Requirements:
    - Factual, based only on the provided clinical flags.
    - Written in Italian.
    - 4–6 sentences for the clinical summary section.
    - If allergies are present, include them prominently.
    - Clear about follow-up requirements.
    """

    allergy_line = (
        f"Known allergies: {', '.join(flags.known_allergies)}"
        if flags.known_allergies else "No known allergies."
    )

    discharge_user = f"""
    Domain context retrieved from RAG.
    This context contains templates, glossary, and safety rules only.
    It is NOT a source of patient facts:
    {rag_context['context_text']}

    Patient: {demo['name']} {demo['surname']}, born {demo['birthDate'][:10]}
    Primary diagnosis: {flags.primary_diagnosis_description}
    Active conditions: {', '.join(flags.active_conditions) or 'none documented'}
    {allergy_line}
    Medications to continue: {', '.join(flags.key_medications) or 'none'}
    Follow-up required: {'Yes' if flags.follow_up_required else 'No'}
    {f'Follow-up notes: {flags.follow_up_notes}' if flags.follow_up_notes else ''}

    Reason for admission: {event.get('eventReason', 'not documented')}

    Write the clinical summary section of the discharge letter.
    """

    generation_start = time.perf_counter()
    samples = call_llm_n_times(
        system=discharge_system,
        user=discharge_user,
        temperature=0.6,
    )

    uncertain_result = build_uncertain_result(samples=samples, mode="freetext")
    generation_end = time.perf_counter()
    guardrail = evaluate_guardrail(
        task_type="discharge_draft",
        confidence=uncertain_result.confidence,
        metadata={
            "latency_ms": measure_latency_ms(generation_start, generation_end),
            "uncertainty_samples": len(samples),
        },
    )

    return {
        "found": True,
        "patient_id": patient_id,
        "patient_name": f"{demo['name']} {demo['surname']}",
        "extracted_flags": flags.model_dump(),
        "rag_context": _rag_context_metadata(rag_context),
        "guardrail": guardrail.model_dump(),
        **uncertain_result.to_dict(),
    }


def _rag_context_metadata(rag_context: dict) -> dict:
    return {
        "query": rag_context["query"],
        "sources": [
            {
                "chunk_id": chunk["chunk_id"],
                "source": chunk["source"],
                "score": chunk["score"],
                "metadata": chunk["metadata"],
            }
            for chunk in rag_context["chunks"]
        ],
    }


def _format_patient_for_prompt(demo: dict, event: dict | None) -> str:
    """Format a patient record into a clean text block for LLM prompts."""
    lines = [
        f"Name: {demo['name']} {demo['surname']}",
        f"Date of birth: {demo['birthDate'][:10]}",
        f"Fiscal code: {demo.get('fiscalCode', 'not provided')}",
    ]

    # Allergy is safety-critical — always surface it
    allergy = demo.get("allergy")
    if allergy:
        lines.append(f"⚠️  KNOWN ALLERGY: {allergy}")
    else:
        lines.append("Allergies: none documented")

    if event:
        lines += [
            "",
            "Current/most recent admission:",
            f"  Ward: {event.get('uoDescription', 'unknown')}",
            f"  Presidio: {event.get('namePresidio', 'unknown')}",
            f"  Admission date: {event.get('dateStart', '')[:10]}",
            f"  Discharge date: {event.get('dateEnd', '')[:10] if event.get('dateEnd') else 'currently admitted'}",
            f"  Event type: {event.get('eventType', 'unknown')}",
            f"  Source: {event.get('eventSource', 'unknown')}",
            f"  Primary diagnosis code: {event.get('diagnosis', {}).get('primary', 'unknown')}",
            f"  DRG code: {event.get('drg', {}).get('code', 'unknown')} "
            f"(MDC {event.get('drg', {}).get('mdc', '?')})",
            f"  Reason for admission: {event.get('eventReason', 'not documented')}",
        ]
    else:
        lines.append("No recorded hospital admissions.")

    return "\n".join(lines)
