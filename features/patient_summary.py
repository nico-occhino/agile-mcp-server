"""
features/patient_summary.py
----------------------------
LLM-based patient summary with uncertainty estimation.

--------------------------------------------------------------
The previous file (patient_lookup.py) was pure data retrieval — no LLM.
This file adds the LLM step AND the uncertainty layer on top.

The workflow for get_patient_summary():
  1. Retrieve structured patient data from the mock store (deterministic)
  2. Format it into a prompt (deterministic)
  3. Call the LLM N times with temperature > 0 (stochastic)
  4. Measure disagreement across N samples (uncertainty estimation)
  5. Return the best sample + confidence score (UncertainResult)

The workflow for get_patient_discharge_draft():
  This is the more complex version, motivated by Nocita and Sisca's
  discharge-letter idea. It shows how an MCP tool can internally run
  multiple LLM calls for different purposes (extraction, then generation).
"""

from __future__ import annotations

from data.mock_store import get_patient
from workflow.llm_client import call_llm, call_llm_n_times, call_llm_structured
from workflow.uncertainty import build_uncertain_result, UncertainResult
from pydantic import BaseModel


# ---------------------------------------------------------------------------
# Prompt templates — treat these as carefully as you would model weights
# They are the interface between your code and the LLM's understanding.
# ---------------------------------------------------------------------------

SUMMARY_SYSTEM = """
You are a clinical assistant supporting hospital staff. You will receive
structured patient data and must produce a concise clinical summary for
a physician.

Rules:
- Write 3–5 sentences maximum.
- Use clinical terminology appropriate for a physician audience.
- State the primary diagnosis, current status, and any active medications.
- Do NOT invent or infer information not present in the data.
- If a field is missing, omit it — do not say "not available".
- Respond in the same language as the patient's record (Italian for Italian names).
"""

SUMMARY_USER_TEMPLATE = """
Patient data:
{patient_data}

Write a concise clinical summary for the attending physician.
"""


# ---------------------------------------------------------------------------
# Structured extraction schema — used internally for discharge draft
# ---------------------------------------------------------------------------

class ClinicalFlags(BaseModel):
    """
    Key clinical flags extracted from a patient's notes.
    Used as an intermediate step in multi-call workflows.
    """
    primary_diagnosis_description: str   # human-readable, not just the ICD-10 code
    active_conditions: list[str]         # comorbidities worth flagging
    key_medications: list[str]           # medications that need continuity
    follow_up_required: bool
    follow_up_notes: str = ""


# ---------------------------------------------------------------------------
# Tool 1 — uncertain patient summary
# ---------------------------------------------------------------------------

def get_patient_summary(patient_id: str) -> dict:
    """
    Generate a concise clinical summary for a patient using an LLM.

    This tool uses uncertainty estimation: the LLM is called N times
    independently, and the response includes a confidence score reflecting
    how consistent the model's answers were. Low confidence means the query
    was ambiguous or the patient data is sparse — the clinician should verify.

    Args:
        patient_id: the patient's hospital identifier (e.g. "P001")

    Returns a dict with: result (the summary), confidence (0–1),
    confidence_level (HIGH/MEDIUM/LOW), and rationale.
    """
    record = get_patient(patient_id)
    if record is None or "patient" not in record:
        return {
            "found": False,
            "error": f"No patient found with ID '{patient_id}'.",
        }

    patient = record["patient"]
    event = record.get("event")

    # --- Step 1: Format the patient data as a structured string for the prompt ---
    # We deliberately flatten to text here rather than passing raw JSON to the LLM.
    # Text forces the model to read the data linearly, which produces more
    # consistent outputs than JSON (which the model might skip over or misparse).
    patient_data_str = _format_patient_for_prompt(patient, event)

    # --- Step 2: Call the LLM N times at temperature > 0 ---
    samples = call_llm_n_times(
        system=SUMMARY_SYSTEM,
        user=SUMMARY_USER_TEMPLATE.format(patient_data=patient_data_str),
        temperature=0.7,
    )

    # --- Step 3: Estimate uncertainty across the N samples ---
    uncertain_result: UncertainResult = build_uncertain_result(
        samples=samples,
        mode="freetext",  # summaries are narratives, not categorical labels
    )

    return {
        "found": True,
        "patient_id": patient_id,
        **uncertain_result.to_dict(),
    }


# ---------------------------------------------------------------------------
# Tool 2 — discharge draft (multi-step internal workflow)
# ---------------------------------------------------------------------------

def get_patient_discharge_draft(patient_id: str) -> dict:
    """
    Generate a draft discharge letter for a patient.

    This is a MULTI-STEP workflow inside a single MCP tool:
      Step A  Extract structured clinical flags from the patient record
              using a structured LLM call (deterministic, temperature=0).
      Step B  Use the extracted flags + raw notes to generate a narrative
              discharge letter section by section.
      Step C  Estimate uncertainty on the final draft.

    Why break it into A→B rather than one big call?
    Smaller, focused calls are more reliable than asking the model to
    "do everything at once". Step A constrains what goes into Step B,
    reducing hallucination surface and making the uncertainty estimate
    more meaningful.

    In Nocita and Sisca's terms: this is where a RAG component would slot in
    between Steps A and B — retrieve the department-specific letter template,
    then use it to guide generation. That's Sisca's thesis angle.
    Your angle is the uncertainty layer that wraps the whole thing.

    Args:
        patient_id: the patient's hospital identifier (e.g. "P001")
    """
    record = get_patient(patient_id)
    if record is None or "patient" not in record:
        return {
            "found": False,
            "error": f"No patient found with ID '{patient_id}'.",
        }

    patient = record["patient"]
    event = record.get("event")

    if not event:
        return {
            "found": True,
            "error": "No admission records found. Cannot generate discharge letter.",
        }

    patient_data_str = _format_patient_for_prompt(patient, event)

    # --- Step A: Structured extraction of clinical flags ---
    # Temperature = 0 here because we want deterministic extraction.
    # The clinical flags are facts, not narratives.
    flags: ClinicalFlags = call_llm_structured(
        system=(
            "You are a clinical data extractor. From the patient data provided, "
            "extract key clinical flags as structured JSON. Be precise and conservative — "
            "only include information explicitly present in the data."
        ),
        user=f"Patient data:\n{patient_data_str}",
        schema=ClinicalFlags,
        temperature=0.0,
    )

    # --- Step B: Generate discharge letter from structured flags ---
    # Now we call N times for uncertainty estimation.
    discharge_system = """
    You are a clinical documentation specialist. Write a professional discharge letter
    section for a hospital physician. The letter must be:
    - Factual and based only on the provided clinical flags.
    - Written in Italian (this is an Italian hospital).
    - Concise: 4–6 sentences for the clinical summary section.
    - Clear about follow-up requirements.
    """

    discharge_user = f"""
    Patient: {patient['name']} {patient['surname']}, born {patient['birthDate'][:10]}
    Primary diagnosis: {flags.primary_diagnosis_description}
    Active conditions: {', '.join(flags.active_conditions) or 'none documented'}
    Medications to continue: {', '.join(flags.key_medications) or 'none'}
    Follow-up required: {'Yes' if flags.follow_up_required else 'No'}
    {f'Follow-up notes: {flags.follow_up_notes}' if flags.follow_up_notes else ''}

    Raw clinical notes: {event.get('eventReason', 'not available')}

    Write the clinical summary section of the discharge letter.
    """

    samples = call_llm_n_times(
        system=discharge_system,
        user=discharge_user,
        temperature=0.6,  # slightly lower than summary — we want consistency
    )

    uncertain_result = build_uncertain_result(samples=samples, mode="freetext")

    return {
        "found": True,
        "patient_id": patient_id,
        "patient_name": f"{patient['name']} {patient['surname']}",
        "extracted_flags": flags.model_dump(),
        **uncertain_result.to_dict(),
    }


# ---------------------------------------------------------------------------
# Internal helper — format patient record for prompt injection
# ---------------------------------------------------------------------------

def _format_patient_for_prompt(patient: dict, event: dict | None) -> str:
    """
    Convert a patient record into a clean text block for LLM prompts.

    Keeping this function separate means you can improve prompt formatting
    in one place and all features that use it benefit immediately.
    """
    lines = [
        f"Name: {patient['name']} {patient['surname']}",
        f"Date of birth: {patient['birthDate'][:10]}",
    ]

    if patient.get("allergy"):
        lines.append(f"⚠️  KNOWN ALLERGY: {patient['allergy']}")

    if event:
        lines += [
            "",
            f"Most recent admission:",
            f"  Ward (reparto): {event['uoDescription']}",
            f"  Admission date: {event['dateStart'][:10]}",
            f"  Discharge date: {event.get('dateEnd') or 'currently admitted'}",
            f"  Primary diagnosis: {event.get('diagnosis', {}).get('primary')}",
            f"  Clinical notes: {event.get('eventReason', 'not available')}",
        ]
    else:
        lines.append("No recorded hospital admissions.")

    return "\n".join(lines)