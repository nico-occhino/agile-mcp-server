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

from data.mock_store import get_patient, get_active_ricovero
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
    patient = get_patient(patient_id)
    if patient is None:
        return {
            "found": False,
            "error": f"No patient found with ID '{patient_id}'.",
        }

    # --- Step 1: Format the patient data as a structured string for the prompt ---
    # We deliberately flatten to text here rather than passing raw JSON to the LLM.
    # Text forces the model to read the data linearly, which produces more
    # consistent outputs than JSON (which the model might skip over or misparse).
    ricoveri = patient.get("ricoveri", [])
    latest_ricovero = (
        max(ricoveri, key=lambda r: r["data_ingresso"]) if ricoveri else None
    )

    patient_data_str = _format_patient_for_prompt(patient, latest_ricovero)

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
    patient = get_patient(patient_id)
    if patient is None:
        return {
            "found": False,
            "error": f"No patient found with ID '{patient_id}'.",
        }

    ricoveri = patient.get("ricoveri", [])
    if not ricoveri:
        return {
            "found": True,
            "error": "No admission records found. Cannot generate discharge letter.",
        }

    latest = max(ricoveri, key=lambda r: r["data_ingresso"])
    patient_data_str = _format_patient_for_prompt(patient, latest)

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
    Patient: {patient['nome']} {patient['cognome']}, born {patient['data_nascita']}
    Primary diagnosis: {flags.primary_diagnosis_description}
    Active conditions: {', '.join(flags.active_conditions) or 'none documented'}
    Medications to continue: {', '.join(flags.key_medications) or 'none'}
    Follow-up required: {'Yes' if flags.follow_up_required else 'No'}
    {f'Follow-up notes: {flags.follow_up_notes}' if flags.follow_up_notes else ''}

    Raw clinical notes: {latest.get('note_cliniche', 'not available')}

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
        "patient_name": f"{patient['nome']} {patient['cognome']}",
        "extracted_flags": flags.model_dump(),
        **uncertain_result.to_dict(),
    }


# ---------------------------------------------------------------------------
# Internal helper — format patient record for prompt injection
# ---------------------------------------------------------------------------

def _format_patient_for_prompt(patient: dict, ricovero: dict | None) -> str:
    """
    Convert a patient record into a clean text block for LLM prompts.

    Keeping this function separate means you can improve prompt formatting
    in one place and all features that use it benefit immediately.
    """
    lines = [
        f"Name: {patient['nome']} {patient['cognome']}",
        f"Date of birth: {patient['data_nascita']}",
        f"Sex: {patient.get('sesso', 'not specified')}",
        f"Municipality of residence: {patient.get('comune_residenza', 'not specified')}",
    ]

    if ricovero:
        lines += [
            "",
            f"Most recent admission:",
            f"  Ward (reparto): {ricovero['reparto']}",
            f"  Admission date: {ricovero['data_ingresso']}",
            f"  Discharge date: {ricovero.get('data_dimissione') or 'currently admitted'}",
            f"  Primary diagnosis (ICD-10): {ricovero['diagnosi_principale']}",
            f"  Secondary diagnoses: {', '.join(ricovero.get('diagnosi_secondarie', [])) or 'none'}",
            f"  Medications: {', '.join(ricovero.get('farmaci', [])) or 'none documented'}",
            f"  Clinical notes: {ricovero.get('note_cliniche', 'not available')}",
        ]
    else:
        lines.append("No recorded hospital admissions.")

    return "\n".join(lines)