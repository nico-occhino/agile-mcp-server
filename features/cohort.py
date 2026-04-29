"""
features/cohort.py
------------------
Multi-step workflow composition for population-level queries.

--------------------------
patient_lookup.py  →  single patient, no LLM, deterministic
patient_summary.py →  single patient, LLM, uncertainty-aware
cohort.py          →  MULTIPLE patients, chained calls, uncertain aggregate

This file demonstrates "workflow composition" — a single MCP tool that
internally executes a sequence of operations against multiple records and
synthesizes the results.

Morana's example from the kick-off: "quanti sono i pazienti che hanno questa
patologia nell'ultima settimana?" is a population query that requires:
  1. Filtering patients by some criterion (diagnosis code, admission period)
  2. Fetching per-patient detail (possibly via multiple sub-calls)
  3. Aggregating or summarizing across the cohort

This is the data-engineering pattern: ETL inside a tool call.
  - E: Extract patients matching a filter from the data layer
  - T: Transform by enriching each record with an LLM-generated summary
  - L: Load into a structured response that the MCP client can display

THE UNCERTAINTY ANGLE IN COHORT CONTEXT
-----------------------------------------
Uncertainty in cohort tools is more nuanced than in single-patient tools:
  - Each per-patient sub-call has its own confidence score.
  - The *aggregate* has a cohort-level confidence = mean of per-patient scores
    weighted by how "complete" each patient's data is.
  - A cohort summary where some patients have LOW confidence is flagged
    differently from one where all are HIGH.

This is an open research question in your domain. A simple mean is the
baseline; you could explore:
  - Worst-case (min) confidence for safety-critical aggregates
  - Uncertainty propagation through the aggregation (formal Bayesian treatment)
  - Per-record flagging rather than aggregate collapse

These are thesis chapters, not just feature improvements.
"""

from __future__ import annotations

from datetime import date, timedelta
from data.mock_store import list_patients_by_diagnosis, get_patient, list_patients
from features.patient_lookup import get_patient_status
from features.patient_summary import get_patient_summary
from workflow.llm_client import call_llm
import numpy as np


# ---------------------------------------------------------------------------
# Tool 1 — cohort filter + status
# No LLM, demonstrates multi-record retrieval and aggregation
# ---------------------------------------------------------------------------

def get_patients_by_diagnosis(icd10_prefix: str) -> dict:
    """
    Return all patients whose most recent admission has a primary diagnosis
    matching the given ICD-10 prefix.

    This is a population query — the kind that an infection-control officer,
    a department head, or an epidemiologist would make.

    Args:
        icd10_prefix: ICD-10 code prefix to filter by (e.g. "J44" for BPCO,
                      "I21" for acute myocardial infarction, "I" for all
                      cardiovascular diagnoses)

    Returns a summary of matching patients with their current status.

    Example queries this answers:
      "How many BPCO patients are currently admitted?"
      "Give me all cardiology patients admitted this month."
    """
    matching = list_patients_by_diagnosis(icd10_prefix)

    if not matching:
        return {
            "icd10_prefix": icd10_prefix,
            "total_found": 0,
            "patients": [],
            "message": f"No patients found with diagnosis code starting with '{icd10_prefix}'.",
        }

    patients_detail = []
    for p in matching:
        patient = p.get("patient")
        if not patient: continue
        status = get_patient_status(patient["internalId"])  # reuse the single-patient tool
        patients_detail.append({
            "patient_id": patient["internalId"],
            "full_name": f"{patient['name']} {patient['surname']}",
            "data_nascita": patient["birthDate"][:10],
            **{k: v for k, v in status.items() if k not in ("found", "patient_id")},
        })

    return {
        "icd10_prefix": icd10_prefix,
        "total_found": len(patients_detail),
        "patients": patients_detail,
    }


# ---------------------------------------------------------------------------
# Tool 2 — cohort summary with per-patient LLM summaries
# The full multi-step workflow with aggregate uncertainty
# ---------------------------------------------------------------------------

def get_cohort_summary(icd10_prefix: str) -> dict:
    """
    Generate a cohort-level summary for all patients matching a diagnosis,
    including a per-patient LLM summary and an aggregate confidence score.

    This is a MULTI-STEP workflow:
      Step 1  Filter patients by diagnosis prefix (data retrieval, deterministic)
      Step 2  For each patient: call get_patient_summary() — an LLM call with
              its own uncertainty score
      Step 3  Aggregate: compute cohort-level confidence (mean of per-patient
              confidences), collect all summaries, flag low-confidence patients
      Step 4  Generate a one-paragraph cohort narrative via a final LLM call
              that synthesizes the per-patient summaries

    Args:
        icd10_prefix: ICD-10 code prefix (e.g. "J44", "I21", "I")

    Returns:
        cohort_narrative: one-paragraph summary of the cohort
        patients: per-patient summaries with individual confidence scores
        cohort_confidence: mean confidence across all patients
        flagged_patients: IDs of patients whose summaries had LOW confidence
    """

    # --- Step 1: filter ---
    matching = list_patients_by_diagnosis(icd10_prefix)
    if not matching:
        return {
            "icd10_prefix": icd10_prefix,
            "total_found": 0,
            "cohort_narrative": f"No patients found with ICD-10 prefix '{icd10_prefix}'.",
            "patients": [],
            "cohort_confidence": None,
            "flagged_patients": [],
        }

    # --- Step 2: per-patient summaries with uncertainty ---
    per_patient = []
    flagged = []

    for p in matching:
        patient = p.get("patient")
        if not patient: continue
        summary_result = get_patient_summary(patient["internalId"])

        confidence = summary_result.get("confidence", 0.0)
        level = summary_result.get("confidence_level", "LOW")

        if level == "LOW":
            flagged.append(patient["internalId"])

        per_patient.append({
            "patient_id": patient["internalId"],
            "full_name": f"{patient['name']} {patient['surname']}",
            "summary": summary_result.get("result", "Summary unavailable."),
            "confidence": confidence,
            "confidence_level": level,
        })

    # --- Step 3: aggregate confidence ---
    confidences = [pp["confidence"] for pp in per_patient]
    cohort_confidence = float(np.mean(confidences)) if confidences else 0.0

        # --- Step 3: aggregate confidence — gated, not averaged ---
    confidences = [pp["confidence"] for pp in per_patient]

    # Worst-case confidence: the weakest link determines the floor.
    # In a clinical setting, mean confidence is misleading — one patient
    # with a failed summary is not compensated by nine successful ones.
    # We report both so the client can make an informed decision.
    cohort_confidence_mean = float(np.mean(confidences)) if confidences else 0.0
    cohort_confidence_min  = float(np.min(confidences))  if confidences else 0.0

    # Safety gate: if ANY patient falls below the LOW threshold (0.5),
    # force the cohort-level confidence to LOW regardless of the mean.
    SAFETY_THRESHOLD = 0.50
    if cohort_confidence_min < SAFETY_THRESHOLD:
        cohort_confidence = cohort_confidence_min
        cohort_confidence_note = (
            f"Cohort confidence degraded to minimum ({cohort_confidence_min:.0%}) "
            f"because {len(flagged)} patient(s) had LOW individual confidence. "
            f"Mean would have been {cohort_confidence_mean:.0%} — do not use mean in clinical context."
        )
    else:
        cohort_confidence = cohort_confidence_mean
        cohort_confidence_note = (
            f"All patients above safety threshold. "
            f"Mean confidence: {cohort_confidence_mean:.0%}, min: {cohort_confidence_min:.0%}."
        )

        # --- Step 4: cohort narrative via one more LLM call ---
        # We take the per-patient summaries (already vetted for confidence)
        # and ask the model to synthesize them into a cohort-level view.
        # This is NOT a raw dump of patient data into the model —
        # it's a synthesis of summaries already produced by the model,
        # which is safer than starting from scratch with all records.
        summaries_block = "\n\n".join(
            f"[{pp['patient_id']}] {pp['full_name']}:\n{pp['summary']}"
            for pp in per_patient
        )

    cohort_narrative = call_llm(
        system=(
            "You are a clinical informatics assistant. You will receive summaries of "
            "multiple patients sharing a common diagnosis category. Write a single cohesive "
            "paragraph (4–6 sentences) that characterizes this patient cohort: common patterns, "
            "notable variations, and any concerns a department head should be aware of. "
            "Do not list individual patients by name. Focus on patterns."
        ),
        user=(
            f"Diagnosis category (ICD-10 prefix): {icd10_prefix}\n"
            f"Number of patients: {len(per_patient)}\n\n"
            f"Per-patient summaries:\n{summaries_block}\n\n"
            f"Write the cohort narrative."
        ),
        temperature=0.4,   # some creativity but not too much; this is clinical text
    )

    return {
    "icd10_prefix": icd10_prefix,
    "total_found": len(per_patient),
    "cohort_confidence": round(cohort_confidence, 3),
    "cohort_confidence_mean": round(cohort_confidence_mean, 3),
    "cohort_confidence_min": round(cohort_confidence_min, 3),
    "cohort_confidence_note": cohort_confidence_note,
    "flagged_patients": flagged,
    "cohort_narrative": cohort_narrative,
    "patients": per_patient,
    }


# ---------------------------------------------------------------------------
# Tool 3 — recently admitted patients (time-window filter)
# Demonstrates the temporal filtering pattern Morana described
# ---------------------------------------------------------------------------

def get_recently_admitted(days: int = 7) -> dict:
    """
    Return all patients admitted within the last N days.

    Args:
        days: look-back window in days (default: 7)

    Example query: "How many patients were admitted to hospital this week?"
    This is a temporal cohort query — very common in clinical operations.
    """
    cutoff = date.today() - timedelta(days=days)
    cutoff_str = cutoff.isoformat()

    all_patients = list_patients()
    recent = []

    for p_summary in all_patients:
        record = get_patient(p_summary["internalId"])
        if not record:
            continue
        patient = record.get("patient")
        event = record.get("event")
        
        if patient and event:
            if event["dateStart"][:10] >= cutoff_str:
                recent.append({
                    "patient_id": patient["internalId"],
                    "full_name": f"{patient['name']} {patient['surname']}",
                    "reparto": event["uoDescription"],
                    "data_ingresso": event["dateStart"][:10],
                    "diagnosi_principale": event.get("diagnosis", {}).get("primary"),
                    "stato": "ricoverato" if event.get("dateEnd") is None else "dimesso",
                })

    return {
        "look_back_days": days,
        "cutoff_date": cutoff_str,
        "total_found": len(recent),
        "admissions": sorted(recent, key=lambda x: x["data_ingresso"], reverse=True),
    }