"""
workflow/uncertainty.py
-----------------------
Uncertainty estimation for LLM-based tool calls.

-----------------------------------------
Standard LLM tool-use pipelines do this:

    query → LLM → answer

They return the answer with no indication of whether the model was
confident or guessing. In a clinical setting this is dangerous — a
confidently-wrong discharge summary is worse than a system that says
"I'm not sure, please verify."

This module implements an MC-sampling approach:

    query → [LLM × N independent samples] → measure disagreement → answer + confidence

N stochastic LLM samples at temperature > 0.
Semantic disagreement across responses = parameters uncertainty.

The math is: you're estimating E[f(x)] and Var[f(x)]
under a distribution induced by stochasticity (temperature sampling).

WHAT WE MEASURE
---------------
For *free-text* responses (summaries, narratives):
    We extract key claims from each sample and measure overlap via Jaccard
    similarity. Low overlap = high uncertainty.

For *structured* responses (classifications, extractions):
    We treat each sample as a vote and compute the normalized Shannon entropy
    of the vote distribution.

    Entropy = -Σ p_i * log(p_i)
    Confidence = 1 - (entropy / log(N))   # normalized to [0, 1]

    If all N samples agree:     entropy = 0,       confidence = 1.0
    If all N samples disagree:  entropy = log(N),  confidence = 0.0

    This is exactly what you computed in the Bayesian CNN for multi-class
    uncertainty, just without the softmax — the "class probabilities" are
    the empirical frequencies of each response category across N samples.

THRESHOLDS (tunable)
--------------------
HIGH_CONFIDENCE  >= 0.75   proceed automatically
MEDIUM_CONFIDENCE [0.5, 0.75)  proceed but flag to clinician
LOW_CONFIDENCE   < 0.5    refuse and ask for clarification
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np
from scipy.stats import entropy as scipy_entropy


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

class ConfidenceLevel(str, Enum):
    HIGH   = "HIGH"    # proceed automatically
    MEDIUM = "MEDIUM"  # proceed with a note to the clinician
    LOW    = "LOW"     # refuse, ask for clarification


@dataclass
class UncertainResult:
    """
    Wraps any feature response with an uncertainty estimate.

    This is the standard return type for all MCP tools that use the LLM
    internally. The MCP client (Agile's gateway) can inspect `confidence`
    and decide whether to surface the result directly or trigger a
    clarification flow.

    Fields
    ------
    result          : the actual answer (string, dict, anything)
    confidence      : float in [0, 1]; 1 = fully confident, 0 = random guess
    confidence_level: categorical bucket (HIGH / MEDIUM / LOW)
    samples         : the raw N LLM responses that produced this estimate
    rationale       : one sentence explaining the confidence level
    """
    result: Any
    confidence: float
    confidence_level: ConfidenceLevel
    samples: list[str]
    rationale: str

    def to_dict(self) -> dict:
        """Serialise to a plain dict for MCP tool return values."""
        return {
            "result": self.result,
            "confidence": round(self.confidence, 3),
            "confidence_level": self.confidence_level.value,
            "rationale": self.rationale,
            # omit raw samples from MCP response (too verbose); include in logs
        }


# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------

HIGH_CONFIDENCE_THRESHOLD   = 0.75
MEDIUM_CONFIDENCE_THRESHOLD = 0.50


def _level(confidence: float) -> ConfidenceLevel:
    if confidence >= HIGH_CONFIDENCE_THRESHOLD:
        return ConfidenceLevel.HIGH
    if confidence >= MEDIUM_CONFIDENCE_THRESHOLD:
        return ConfidenceLevel.MEDIUM
    return ConfidenceLevel.LOW


# ---------------------------------------------------------------------------
# Structured uncertainty — for categorical / extractive responses
# ---------------------------------------------------------------------------

def estimate_confidence_categorical(samples: list[str]) -> tuple[float, str]:
    """
    Given N categorical responses (e.g. the same patient ID extracted N times),
    compute confidence via normalized Shannon entropy.

    Returns (confidence: float, majority_answer: str)

    Example
    -------
    samples = ["P001", "P001", "P002"]
    → counts = {"P001": 2, "P002": 1}
    → probs  = [0.667, 0.333]
    → entropy = 0.918 bits (out of log(3) = 1.099 max)
    → confidence = 1 - 0.918/1.099 = 0.164   (low — the samples disagreed)

    samples = ["P001", "P001", "P001"]
    → entropy = 0
    → confidence = 1.0   (the model was perfectly consistent)
    """
    n = len(samples)
    if n == 0:
        return 0.0, ""

    # Normalise by stripping whitespace and lowercasing for robust comparison
    normalised = [s.strip().lower() for s in samples]
    counts = Counter(normalised)
    majority = counts.most_common(1)[0][0]

    if len(counts) == 1:
        # Perfect agreement — zero entropy
        return 1.0, majority

    probs = np.array([c / n for c in counts.values()], dtype=float)
    # scipy_entropy uses natural log by default; we normalise to [0,1]
    raw_entropy = scipy_entropy(probs)
    max_entropy = np.log(n)       # maximum possible entropy for n samples
    confidence = 1.0 - (raw_entropy / max_entropy)

    return float(np.clip(confidence, 0.0, 1.0)), majority


# ---------------------------------------------------------------------------
# Free-text uncertainty — for narrative / summary responses
# ---------------------------------------------------------------------------

def _extract_key_phrases(text: str) -> set[str]:
    """
    Extract 2–3 gram phrases from a text for overlap computation.
    Minimal NLP — no external dependencies beyond stdlib.
    When Nocita gives us proper clinical text, swap this for spaCy or a
    domain-specific extractor.
    """
    # Lowercase, keep only alphanumeric + spaces
    clean = re.sub(r"[^a-zA-Z0-9\s]", " ", text.lower())
    tokens = clean.split()
    bigrams  = {" ".join(tokens[i:i+2]) for i in range(len(tokens) - 1)}
    trigrams = {" ".join(tokens[i:i+3]) for i in range(len(tokens) - 2)}
    return bigrams | trigrams


def estimate_confidence_freetext(samples: list[str]) -> tuple[float, str]:
    """
    Given N free-text responses (e.g. patient summaries), estimate confidence
    via pairwise Jaccard similarity of key phrases.

    Jaccard(A, B) = |A ∩ B| / |A ∪ B|

    Mean pairwise Jaccard across all pairs = confidence score.

    Returns (confidence: float, representative_sample: str)
    The representative sample is the one with the highest average similarity
    to all other samples — the "consensus" response.
    """
    n = len(samples)
    if n == 0:
        return 0.0, ""
    if n == 1:
        return 1.0, samples[0]

    phrase_sets = [_extract_key_phrases(s) for s in samples]

    # Compute pairwise Jaccard similarities
    similarities: list[float] = []
    per_sample_sim: list[float] = [0.0] * n

    for i in range(n):
        for j in range(i + 1, n):
            a, b = phrase_sets[i], phrase_sets[j]
            union = a | b
            if not union:
                sim = 1.0
            else:
                sim = len(a & b) / len(union)
            similarities.append(sim)
            per_sample_sim[i] += sim
            per_sample_sim[j] += sim

    mean_similarity = float(np.mean(similarities))

    # Pick the sample that is most similar to all others as the representative
    best_idx = int(np.argmax(per_sample_sim))
    return float(np.clip(mean_similarity, 0.0, 1.0)), samples[best_idx]


# ---------------------------------------------------------------------------
# High-level factory — what features actually call
# ---------------------------------------------------------------------------

def build_uncertain_result(
    samples: list[str],
    mode: str = "categorical",   # "categorical" | "freetext"
) -> UncertainResult:
    """
    Given N raw LLM samples, build an UncertainResult.

    Parameters
    ----------
    samples : list of N raw LLM response strings
    mode    : how to measure disagreement
              "categorical" — each sample is a short label/value
              "freetext"    — each sample is a multi-sentence narrative

    Returns
    -------
    UncertainResult ready to be returned by an MCP tool.
    """
    if mode == "freetext":
        confidence, best_sample = estimate_confidence_freetext(samples)
    else:
        confidence, best_sample = estimate_confidence_categorical(samples)

    level = _level(confidence)

    rationale_map = {
        ConfidenceLevel.HIGH: (
            f"All {len(samples)} samples were highly consistent "
            f"(confidence {confidence:.0%}). Proceeding automatically."
        ),
        ConfidenceLevel.MEDIUM: (
            f"Samples showed moderate agreement "
            f"(confidence {confidence:.0%}). Result surfaced with a note — "
            f"clinician should verify."
        ),
        ConfidenceLevel.LOW: (
            f"Samples disagreed substantially "
            f"(confidence {confidence:.0%}). The query may be ambiguous or "
            f"the model lacks enough context. Clarification required."
        ),
    }

    return UncertainResult(
        result=best_sample,
        confidence=confidence,
        confidence_level=level,
        samples=samples,
        rationale=rationale_map[level],
    )