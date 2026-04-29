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
# Free-text uncertainty — semantic similarity via sentence embeddings
# ---------------------------------------------------------------------------
# UPGRADE from Jaccard n-gram overlap to cosine similarity on dense embeddings.
#
# WHY THIS MATTERS FOR THE THESIS
# --------------------------------
# Jaccard on n-grams measures surface-form overlap: "pressione alta" and
# "ipertensione" share zero bigrams despite being semantically equivalent.
# This causes false LOW-confidence readings whenever the LLM paraphrases.
#
# Sentence embeddings map text into a high-dimensional semantic space where
# meaning-preserving paraphrases land close together (high cosine similarity)
# and genuinely different claims land far apart (low cosine similarity).
#
# MODEL CHOICE: all-MiniLM-L6-v2
#   - 22M parameters, runs on CPU in < 100ms per sentence
#   - Strong multilingual performance (handles Italian clinical text)
#   - No API key required, runs locally — important for clinical data privacy
#   - Trained on 1B+ sentence pairs, strong on paraphrase detection
#
# COSINE SIMILARITY MATH
# ----------------------
# Given embedding vectors u and v:
#   cos(u,v) = (u · v) / (||u|| * ||v||)   ∈ [-1, 1]
# For sentence embeddings in practice: ∈ [0, 1] for semantically related text.
# Mean pairwise cosine = confidence score.
# This replaces mean pairwise Jaccard.

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine

# Load once at module level — takes ~1s on first import, then cached.
# The model is downloaded to ~/.cache/huggingface on first use (~90MB).
_EMBEDDING_MODEL: SentenceTransformer | None = None

def _get_embedding_model() -> SentenceTransformer:
    global _EMBEDDING_MODEL
    if _EMBEDDING_MODEL is None:
        _EMBEDDING_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
    return _EMBEDDING_MODEL


def estimate_confidence_freetext(samples: list[str]) -> tuple[float, str]:
    """
    Given N free-text responses, estimate confidence via mean pairwise
    cosine similarity of sentence embeddings.

    Steps:
      1. Encode each sample into a 384-dim embedding vector.
      2. Compute the N×N cosine similarity matrix.
      3. Average the upper triangle (excluding diagonal) → confidence.
      4. Return the sample closest to the centroid as the representative.

    Confidence interpretation:
      ~1.0  → all samples say the same thing semantically → HIGH
      ~0.5  → samples share some themes but diverge in content → MEDIUM
      ~0.0  → samples are semantically unrelated → LOW (should not happen
              for well-formed clinical summaries; signals a prompt failure)
    """
    n = len(samples)
    if n == 0:
        return 0.0, ""
    if n == 1:
        return 1.0, samples[0]

    model = _get_embedding_model()

    # Shape: (N, 384)
    embeddings = model.encode(samples, convert_to_numpy=True)

    # Shape: (N, N) — pairwise cosine similarities
    sim_matrix = sklearn_cosine(embeddings)

    # Extract upper triangle (i < j), excluding diagonal (which is always 1.0)
    upper = [
        sim_matrix[i][j]
        for i in range(n)
        for j in range(i + 1, n)
    ]
    mean_sim = float(np.mean(upper))

    # Representative sample: the one with highest mean similarity to all others
    # (equivalent to finding the sample closest to the centroid)
    row_means = [
        np.mean([sim_matrix[i][j] for j in range(n) if j != i])
        for i in range(n)
    ]
    best_idx = int(np.argmax(row_means))

    return float(np.clip(mean_sim, 0.0, 1.0)), samples[best_idx]


def build_uncertain_result(samples: list[str], mode: str) -> UncertainResult:
    """
    Build an UncertainResult from a list of LLM samples.
    
    Args:
        samples: The raw text samples from the LLM.
        mode: "categorical" or "freetext"
    """
    if mode == "categorical":
        confidence, result = estimate_confidence_categorical(samples)
    elif mode == "freetext":
        confidence, result = estimate_confidence_freetext(samples)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    level = _level(confidence)
    
    if level == ConfidenceLevel.HIGH:
        rationale = "High agreement across LLM samples."
    elif level == ConfidenceLevel.MEDIUM:
        rationale = "Medium agreement across LLM samples — some variation detected."
    else:
        rationale = "Low agreement across LLM samples — results are inconsistent."

    return UncertainResult(
        result=result,
        confidence=confidence,
        confidence_level=level,
        samples=samples,
        rationale=rationale
    )