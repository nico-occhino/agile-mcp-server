"""
workflow/uncertainty.py
-----------------------
Uncertainty estimation for LLM-based tool calls via MC-sampling.

APPROACH
--------
query → [LLM × N samples at temperature > 0] → measure disagreement → confidence

For categorical responses: normalized Shannon entropy over the vote distribution.
For free-text responses:   mean pairwise cosine similarity of sentence embeddings.

The embedding approach replaces the original Jaccard n-gram overlap because
Jaccard is surface-form — "pressione alta" and "ipertensione" score 0.0 Jaccard
despite being semantically identical. Sentence embeddings capture meaning.

THRESHOLDS
----------
HIGH   >= 0.75   proceed automatically
MEDIUM [0.5, 0.75)  proceed with a note to clinician
LOW    < 0.5    refuse, ask for clarification
"""
from __future__ import annotations
import re
from collections import Counter
from dataclasses import dataclass
from enum import Enum
from typing import Any
import numpy as np
from scipy.stats import entropy as scipy_entropy
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine


class ConfidenceLevel(str, Enum):
    HIGH   = "HIGH"
    MEDIUM = "MEDIUM"
    LOW    = "LOW"


@dataclass
class UncertainResult:
    result: Any
    confidence: float
    confidence_level: ConfidenceLevel
    samples: list[str]
    rationale: str

    def to_dict(self) -> dict:
        return {"result": self.result, "confidence": round(self.confidence, 3), "confidence_level": self.confidence_level.value, "rationale": self.rationale}


HIGH_CONFIDENCE_THRESHOLD   = 0.75
MEDIUM_CONFIDENCE_THRESHOLD = 0.50


def _level(confidence: float) -> ConfidenceLevel:
    if confidence >= HIGH_CONFIDENCE_THRESHOLD:
        return ConfidenceLevel.HIGH
    if confidence >= MEDIUM_CONFIDENCE_THRESHOLD:
        return ConfidenceLevel.MEDIUM
    return ConfidenceLevel.LOW


def estimate_confidence_categorical(samples: list[str]) -> tuple[float, str]:
    """Confidence via normalized Shannon entropy over N categorical votes."""
    n = len(samples)
    if n == 0:
        return 0.0, ""
    normalised = [s.strip().lower() for s in samples]
    counts = Counter(normalised)
    majority = counts.most_common(1)[0][0]
    if len(counts) == 1:
        return 1.0, majority
    probs = np.array([c / n for c in counts.values()], dtype=float)
    raw_entropy = scipy_entropy(probs)
    max_entropy = np.log(n)
    confidence = 1.0 - (raw_entropy / max_entropy)
    return float(np.clip(confidence, 0.0, 1.0)), majority


_EMBEDDING_MODEL: SentenceTransformer | None = None


def _get_embedding_model() -> SentenceTransformer:
    global _EMBEDDING_MODEL
    if _EMBEDDING_MODEL is None:
        _EMBEDDING_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
    return _EMBEDDING_MODEL


def estimate_confidence_freetext(samples: list[str]) -> tuple[float, str]:
    """
    Confidence via mean pairwise cosine similarity of sentence embeddings.

    Steps:
      1. Encode each sample to a 384-dim vector (all-MiniLM-L6-v2).
      2. Compute N×N cosine similarity matrix.
      3. Average upper triangle (excluding diagonal) → confidence.
      4. Return sample closest to centroid as the representative.

    This replaces Jaccard n-gram overlap, which scores semantically equivalent
    Italian clinical paraphrases as 0.0 (wrong). Embeddings score them ~0.85 (correct).
    """
    n = len(samples)
    if n == 0:
        return 0.0, ""
    if n == 1:
        return 1.0, samples[0]
    model = _get_embedding_model()
    embeddings = model.encode(samples, convert_to_numpy=True)
    sim_matrix = sklearn_cosine(embeddings)
    upper = [sim_matrix[i][j] for i in range(n) for j in range(i + 1, n)]
    mean_sim = float(np.mean(upper))
    row_means = [np.mean([sim_matrix[i][j] for j in range(n) if j != i]) for i in range(n)]
    best_idx = int(np.argmax(row_means))
    return float(np.clip(mean_sim, 0.0, 1.0)), samples[best_idx]


def build_uncertain_result(samples: list[str], mode: str = "categorical") -> UncertainResult:
    """Build an UncertainResult from N LLM samples."""
    if mode == "freetext":
        confidence, result = estimate_confidence_freetext(samples)
    else:
        confidence, result = estimate_confidence_categorical(samples)
    level = _level(confidence)
    rationale_map = {
        ConfidenceLevel.HIGH:   f"All {len(samples)} samples highly consistent (confidence {confidence:.0%}). Proceeding automatically.",
        ConfidenceLevel.MEDIUM: f"Samples showed moderate agreement (confidence {confidence:.0%}). Clinician should verify.",
        ConfidenceLevel.LOW:    f"Samples disagreed substantially (confidence {confidence:.0%}). Clarification required.",
    }
    return UncertainResult(result=result, confidence=confidence, confidence_level=level, samples=samples, rationale=rationale_map[level])