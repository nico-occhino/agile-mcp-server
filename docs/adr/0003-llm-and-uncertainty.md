# ADR 0003: Optional LLM and explicit review

Status: accepted for the v2 portfolio demonstration.

Deterministic parsing and data retrieval work with no API key. An optional
OpenAI-compatible generator may interpret otherwise unsupported phrasing or
summarize structured synthetic facts. Its parser output must pass typed IR
validation. LLM summaries carry `human_review_required`; provider failures
return a safe review status without echoing prompts or credentials.

Legacy self-consistency scores are not clinical calibration. V2 therefore
does not present a probability of correctness. A genuine calibration study
would require labelled domain cases, independent clinical review, error
analysis, and predefined acceptance criteria before any safety claim.

