# Theoretical Framework: Uncertainty-Aware Clinical MCP Server

This document outlines the methodological foundation for the Agile S.r.l. Model Context Protocol (MCP) server proof of concept. The system bridges the gap between structured clinical systems (SDO/API) and generative AI interfaces, ensuring high reliability and verifiable safety in a domain where hallucinations are unacceptable.

## 1. Architectural Paradigms
The system adopts three core paradigms:
1. **Model Context Protocol (MCP)**: A standardized JSON-RPC 2.0 interface that abstracts the hospital's internal APIs from the consuming LLM. The server exposes deterministic, strongly-typed tools that the client LLM can invoke.
2. **Repository Pattern for Data Decoupling**: The data layer is decoupled via a `PatientRepository` protocol. This allows seamless transition from the synthetic `mock_store.py` to the production `api_client.py` without modifying feature logic.
3. **Pydantic-driven Intermediate Representation (IR)**: The client parses natural language queries into a discriminated union of strictly validated Pydantic models. This ensures O(1) routing and structural guarantees before any backend tool is executed.

## 2. Deterministic vs. Stochastic Pathways
To mitigate hallucinations, the architecture strictly separates deterministic extraction from stochastic generation:
- **Parser (Temperature = 0.0)**: The query parser uses a strict, zero-temperature prompt. It is forced to emit only valid JSON conforming to the IR schema. If it fails, it emits an `UnknownIR`, gracefully halting the pipeline.
- **Synthesizer (Temperature > 0.0)**: When generating clinical summaries or discharge drafts, the model is permitted a non-zero temperature (e.g., 0.6 or 0.7) to ensure narrative fluency. However, this output is *never* trusted implicitly.

## 3. Uncertainty Estimation via Monte Carlo Sampling
The cornerstone of the system's safety is its uncertainty estimation layer. Since standard LLM pipelines do not output calibrated confidence bounds, we implement a Monte Carlo sampling approach:
1. **Sampling**: The synthesizer is queried $N$ times independently.
2. **Semantic Similarity**: The $N$ responses are embedded using a local `sentence-transformers` model (e.g., `all-MiniLM-L6-v2`).
3. **Disagreement Measurement**: The pairwise cosine similarity between embeddings is calculated. High variance indicates that the model is hallucinating or uncertain about the underlying clinical data.
4. **Safety Gating**: If the semantic agreement falls below an empirical threshold (e.g., 0.60), the system downgrades the confidence level to `LOW` and prompts the human clinician for verification.

## 4. Cohort Degradation Logic
For population-level queries (e.g., summarizing an entire ward), the system aggregates per-patient summaries. The aggregate cohort confidence is conservatively gated by the *minimum* confidence of any individual patient summary. A single hallucinated patient record cannot be hidden by high confidence in the remainder of the cohort.
