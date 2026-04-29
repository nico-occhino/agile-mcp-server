"""
workflow/llm_client.py
----------------------
A thin wrapper around any OpenAI-compatible LLM API.

WHY A WRAPPER AND NOT JUST `openai` DIRECTLY
---------------------------------------------
1.  Swappability. Nocita wants us to be able to change the LLM underneath
    without rewriting feature code. All features import `call_llm()` from here.
    To switch from GPT-4o-mini to Mistral to Agile's internal model, you change
    two environment variables and nothing else.

2.  The uncertainty-sampling pattern lives here. `call_llm_n_times()` runs N
    independent calls at temperature > 0 and returns all responses. This is the
    inference-time equivalent of MC Dropout — instead of T stochastic forward
    passes through a network with dropped neurons, you do N stochastic samples
    from the LLM's distribution. Your BioVid Bayesian CNN did this in latent
    space; here we do it in token space.

3.  A single place for error handling, retries, and observability.

OPENAI-COMPATIBLE APIs (reference for when you switch)
------------------------------------------------------
Provider       | base_url                              | notes
---------------|---------------------------------------|------------------------
OpenAI         | https://api.openai.com/v1             | default
Mistral        | https://api.mistral.ai/v1             | good free tier
Groq           | https://api.groq.com/openai/v1        | very fast inference
Together.ai    | https://api.together.xyz/v1           | many open models
Ollama (local) | http://localhost:11434/v1             | no API key needed
Agile internal | set AGILE_API_BASE_URL in .env        | mid-internship
"""

from __future__ import annotations

import os
from typing import Any
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel

load_dotenv()

# ---------------------------------------------------------------------------
# Client — instantiated once, reused across all calls
# ---------------------------------------------------------------------------

_client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("LLM_BASE_URL", "https://api.openai.com/v1"),
)

DEFAULT_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
UNCERTAINTY_SAMPLES = int(os.getenv("UNCERTAINTY_SAMPLES", "3"))


# ---------------------------------------------------------------------------
# Core call — one completion
# ---------------------------------------------------------------------------

def call_llm(
    system: str,
    user: str,
    temperature: float = 0.0,   # 0 = deterministic, good for structured extraction
    model: str | None = None,
) -> str:
    """
    Make a single LLM call and return the text response.

    Use temperature=0 when you want a deterministic answer (structured extraction,
    yes/no classification). Use temperature > 0 when you want sampling (narrative
    generation, uncertainty estimation).
    """
    response = _client.chat.completions.create(
        model=model or DEFAULT_MODEL,
        temperature=temperature,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    return response.choices[0].message.content or ""


# ---------------------------------------------------------------------------
# Uncertainty-sampling call — N completions, same prompt
# ---------------------------------------------------------------------------

def call_llm_n_times(
    system: str,
    user: str,
    n: int | None = None,
    temperature: float = 0.7,   # must be > 0 to get different samples
    model: str | None = None,
) -> list[str]:
    """
    Make N independent LLM calls at temperature > 0 and return all responses.

    This is the language-model analogue of MC Dropout:
      - MC Dropout:         T forward passes, each with a different dropout mask
      - This function:      N completions, each sampling from P(token | context)

    The variance across the N responses is our uncertainty signal.
    High variance → the model is unsure → low confidence.
    Low variance  → the model is consistent → high confidence.

    The caller (workflow/uncertainty.py) takes these N strings and computes
    a numerical confidence score. See that module for the math.

    WHY SEPARATE CALLS AND NOT `n=N` IN ONE API CALL
    -------------------------------------------------
    The OpenAI `n` parameter returns N completions from one request but they are
    correlated — they share the same KV-cache prefix. For true uncertainty
    estimation you want independent samples. Separate HTTP calls give you that.
    This is slower and costs more tokens; that's the price of honest uncertainty.
    For production you'd optimize; for a POC and a thesis it's fine.
    """
    n = n or UNCERTAINTY_SAMPLES
    return [
        call_llm(system=system, user=user, temperature=temperature, model=model)
        for _ in range(n)
    ]


# ---------------------------------------------------------------------------
# Structured output call — returns a Pydantic model
# ---------------------------------------------------------------------------

def call_llm_structured(
    system: str,
    user: str,
    schema: type[BaseModel],
    temperature: float = 0.0,
    model: str | None = None,
) -> Any:
    """
    Call the LLM with JSON mode and parse the response into a Pydantic model.

    This is the right call for structured extraction tasks (e.g. "extract the
    patient's primary diagnosis from this note"). The schema doubles as both the
    prompt constraint and the runtime validator — if the LLM returns malformed
    JSON, Pydantic raises immediately rather than propagating garbage downstream.

    Example schema:
        class DiagnosisExtraction(BaseModel):
            icd10_code: str
            description: str
            confidence: float   # self-reported by the LLM, 0–1

    Note: JSON mode is available on GPT-4o-mini and most Mistral models.
    For Llama-based models you may need to add explicit JSON instructions in
    the system prompt and fall back to manual parsing.
    """
    schema_json = schema.model_json_schema()
    augmented_system = (
        f"{system}\n\n"
        f"Respond ONLY with a JSON object that strictly follows this schema:\n"
        f"{schema_json}\n"
        f"No preamble, no markdown fences, no explanation — raw JSON only."
    )
    raw = call_llm(
        system=augmented_system,
        user=user,
        temperature=temperature,
        model=model,
    )
    # Strip accidental markdown fences from models that ignore instructions
    raw = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    return schema.model_validate_json(raw)