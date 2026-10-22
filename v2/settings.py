"""Validated configuration for the isolated v2 services."""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    hospital_url: str
    jwt_secret: str
    jwt_issuer: str = "agile-v2-demo"
    jwt_audience: str = "agile-v2"
    input_guardrail_enabled: bool = True
    llm_base_url: str | None = None
    llm_api_key: str | None = None
    llm_model: str | None = None

    @classmethod
    def from_env(cls) -> Settings:
        demo = os.getenv("V2_DEMO_MODE", "false").lower() == "true"
        secret = os.getenv("V2_JWT_SECRET", "")
        if demo and not secret:
            secret = "synthetic-demo-only-secret-change-for-real-deployment"
        if len(secret) < 32:
            raise ValueError("V2_JWT_SECRET must contain at least 32 characters")
        return cls(
            hospital_url=os.getenv("V2_HOSPITAL_URL", "http://127.0.0.1:8081"),
            jwt_secret=secret,
            input_guardrail_enabled=os.getenv("V2_INPUT_GUARDRAIL", "true").lower() == "true",
            llm_base_url=os.getenv("V2_LLM_BASE_URL") or None,
            llm_api_key=os.getenv("V2_LLM_API_KEY") or None,
            llm_model=os.getenv("V2_LLM_MODEL") or None,
        )

