"""Create, decode, and authorize a local HS256 JWT demo token."""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import jwt

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from auth.jwt import decode_and_verify_jwt
from auth.permissions import has_required_permissions, required_permissions_for_tool


def main() -> None:
    secret = os.getenv("JWT_SECRET") or "local-demo-secret-change-before-real-use"
    if secret == "local-demo-secret-change-before-real-use":
        print("WARNING: JWT_SECRET is missing; using a local demo secret only.")

    os.environ["JWT_ALGORITHM"] = "HS256"
    os.environ["JWT_SECRET"] = secret

    issuer = os.getenv("JWT_ISSUER") or "aria"
    audience = os.getenv("JWT_AUDIENCE") or "agile-mcp-server"
    now = int(time.time())
    claims = {
        "sub": "user-123",
        "username": "nico",
        "role": "doctor",
        "department": "orthopedics",
        "permissions": ["patient:read", "summary:generate", "discharge:generate"],
        "iss": issuer,
        "aud": audience,
        "iat": now,
        "exp": now + 3600,
    }

    token = jwt.encode(claims, secret, algorithm="HS256")
    context = decode_and_verify_jwt(token)

    print("Demo JWT:")
    print(token)
    print("\nDecoded AuthContext:")
    print(context.model_dump())

    print("\nAuthorization checks:")
    for tool_name in [
        "get_patient_status",
        "get_patient_discharge_draft",
        "get_cohort_summary",
        "evaluate_input_prompt_guardrail",
    ]:
        required = required_permissions_for_tool(tool_name)
        authorized = has_required_permissions(context, tool_name)
        print(f"- {tool_name}: {authorized} required={required}")


if __name__ == "__main__":
    main()
