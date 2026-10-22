"""Issue a short-lived token for the synthetic-only offline demo."""

from __future__ import annotations

import os
import time

import jwt

from v2.settings import Settings


def make_demo_token(settings: Settings) -> str:
    return jwt.encode(
        {
            "sub": "synthetic-demo-user",
            "iss": settings.jwt_issuer,
            "aud": settings.jwt_audience,
            "exp": int(time.time()) + 900,
            "permissions": ["patient:read"],
        },
        settings.jwt_secret,
        algorithm="HS256",
    )


if __name__ == "__main__":
    if os.getenv("V2_DEMO_MODE", "false").lower() != "true":
        raise SystemExit("Set V2_DEMO_MODE=true for synthetic data only")
    print(make_demo_token(Settings.from_env()))

