"""Provider-independent text generation with an OpenAI-compatible adapter."""

from __future__ import annotations

from typing import Protocol

import httpx


class TextGenerator(Protocol):
    async def generate(self, system: str, user: str) -> str: ...


class FakeGenerator:
    """Deterministic offline test double; never invents clinical assertions."""

    async def generate(self, system: str, user: str) -> str:
        return "Synthetic data only. Review the structured result."


class OpenAICompatibleGenerator:
    def __init__(self, client: httpx.AsyncClient, base_url: str, api_key: str, model: str):
        self.client = client
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model

    async def generate(self, system: str, user: str) -> str:
        response = await self.client.post(
            f"{self.base_url}/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={
                "model": self.model,
                "temperature": 0,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            },
        )
        response.raise_for_status()
        return str(response.json()["choices"][0]["message"]["content"])

