"""Backend-agnostic retrieval models and protocol."""

from __future__ import annotations

from typing import Protocol

from pydantic import BaseModel, Field


class DocumentChunk(BaseModel):
    chunk_id: str
    source: str
    text: str
    metadata: dict[str, str] = Field(default_factory=dict)


class RetrievedChunk(BaseModel):
    chunk_id: str
    source: str
    text: str
    score: float
    metadata: dict[str, str] = Field(default_factory=dict)


class Retriever(Protocol):
    """Minimal interface for keyword, vector, and learned-index backends."""

    def index(self, chunks: list[DocumentChunk]) -> None:
        """Index chunks for later search."""

    def search(
        self,
        query: str,
        top_k: int = 5,
        filters: dict[str, str] | None = None,
    ) -> list[RetrievedChunk]:
        """Return the most relevant chunks for a query."""
