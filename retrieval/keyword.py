"""Deterministic local keyword retriever."""

from __future__ import annotations

import string

from retrieval.base import DocumentChunk, RetrievedChunk


class KeywordRetriever:
    """Simple token-overlap retriever with exact metadata filtering."""

    def __init__(self) -> None:
        self._chunks: list[DocumentChunk] = []
        self._chunk_tokens: dict[str, set[str]] = {}

    def index(self, chunks: list[DocumentChunk]) -> None:
        self._chunks = list(chunks)
        self._chunk_tokens = {
            chunk.chunk_id: set(_tokenize(chunk.text))
            for chunk in self._chunks
        }

    def search(
        self,
        query: str,
        top_k: int = 5,
        filters: dict[str, str] | None = None,
    ) -> list[RetrievedChunk]:
        query_tokens = set(_tokenize(query))
        candidates = [
            chunk for chunk in self._chunks
            if _matches_filters(chunk, filters)
        ]

        scored = [
            RetrievedChunk(
                chunk_id=chunk.chunk_id,
                source=chunk.source,
                text=chunk.text,
                score=_score(query_tokens, self._chunk_tokens.get(chunk.chunk_id, set())),
                metadata=chunk.metadata,
            )
            for chunk in candidates
        ]
        scored.sort(key=lambda chunk: (-chunk.score, chunk.chunk_id))

        positive = [chunk for chunk in scored if chunk.score > 0]
        if positive:
            return positive[:top_k]
        return scored[:top_k]


def _tokenize(text: str) -> list[str]:
    translation = str.maketrans({char: " " for char in string.punctuation})
    return text.lower().translate(translation).split()


def _matches_filters(chunk: DocumentChunk, filters: dict[str, str] | None) -> bool:
    if not filters:
        return True
    return all(chunk.metadata.get(key) == value for key, value in filters.items())


def _score(query_tokens: set[str], chunk_tokens: set[str]) -> float:
    if not query_tokens or not chunk_tokens:
        return 0.0
    overlap = query_tokens & chunk_tokens
    return len(overlap) / len(query_tokens)
