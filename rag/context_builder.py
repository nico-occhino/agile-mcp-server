"""Build prompt-ready RAG context from the Phase 1 domain corpus."""

from __future__ import annotations

from retrieval.keyword import KeywordRetriever
from rag.documents import PHASE1_DOMAIN_CORPUS


_RETRIEVER: KeywordRetriever | None = None


def build_rag_context(
    query: str,
    top_k: int = 3,
    filters: dict[str, str] | None = None,
) -> dict:
    """Return retrieved chunks and formatted context text for an LLM prompt."""
    retriever = _get_retriever()
    chunks = retriever.search(query=query, top_k=top_k, filters=filters)

    return {
        "query": query,
        "chunks": [chunk.model_dump() for chunk in chunks],
        "context_text": _format_context(chunks),
    }


def _get_retriever() -> KeywordRetriever:
    global _RETRIEVER
    if _RETRIEVER is None:
        _RETRIEVER = KeywordRetriever()
        _RETRIEVER.index(PHASE1_DOMAIN_CORPUS)
    return _RETRIEVER


def _format_context(chunks) -> str:
    lines: list[str] = []
    for idx, chunk in enumerate(chunks, start=1):
        lines.append(
            f"[{idx}] Source: {chunk.source} | Chunk: {chunk.chunk_id} | "
            f"Score: {chunk.score:.3f}"
        )
        lines.append(chunk.text)
        lines.append("")
    return "\n".join(lines).strip()
