"""Small Phase 1 RAG corpus and context builder."""

from rag.context_builder import build_rag_context
from rag.documents import PHASE1_DOMAIN_CORPUS

__all__ = ["PHASE1_DOMAIN_CORPUS", "build_rag_context"]
