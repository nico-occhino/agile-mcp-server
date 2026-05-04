from rag.documents import PHASE1_DOMAIN_CORPUS
from retrieval.keyword import KeywordRetriever


def _indexed_retriever() -> KeywordRetriever:
    retriever = KeywordRetriever()
    retriever.index(PHASE1_DOMAIN_CORPUS)
    return retriever


def test_keyword_retrieval_returns_allergy_safety_rule():
    results = _indexed_retriever().search("allergy safety discharge", top_k=3)
    chunk_ids = [chunk.chunk_id for chunk in results]
    assert "allergy_safety_rule" in chunk_ids


def test_keyword_retrieval_metadata_filter_template():
    results = _indexed_retriever().search(
        "discharge",
        top_k=3,
        filters={"doc_type": "template"},
    )
    assert results
    assert results[0].chunk_id == "discharge_summary_template"
    assert all(chunk.metadata["doc_type"] == "template" for chunk in results)
