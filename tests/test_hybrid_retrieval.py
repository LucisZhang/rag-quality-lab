# -*- coding: utf-8 -*-
"""Unit tests for hybrid retrieval merge/dedupe/rerank logic (no models).

HybridRerankRAG.__init__ builds a vector store and downloads a cross-encoder,
so instances here are assembled via __new__ with a real BM25 index, a scripted
in-memory "vector store", and a scripted "reranker". Generation tests use a
fake LLM - no model is ever called.
"""

from __future__ import annotations

from types import SimpleNamespace

from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

from src.rag_pipelines import (
    HybridRerankRAG,
    _message_content_to_text,
    _tokenize,
    _unique_preserve_order,
)


class ScriptedVectorStore:
    """Returns a fixed list of documents for any query."""

    def __init__(self, docs: list[Document]) -> None:
        self._docs = docs

    def similarity_search(self, query: str, k: int = 4) -> list[Document]:
        return self._docs[:k]


class ScriptedReranker:
    """Scores each (query, text) pair from a fixed content -> score table."""

    def __init__(self, scores_by_content: dict[str, float]) -> None:
        self._scores = scores_by_content

    def predict(self, pairs: list[tuple[str, str]]) -> list[float]:
        return [self._scores[text] for _query, text in pairs]


class FakeLLM:
    """Records prompts and answers with a canned string."""

    def __init__(self, reply: str = "stub answer") -> None:
        self.reply = reply
        self.prompts: list[str] = []

    def invoke(self, prompt: str) -> SimpleNamespace:
        self.prompts.append(prompt)
        return SimpleNamespace(content=self.reply)


def make_doc(doc_id: str, content: str) -> Document:
    return Document(
        id=doc_id,
        page_content=content,
        metadata={"id": doc_id, "title": f"Title {doc_id}"},
    )


def build_hybrid_pipeline(
    chunks: list[Document],
    dense_docs: list[Document],
    reranker_scores: dict[str, float],
) -> HybridRerankRAG:
    pipeline = HybridRerankRAG.__new__(HybridRerankRAG)
    pipeline.chunks = chunks
    pipeline.tokenized_chunks = [_tokenize(doc.page_content) for doc in chunks]
    pipeline.bm25 = BM25Okapi(pipeline.tokenized_chunks) if pipeline.tokenized_chunks else None
    pipeline.vectorstore = ScriptedVectorStore(dense_docs)
    pipeline.reranker = ScriptedReranker(reranker_scores)
    return pipeline


def test_merge_deduplicates_documents_with_identical_content():
    shared = make_doc("kb1", "quantum widgets power the flux capacitor")
    other = make_doc("kb2", "an unrelated passage about databases")
    dense_only = make_doc("dense1", "a dense-only passage about caching")
    # The dense store returns a *copy* of the shared chunk: dedupe must key on
    # page_content, not object identity.
    dense_duplicate = make_doc("kb1-copy", shared.page_content)

    pipeline = build_hybrid_pipeline(
        chunks=[shared, other],
        dense_docs=[dense_duplicate, dense_only],
        reranker_scores={
            shared.page_content: 3.0,
            other.page_content: 1.0,
            dense_only.page_content: 2.0,
        },
    )

    results = pipeline.retrieve("quantum widgets", k=10)
    contents = [doc.page_content for doc in results]
    assert len(contents) == len(set(contents)), "duplicate content survived the merge"
    assert set(contents) == {shared.page_content, other.page_content, dense_only.page_content}


def test_reranker_scores_determine_final_order_and_k_truncation():
    doc_low = make_doc("low", "passage scored lowest by the reranker")
    doc_mid = make_doc("mid", "passage scored in the middle by the reranker")
    doc_high = make_doc("high", "passage scored highest by the reranker")

    pipeline = build_hybrid_pipeline(
        chunks=[doc_low, doc_mid, doc_high],
        dense_docs=[doc_low, doc_mid],  # dense order deliberately disagrees
        reranker_scores={
            doc_low.page_content: 0.1,
            doc_mid.page_content: 0.5,
            doc_high.page_content: 0.9,
        },
    )

    results = pipeline.retrieve("any query", k=2)
    assert [doc.metadata["id"] for doc in results] == ["high", "mid"]


def test_bm25_recovers_lexical_match_missed_by_dense_search():
    lexical = make_doc("lex", "zephyr calibration protocol for widget arrays")
    filler_a = make_doc("fa", "generic text about cooking recipes")
    filler_b = make_doc("fb", "generic text about garden furniture")

    pipeline = build_hybrid_pipeline(
        chunks=[lexical, filler_a, filler_b],
        dense_docs=[filler_a, filler_b],  # dense search misses the lexical hit
        reranker_scores={
            lexical.page_content: 5.0,
            filler_a.page_content: 0.2,
            filler_b.page_content: 0.1,
        },
    )

    results = pipeline.retrieve("zephyr calibration protocol", k=1)
    assert [doc.metadata["id"] for doc in results] == ["lex"]


def test_retrieve_with_empty_index_returns_empty_list():
    pipeline = build_hybrid_pipeline(chunks=[], dense_docs=[], reranker_scores={})
    assert pipeline.bm25 is None
    assert pipeline.retrieve("anything", k=4) == []


def test_query_returns_structured_result_with_deduped_doc_ids():
    doc_a = make_doc("a", "first context passage")
    doc_b = make_doc("b", "second context passage")

    pipeline = build_hybrid_pipeline(
        chunks=[doc_a, doc_b],
        dense_docs=[doc_a],
        reranker_scores={doc_a.page_content: 1.0, doc_b.page_content: 0.5},
    )
    pipeline.llm = FakeLLM(reply="  the answer  ")

    result = pipeline.query("what is in the passages?", k=2)

    assert result["question"] == "what is in the passages?"
    assert result["answer"] == "the answer"
    assert result["contexts"] == [doc_a.page_content, doc_b.page_content]
    assert result["retrieved_doc_ids"] == ["a", "b"]
    # The grounding prompt must contain the question and every context passage.
    assert len(pipeline.llm.prompts) == 1
    prompt = pipeline.llm.prompts[0]
    assert "what is in the passages?" in prompt
    assert doc_a.page_content in prompt
    assert doc_b.page_content in prompt


def test_tokenize_lowercases_and_splits_on_whitespace():
    assert _tokenize("Hello  WORLD\nfoo\tBar") == ["hello", "world", "foo", "bar"]
    assert _tokenize("") == []


def test_unique_preserve_order_keeps_first_occurrence():
    assert _unique_preserve_order(["b", "a", "b", "c", "a"]) == ["b", "a", "c"]
    assert _unique_preserve_order([]) == []


def test_message_content_to_text_handles_string_and_block_lists():
    assert _message_content_to_text("  plain  ") == "plain"
    assert _message_content_to_text(["part one", {"text": "part two"}]) == "part one\npart two"
    assert _message_content_to_text([{"no_text_key": 1}]) == ""
    assert _message_content_to_text(123) == "123"
