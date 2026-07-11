# -*- coding: utf-8 -*-
"""Unit tests for the model-free chunking logic in src/utils.py."""

from __future__ import annotations

from langchain_core.documents import Document

from src.utils import chunk_documents

# Production values used by BaseRAGPipeline.__init__.
CHUNK_SIZE = 512
CHUNK_OVERLAP = 64


def make_doc(doc_id: str, content: str, title: str | None = None) -> Document:
    return Document(
        id=doc_id,
        page_content=content,
        metadata={"id": doc_id, "title": title or f"Title {doc_id}"},
    )


def make_long_content(word_count: int = 400) -> str:
    """Unique, whitespace-separated tokens so overlap can be traced exactly."""
    return " ".join(f"word{index:04d}" for index in range(word_count))


def test_short_document_stays_a_single_chunk():
    doc = make_doc("doc1", "A short document well under the chunk size.")
    chunks = chunk_documents([doc], chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    assert len(chunks) == 1
    assert chunks[0].page_content == doc.page_content


def test_long_document_splits_within_chunk_size():
    doc = make_doc("doc1", make_long_content())
    chunks = chunk_documents([doc], chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    assert len(chunks) > 1
    assert all(len(chunk.page_content) <= CHUNK_SIZE for chunk in chunks)
    assert all(chunk.page_content.strip() for chunk in chunks)


def test_chunks_preserve_source_metadata():
    doc = make_doc("doc42", make_long_content(), title="Metadata Title")
    chunks = chunk_documents([doc], chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    assert len(chunks) > 1
    for chunk in chunks:
        assert chunk.metadata["id"] == "doc42"
        assert chunk.metadata["title"] == "Metadata Title"


def test_consecutive_chunks_share_overlap_text():
    doc = make_doc("doc1", make_long_content())
    chunks = chunk_documents([doc], chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    assert len(chunks) > 1
    for previous, current in zip(chunks, chunks[1:]):
        last_word = previous.page_content.split()[-1]
        assert last_word in current.page_content.split(), (
            "expected the tail of each chunk to reappear at the head of the next "
            f"(overlap={CHUNK_OVERLAP}), but {last_word!r} is missing"
        )


def test_no_content_is_lost_when_splitting():
    doc = make_doc("doc1", make_long_content())
    chunks = chunk_documents([doc], chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    original_words = set(doc.page_content.split())
    chunked_words = set()
    for chunk in chunks:
        chunked_words.update(chunk.page_content.split())
    assert chunked_words == original_words


def test_multiple_documents_chunk_independently_and_keep_order():
    doc_a = make_doc("docA", make_long_content())
    doc_b = make_doc("docB", "Short second document.")
    chunks = chunk_documents([doc_a, doc_b], chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)

    ids_in_order = [chunk.metadata["id"] for chunk in chunks]
    assert set(ids_in_order) == {"docA", "docB"}
    # All docA chunks come before the docB chunk (splitter preserves input order).
    assert ids_in_order == sorted(ids_in_order, key=lambda doc_id: doc_id != "docA")
    assert ids_in_order.count("docB") == 1


def test_empty_input_returns_empty_list():
    assert chunk_documents([], chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP) == []
