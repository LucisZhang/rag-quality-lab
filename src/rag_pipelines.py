# -*- coding: utf-8 -*-
"""RAG pipelines built on top of the local Ollama and Chroma utilities."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

try:
    from .utils import (
        chunk_documents,
        create_chroma_collection,
        get_llm,
        load_knowledge_base,
    )
except ImportError:
    try:
        from src.utils import (  # type: ignore
            chunk_documents,
            create_chroma_collection,
            get_llm,
            load_knowledge_base,
        )
    except ImportError:
        from utils import (  # type: ignore
            chunk_documents,
            create_chroma_collection,
            get_llm,
            load_knowledge_base,
        )

ANSWER_PROMPT = PromptTemplate.from_template(
    """You are a helpful assistant. Answer the question based ONLY on the provided context.
If the context doesn't contain enough information, say "I don't have enough information to answer this."

Context:
{context}

Question: {question}

Answer:"""
)


def _tokenize(text: str) -> list[str]:
    """Lowercase whitespace tokenization for BM25 indexing and search."""
    return text.lower().split()


def _message_content_to_text(content: Any) -> str:
    """Normalize LangChain message content to a plain string."""
    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
                continue

            if isinstance(item, dict):
                text = item.get("text")
                if text:
                    parts.append(str(text))

        return "\n".join(parts).strip()

    return str(content).strip()


def _unique_preserve_order(values: list[str]) -> list[str]:
    """Deduplicate a list while preserving the first-seen order."""
    seen: set[str] = set()
    unique_values: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        unique_values.append(value)
    return unique_values


class BaseRAGPipeline(ABC):
    """Shared setup and generation logic for RAG pipelines."""

    pipeline_name = "base_rag"

    def __init__(
        self,
        knowledge_base_path: str,
        collection_name: str,
        persist_dir: str,
    ) -> None:
        self.knowledge_base_path = str(Path(knowledge_base_path).expanduser())
        self.collection_name = collection_name
        self.persist_dir = str(Path(persist_dir).expanduser())
        self.chunk_size = 512
        self.chunk_overlap = 64
        self.llm = get_llm()

        docs = load_knowledge_base(self.knowledge_base_path)
        print(f"✅ Loaded {len(docs)} documents")
        self.documents = docs

        chunks = chunk_documents(
            self.documents,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )
        print(f"✅ Chunked into {len(chunks)} pieces")
        self.chunks = chunks

        print("⏳ Building vector index (this takes a few minutes)...")
        self.vectorstore = create_chroma_collection(
            documents=self.chunks,
            collection_name=self.collection_name,
            persist_directory=self.persist_dir,
        )
        print("✅ Vector index ready")

    @abstractmethod
    def retrieve(self, query: str, k: int = 4) -> list[Document]:
        """Return the most relevant documents for the query."""

    def generate(self, query: str, context_docs: list[Document]) -> str:
        """Generate an answer grounded only in the retrieved context."""
        context = self._format_context(context_docs)
        prompt = ANSWER_PROMPT.format(context=context, question=query)
        response = self.llm.invoke(prompt)
        return _message_content_to_text(response.content)

    def query(self, query: str, k: int = 4) -> dict[str, Any]:
        """Run retrieve-then-generate and return a structured response."""
        print(f"⏳ Querying: {query[:50]}...")
        retrieved_docs = self.retrieve(query, k=k)
        answer = self.generate(query, retrieved_docs)
        retrieved_doc_ids = _unique_preserve_order(
            [
                str(doc.metadata.get("id", doc.id or "")).strip()
                for doc in retrieved_docs
                if str(doc.metadata.get("id", doc.id or "")).strip()
            ]
        )
        print(f"✅ Answer generated ({len(answer)} chars)")
        return {
            "question": query,
            "answer": answer,
            "contexts": [doc.page_content for doc in retrieved_docs],
            "retrieved_doc_ids": retrieved_doc_ids,
        }

    def get_pipeline_info(self) -> dict[str, Any]:
        """Describe the pipeline and its runtime configuration."""
        return {
            "name": self.pipeline_name,
            "description": self._pipeline_description(),
            "knowledge_base_path": self.knowledge_base_path,
            "collection_name": self.collection_name,
            "persist_dir": self.persist_dir,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
        }

    def _format_context(self, context_docs: list[Document]) -> str:
        if not context_docs:
            return ""

        sections: list[str] = []
        for index, doc in enumerate(context_docs, start=1):
            title = doc.metadata.get("title", f"Document {index}")
            sections.append(f"[{index}] {title}\n{doc.page_content}")
        return "\n\n".join(sections)

    @abstractmethod
    def _pipeline_description(self) -> str:
        """Return a concise description of the retrieval strategy."""


class NaiveVectorRAG(BaseRAGPipeline):
    """Dense retrieval with direct generation over the top-k results."""

    pipeline_name = "NaiveVectorRAG"

    def __init__(
        self,
        knowledge_base_path: str,
        collection_name: str = "naive_rag",
        persist_dir: str = "./chroma_db",
    ) -> None:
        super().__init__(
            knowledge_base_path=knowledge_base_path,
            collection_name=collection_name,
            persist_dir=persist_dir,
        )

    def retrieve(self, query: str, k: int = 4) -> list[Document]:
        """Retrieve the top-k documents using Chroma similarity search."""
        return self.vectorstore.similarity_search(query, k=k)

    def _pipeline_description(self) -> str:
        return "Naive dense retrieval pipeline using Chroma similarity search."


class HybridRerankRAG(BaseRAGPipeline):
    """Hybrid retrieval with BM25 and dense search followed by re-ranking."""

    pipeline_name = "HybridRerankRAG"

    def __init__(
        self,
        knowledge_base_path: str,
        collection_name: str = "hybrid_rag",
        persist_dir: str = "./chroma_db",
    ) -> None:
        super().__init__(
            knowledge_base_path=knowledge_base_path,
            collection_name=collection_name,
            persist_dir=persist_dir,
        )
        self.tokenized_chunks = [_tokenize(doc.page_content) for doc in self.chunks]
        self.bm25 = BM25Okapi(self.tokenized_chunks) if self.tokenized_chunks else None
        print("✅ BM25 index ready")
        self.reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        print("✅ Cross-encoder reranker loaded")

    def retrieve(self, query: str, k: int = 4) -> list[Document]:
        """Retrieve candidates from dense and lexical search, then rerank."""
        dense_docs = self.vectorstore.similarity_search(query, k=10)
        bm25_docs = (
            self.bm25.get_top_n(_tokenize(query), self.chunks, n=10)
            if self.bm25 is not None
            else []
        )

        merged_docs: list[Document] = []
        seen_contents: set[str] = set()
        for doc in dense_docs + bm25_docs:
            if doc.page_content in seen_contents:
                continue
            seen_contents.add(doc.page_content)
            merged_docs.append(doc)

        if not merged_docs:
            return []

        pairs = [(query, doc.page_content) for doc in merged_docs]
        scores = self.reranker.predict(pairs)
        ranked_docs = sorted(
            zip(merged_docs, scores, strict=True),
            key=lambda item: float(item[1]),
            reverse=True,
        )
        return [doc for doc, _score in ranked_docs[:k]]

    def _pipeline_description(self) -> str:
        return (
            "Hybrid retrieval pipeline combining Chroma dense search, BM25 lexical "
            "search, and CrossEncoder re-ranking."
        )


if __name__ == "__main__":
    knowledge_base = (
        Path(__file__).resolve().parent.parent / "data" / "knowledge_base_v2.json"
    )

    naive_pipeline = NaiveVectorRAG(str(knowledge_base))
    hybrid_pipeline = HybridRerankRAG(str(knowledge_base))

    print("NaiveVectorRAG result:")
    print(naive_pipeline.query("What is RAG?"))
    print()

    print("HybridRerankRAG result:")
    print(hybrid_pipeline.query("What is RAG?"))
