# -*- coding: utf-8 -*-
"""Utilities for local LangChain and ChromaDB workflows.

Model access goes through two factory functions, ``get_llm()`` and
``get_embeddings()``, each dispatching on an environment-selected backend:

- ``ollama`` (default) — the original local Ollama clients; behavior is
  unchanged when no ``RAG_*`` environment variables are set.
- ``hf`` — Hugging Face ``transformers``/``sentence-transformers`` via
  ``langchain-huggingface`` (imported lazily, so the package is only required
  when the backend is actually selected). Adopted for GPU workstations where
  Ollama is unavailable (NVIDIA driver 470 predates current Ollama's CUDA-12
  requirement — see evidence/workstation-c0c1-20260711/).

Backend selection: ``RAG_MODEL_BACKEND`` sets both components;
``RAG_LLM_BACKEND`` / ``RAG_EMBEDDING_BACKEND`` override per component (e.g.,
retrieval-only runs on a non-Ollama machine set ``RAG_EMBEDDING_BACKEND=hf``
and leave the LLM default — the LLM client is constructed but never invoked).

The ``hf`` backend has deliberately NO default model ids: verify the model
card (license, size/VRAM) before the first download, then export
``RAG_HF_LLM_MODEL`` / ``RAG_HF_EMBEDDING_MODEL``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import chromadb
from chromadb.errors import NotFoundError
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

OLLAMA_BASE_URL = "http://127.0.0.1:11434"
OLLAMA_CLIENT_KWARGS = {"trust_env": False}
LLM_MODEL_NAME = "gemma4:e4b"
EMBEDDING_MODEL_NAME = "nomic-embed-text"

MODEL_BACKENDS = {"ollama", "hf"}
MODEL_BACKEND_ENV = "RAG_MODEL_BACKEND"
LLM_BACKEND_ENV = "RAG_LLM_BACKEND"
EMBEDDING_BACKEND_ENV = "RAG_EMBEDDING_BACKEND"
HF_LLM_MODEL_ENV = "RAG_HF_LLM_MODEL"
HF_EMBEDDING_MODEL_ENV = "RAG_HF_EMBEDDING_MODEL"
HF_DEVICE_ENV = "RAG_HF_DEVICE"
HF_MAX_NEW_TOKENS_ENV = "RAG_HF_MAX_NEW_TOKENS"
HF_NORMALIZE_EMBEDDINGS_ENV = "RAG_HF_NORMALIZE_EMBEDDINGS"
_TRUTHY = {"1", "true", "yes", "on"}


def _resolve_backend(component_env: str) -> str:
    """Resolve a component's backend from env (component overrides shared)."""
    backend = (
        os.environ.get(component_env, "").strip()
        or os.environ.get(MODEL_BACKEND_ENV, "").strip()
        or "ollama"
    ).lower()
    if backend not in MODEL_BACKENDS:
        raise ValueError(
            f"Unknown model backend {backend!r} (from {component_env} or "
            f"{MODEL_BACKEND_ENV}); expected one of {sorted(MODEL_BACKENDS)}."
        )
    return backend


def get_llm_backend() -> str:
    """Return the active LLM backend name."""
    return _resolve_backend(LLM_BACKEND_ENV)


def get_embedding_backend() -> str:
    """Return the active embedding backend name."""
    return _resolve_backend(EMBEDDING_BACKEND_ENV)


def _require_hf_model_id(env_name: str) -> str:
    """Return the HF model id from env; fail loudly when unset."""
    model_id = os.environ.get(env_name, "").strip()
    if not model_id:
        raise ValueError(
            f"{env_name} must name a Hugging Face model id when the 'hf' "
            "backend is selected. There is deliberately no default: verify "
            "the model card (license, size/VRAM) first, then export the id."
        )
    return model_id


def _hf_device() -> str:
    return os.environ.get(HF_DEVICE_ENV, "").strip()


def _get_hf_llm() -> Any:
    """Build a chat model on the Hugging Face transformers backend."""
    # Validate configuration before the heavy import: a missing model id is a
    # config error and must surface as such even where the package is absent.
    model_id = _require_hf_model_id(HF_LLM_MODEL_ENV)

    from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline

    max_new_tokens = int(os.environ.get(HF_MAX_NEW_TOKENS_ENV, "512"))
    pipeline_kwargs: dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "do_sample": False,
    }
    device = _hf_device()
    from_model_id_kwargs: dict[str, Any] = {
        "model_id": model_id,
        "task": "text-generation",
        "pipeline_kwargs": pipeline_kwargs,
    }
    if device:
        from_model_id_kwargs["device"] = device
    else:
        from_model_id_kwargs["device_map"] = "auto"
    llm = HuggingFacePipeline.from_model_id(**from_model_id_kwargs)
    return ChatHuggingFace(llm=llm)


def _get_hf_embeddings() -> Any:
    """Build an embedding client on the sentence-transformers backend."""
    model_id = _require_hf_model_id(HF_EMBEDDING_MODEL_ENV)

    from langchain_huggingface import HuggingFaceEmbeddings

    model_kwargs: dict[str, Any] = {}
    device = _hf_device()
    if device:
        model_kwargs["device"] = device
    normalize = (
        os.environ.get(HF_NORMALIZE_EMBEDDINGS_ENV, "").strip().lower() in _TRUTHY
    )
    return HuggingFaceEmbeddings(
        model_name=model_id,
        model_kwargs=model_kwargs,
        encode_kwargs={"normalize_embeddings": normalize},
    )


def get_llm() -> Any:
    """Return the chat model for the active LLM backend."""
    if get_llm_backend() == "hf":
        return _get_hf_llm()
    return ChatOllama(
        model=LLM_MODEL_NAME,
        base_url=OLLAMA_BASE_URL,
        client_kwargs=OLLAMA_CLIENT_KWARGS,
        temperature=0,
    )


def get_embeddings() -> Any:
    """Return the embedding client for the active embedding backend."""
    if get_embedding_backend() == "hf":
        return _get_hf_embeddings()
    return OllamaEmbeddings(
        model=EMBEDDING_MODEL_NAME,
        base_url=OLLAMA_BASE_URL,
        client_kwargs=OLLAMA_CLIENT_KWARGS,
    )


def get_active_model_config() -> dict[str, Any]:
    """Describe the active backends/models without constructing clients."""
    llm_backend = get_llm_backend()
    embedding_backend = get_embedding_backend()
    return {
        "llm_backend": llm_backend,
        "llm_model": (
            LLM_MODEL_NAME
            if llm_backend == "ollama"
            else os.environ.get(HF_LLM_MODEL_ENV, "").strip() or None
        ),
        "embedding_backend": embedding_backend,
        "embedding_model": (
            EMBEDDING_MODEL_NAME
            if embedding_backend == "ollama"
            else os.environ.get(HF_EMBEDDING_MODEL_ENV, "").strip() or None
        ),
    }


def load_knowledge_base(json_path: str) -> list[Document]:
    """Load documents from a JSON knowledge base file."""
    path = Path(json_path).expanduser()
    with path.open("r", encoding="utf-8") as file:
        records = json.load(file)

    if not isinstance(records, list):
        raise ValueError("Knowledge base JSON must be a list of objects.")

    documents: list[Document] = []
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("Each knowledge base entry must be a JSON object.")

        document_id = str(record["id"])
        title = str(record["title"])
        content = str(record["content"])

        documents.append(
            Document(
                id=document_id,
                page_content=content,
                metadata={"id": document_id, "title": title},
            )
        )

    return documents


def create_chroma_collection(
    documents: list[Document],
    collection_name: str,
    persist_directory: str,
) -> Chroma:
    """Recreate a Chroma collection from the provided documents."""
    persist_path = Path(persist_directory).expanduser()
    persist_path.mkdir(parents=True, exist_ok=True)

    client = chromadb.PersistentClient(path=str(persist_path))
    try:
        client.delete_collection(name=collection_name)
    except NotFoundError:
        pass

    embeddings = get_embeddings()
    vectorstore = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=str(persist_path),
    )

    if not documents:
        return vectorstore

    ids = [
        f"{document.metadata.get('id', document.id or 'document')}_{index}"
        for index, document in enumerate(documents)
    ]
    batch_size = len(documents)
    if hasattr(client, "get_max_batch_size"):
        try:
            batch_size = max(1, int(client.get_max_batch_size()))
        except Exception:
            batch_size = len(documents)

    for start in range(0, len(documents), batch_size):
        end = start + batch_size
        vectorstore.add_documents(
            documents=documents[start:end],
            ids=ids[start:end],
        )
    return vectorstore


def chunk_documents(
    documents: list[Document],
    chunk_size: int = 512,
    chunk_overlap: int = 64,
) -> list[Document]:
    """Split documents into overlapping chunks while preserving metadata."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return splitter.split_documents(documents)
