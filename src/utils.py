# -*- coding: utf-8 -*-
"""Utilities for local Ollama-backed LangChain and ChromaDB workflows."""

from __future__ import annotations

import json
from pathlib import Path

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


def get_llm() -> ChatOllama:
    """Return a ChatOllama client configured for the local Ollama server."""
    return ChatOllama(
        model=LLM_MODEL_NAME,
        base_url=OLLAMA_BASE_URL,
        client_kwargs=OLLAMA_CLIENT_KWARGS,
        temperature=0,
    )


def get_embeddings() -> OllamaEmbeddings:
    """Return an Ollama embedding client configured for the local Ollama server."""
    return OllamaEmbeddings(
        model=EMBEDDING_MODEL_NAME,
        base_url=OLLAMA_BASE_URL,
        client_kwargs=OLLAMA_CLIENT_KWARGS,
    )


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
