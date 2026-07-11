# -*- coding: utf-8 -*-
"""Stubs for heavy optional dependencies (vector store / model runtimes).

The unit tests and the CI deterministic checks exercise model-free logic
only, but ``src/utils.py`` and ``src/rag_pipelines.py`` import their heavy
dependencies (chromadb, langchain-chroma, langchain-ollama,
sentence-transformers) at module import time. On machines without the full
lockfile environment - notably CI - this module installs lightweight
stand-ins for exactly the packages that are missing. On a full environment
the real packages win and nothing is stubbed.

Every stub that could trigger model inference or vector-store I/O raises
``RuntimeError`` when used, so a test that accidentally reaches a model
path fails loudly instead of silently calling a local LLM.
"""

from __future__ import annotations

import importlib.util
import sys
import types


def _is_missing(module_name: str) -> bool:
    if module_name in sys.modules:
        return False
    return importlib.util.find_spec(module_name) is None


def _new_module(name: str) -> types.ModuleType:
    module = types.ModuleType(name)
    module.__stub__ = True  # type: ignore[attr-defined]
    return module


def _forbidden(action: str) -> RuntimeError:
    return RuntimeError(
        f"dependency stub: {action} is not available in model-free tests/CI. "
        "Install the full environment from requirements-lock-py311.txt to use it."
    )


def install_missing_dependency_stubs() -> list[str]:
    """Install stub modules for missing heavy deps; return the stubbed names."""
    stubbed: list[str] = []

    if _is_missing("chromadb"):
        chromadb_module = _new_module("chromadb")
        errors_module = _new_module("chromadb.errors")

        class NotFoundError(Exception):
            """Mirror of chromadb.errors.NotFoundError."""

        class PersistentClient:
            def __init__(self, *args, **kwargs) -> None:
                raise _forbidden("chromadb.PersistentClient (vector store)")

        errors_module.NotFoundError = NotFoundError  # type: ignore[attr-defined]
        chromadb_module.errors = errors_module  # type: ignore[attr-defined]
        chromadb_module.PersistentClient = PersistentClient  # type: ignore[attr-defined]
        sys.modules["chromadb"] = chromadb_module
        sys.modules["chromadb.errors"] = errors_module
        stubbed.append("chromadb")

    if _is_missing("langchain_chroma"):
        langchain_chroma_module = _new_module("langchain_chroma")

        class Chroma:
            def __init__(self, *args, **kwargs) -> None:
                raise _forbidden("langchain_chroma.Chroma (vector store)")

        Chroma.__module__ = "langchain_chroma"
        langchain_chroma_module.Chroma = Chroma  # type: ignore[attr-defined]
        sys.modules["langchain_chroma"] = langchain_chroma_module
        stubbed.append("langchain_chroma")

    if _is_missing("langchain_ollama"):
        langchain_ollama_module = _new_module("langchain_ollama")

        class ChatOllama:
            """Constructible placeholder; any model call raises."""

            def __init__(self, **kwargs) -> None:
                self.kwargs = kwargs

            def invoke(self, *args, **kwargs):
                raise _forbidden("ChatOllama.invoke (LLM call)")

        class OllamaEmbeddings:
            """Constructible placeholder; any embedding call raises."""

            def __init__(self, **kwargs) -> None:
                self.kwargs = kwargs

            def embed_documents(self, *args, **kwargs):
                raise _forbidden("OllamaEmbeddings.embed_documents (embedding call)")

            def embed_query(self, *args, **kwargs):
                raise _forbidden("OllamaEmbeddings.embed_query (embedding call)")

        ChatOllama.__module__ = "langchain_ollama"
        OllamaEmbeddings.__module__ = "langchain_ollama"
        langchain_ollama_module.ChatOllama = ChatOllama  # type: ignore[attr-defined]
        langchain_ollama_module.OllamaEmbeddings = OllamaEmbeddings  # type: ignore[attr-defined]
        sys.modules["langchain_ollama"] = langchain_ollama_module
        stubbed.append("langchain_ollama")

    if _is_missing("sentence_transformers"):
        sentence_transformers_module = _new_module("sentence_transformers")

        class CrossEncoder:
            def __init__(self, *args, **kwargs) -> None:
                raise _forbidden("sentence_transformers.CrossEncoder (model download)")

        CrossEncoder.__module__ = "sentence_transformers"
        sentence_transformers_module.CrossEncoder = CrossEncoder  # type: ignore[attr-defined]
        sys.modules["sentence_transformers"] = sentence_transformers_module
        stubbed.append("sentence_transformers")

    return stubbed
