# -*- coding: utf-8 -*-
"""Unit tests for the ollama|hf model-backend seam in src/utils.py.

No model is ever constructed for real: the ollama path yields client objects
that are never invoked (or CI stubs), and the hf path is exercised against a
fake ``langchain_huggingface`` module injected into ``sys.modules`` — the real
package is imported lazily and only when the hf backend is selected.
"""

from __future__ import annotations

import sys
import types

import pytest

from src import utils

BACKEND_ENVS = (
    utils.MODEL_BACKEND_ENV,
    utils.LLM_BACKEND_ENV,
    utils.EMBEDDING_BACKEND_ENV,
    utils.HF_LLM_MODEL_ENV,
    utils.HF_EMBEDDING_MODEL_ENV,
    utils.HF_DEVICE_ENV,
    utils.HF_MAX_NEW_TOKENS_ENV,
    utils.HF_NORMALIZE_EMBEDDINGS_ENV,
)


@pytest.fixture(autouse=True)
def clean_backend_env(monkeypatch: pytest.MonkeyPatch):
    for name in BACKEND_ENVS:
        monkeypatch.delenv(name, raising=False)


class FakeHuggingFacePipeline:
    """Records from_model_id kwargs without touching transformers."""

    last_kwargs: dict | None = None

    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs

    @classmethod
    def from_model_id(cls, **kwargs):
        cls.last_kwargs = kwargs
        return cls(**kwargs)


class FakeChatHuggingFace:
    def __init__(self, llm) -> None:
        self.llm = llm


class FakeHuggingFaceEmbeddings:
    def __init__(self, model_name: str, model_kwargs=None, encode_kwargs=None) -> None:
        self.model_name = model_name
        self.model_kwargs = model_kwargs
        self.encode_kwargs = encode_kwargs


@pytest.fixture()
def fake_langchain_huggingface(monkeypatch: pytest.MonkeyPatch):
    module = types.ModuleType("langchain_huggingface")
    module.ChatHuggingFace = FakeChatHuggingFace
    module.HuggingFacePipeline = FakeHuggingFacePipeline
    module.HuggingFaceEmbeddings = FakeHuggingFaceEmbeddings
    monkeypatch.setitem(sys.modules, "langchain_huggingface", module)
    FakeHuggingFacePipeline.last_kwargs = None
    return module


def test_default_backend_is_ollama_and_behavior_unchanged():
    assert utils.get_llm_backend() == "ollama"
    assert utils.get_embedding_backend() == "ollama"
    assert type(utils.get_llm()).__name__ == "ChatOllama"
    assert type(utils.get_embeddings()).__name__ == "OllamaEmbeddings"


def test_unknown_backend_raises(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv(utils.MODEL_BACKEND_ENV, "vllm")
    with pytest.raises(ValueError, match="Unknown model backend"):
        utils.get_llm_backend()


def test_hf_backend_requires_model_ids(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv(utils.MODEL_BACKEND_ENV, "hf")
    with pytest.raises(ValueError, match=utils.HF_LLM_MODEL_ENV):
        utils.get_llm()
    with pytest.raises(ValueError, match=utils.HF_EMBEDDING_MODEL_ENV):
        utils.get_embeddings()


def test_hf_llm_constructed_from_env(
    monkeypatch: pytest.MonkeyPatch, fake_langchain_huggingface
):
    monkeypatch.setenv(utils.MODEL_BACKEND_ENV, "hf")
    monkeypatch.setenv(utils.HF_LLM_MODEL_ENV, "example-org/example-4b-it")
    monkeypatch.setenv(utils.HF_MAX_NEW_TOKENS_ENV, "128")

    llm = utils.get_llm()

    assert isinstance(llm, FakeChatHuggingFace)
    kwargs = FakeHuggingFacePipeline.last_kwargs
    assert kwargs is not None
    assert kwargs["model_id"] == "example-org/example-4b-it"
    assert kwargs["task"] == "text-generation"
    assert kwargs["pipeline_kwargs"] == {"max_new_tokens": 128, "do_sample": False}
    assert kwargs["device_map"] == "auto"
    assert "device" not in kwargs


def test_hf_llm_device_env_overrides_device_map(
    monkeypatch: pytest.MonkeyPatch, fake_langchain_huggingface
):
    monkeypatch.setenv(utils.MODEL_BACKEND_ENV, "hf")
    monkeypatch.setenv(utils.HF_LLM_MODEL_ENV, "example-org/example-4b-it")
    monkeypatch.setenv(utils.HF_DEVICE_ENV, "cuda:0")

    utils.get_llm()

    kwargs = FakeHuggingFacePipeline.last_kwargs
    assert kwargs is not None
    assert kwargs["device"] == "cuda:0"
    assert "device_map" not in kwargs


def test_hf_embeddings_constructed_from_env(
    monkeypatch: pytest.MonkeyPatch, fake_langchain_huggingface
):
    monkeypatch.setenv(utils.MODEL_BACKEND_ENV, "hf")
    monkeypatch.setenv(utils.HF_EMBEDDING_MODEL_ENV, "BAAI/bge-m3")
    monkeypatch.setenv(utils.HF_NORMALIZE_EMBEDDINGS_ENV, "1")
    monkeypatch.setenv(utils.HF_DEVICE_ENV, "cuda:0")

    embeddings = utils.get_embeddings()

    assert isinstance(embeddings, FakeHuggingFaceEmbeddings)
    assert embeddings.model_name == "BAAI/bge-m3"
    assert embeddings.model_kwargs == {"device": "cuda:0"}
    assert embeddings.encode_kwargs == {"normalize_embeddings": True}


def test_per_component_override_allows_mixed_backends(
    monkeypatch: pytest.MonkeyPatch, fake_langchain_huggingface
):
    # retrieval-only pattern: hf embeddings, default (ollama) LLM object
    monkeypatch.setenv(utils.EMBEDDING_BACKEND_ENV, "hf")
    monkeypatch.setenv(utils.HF_EMBEDDING_MODEL_ENV, "BAAI/bge-m3")

    assert utils.get_llm_backend() == "ollama"
    assert utils.get_embedding_backend() == "hf"
    assert type(utils.get_llm()).__name__ == "ChatOllama"
    assert isinstance(utils.get_embeddings(), FakeHuggingFaceEmbeddings)


def test_active_model_config_reports_without_constructing(
    monkeypatch: pytest.MonkeyPatch,
):
    config = utils.get_active_model_config()
    assert config == {
        "llm_backend": "ollama",
        "llm_model": utils.LLM_MODEL_NAME,
        "embedding_backend": "ollama",
        "embedding_model": utils.EMBEDDING_MODEL_NAME,
    }

    monkeypatch.setenv(utils.MODEL_BACKEND_ENV, "hf")
    config = utils.get_active_model_config()
    # unset hf model ids report as None instead of raising: reporting must
    # stay side-effect-free for verify_a3 deterministic mode
    assert config["llm_backend"] == "hf"
    assert config["llm_model"] is None
    assert config["embedding_model"] is None
