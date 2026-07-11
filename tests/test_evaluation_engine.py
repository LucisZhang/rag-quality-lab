# -*- coding: utf-8 -*-
"""Unit tests for the model-free parts of src/evaluation_engine.py.

The LLM-judged path is exercised with a scripted fake LLM (the A6 rule:
no model inference in tests/CI - LLM calls are mocked).
"""

from __future__ import annotations

import json
import math
from types import SimpleNamespace

import pytest

from src.evaluation_engine import RAGEvaluator


class ScriptedLLM:
    """Returns a canned scoring reply and records every prompt."""

    def __init__(self, reply: str = "0.8") -> None:
        self.reply = reply
        self.prompts: list[str] = []

    def invoke(self, prompt: str) -> SimpleNamespace:
        self.prompts.append(prompt)
        return SimpleNamespace(content=self.reply)


class ScriptedPipeline:
    """Returns canned query results keyed by question."""

    def __init__(self, results_by_question: dict[str, dict]) -> None:
        self._results = results_by_question

    def query(self, question: str) -> dict:
        return self._results[question]


def make_evaluator(reply: str = "0.8") -> tuple[RAGEvaluator, ScriptedLLM]:
    llm = ScriptedLLM(reply=reply)
    evaluator = RAGEvaluator(
        llm=llm,
        embeddings=object(),
        evaluation_backend="fallback",
    )
    return evaluator, llm


def test_rejects_unknown_evaluation_backend():
    with pytest.raises(ValueError, match="evaluation_backend"):
        RAGEvaluator(llm=object(), embeddings=object(), evaluation_backend="bogus")


def test_compute_retrieval_metrics_partial_match():
    evaluator, _ = make_evaluator()
    metrics = evaluator._compute_retrieval_metrics(
        retrieved_doc_ids=["d1", "d3", "d4", "d5"],
        relevant_doc_ids=["d1", "d2"],
    )
    assert metrics["retrieval_precision"] == pytest.approx(1 / 4)
    assert metrics["retrieval_recall"] == pytest.approx(1 / 2)
    assert metrics["retrieval_hit"] == 1.0


def test_compute_retrieval_metrics_no_retrieval_and_no_relevant():
    evaluator, _ = make_evaluator()

    nothing_retrieved = evaluator._compute_retrieval_metrics([], ["d1"])
    assert nothing_retrieved == {
        "retrieval_precision": 0.0,
        "retrieval_recall": 0.0,
        "retrieval_hit": 0.0,
    }

    # No relevant docs annotated: recall is vacuously perfect.
    no_relevant = evaluator._compute_retrieval_metrics(["d1"], [])
    assert no_relevant["retrieval_recall"] == 1.0
    assert no_relevant["retrieval_precision"] == 0.0
    assert no_relevant["retrieval_hit"] == 0.0


def test_compute_retrieval_metrics_deduplicates_before_scoring():
    evaluator, _ = make_evaluator()
    metrics = evaluator._compute_retrieval_metrics(
        retrieved_doc_ids=["d1", "d1", "d2"],
        relevant_doc_ids=["d1", "d1"],
    )
    assert metrics["retrieval_precision"] == pytest.approx(1 / 2)
    assert metrics["retrieval_recall"] == 1.0


def test_normalize_doc_ids_variants():
    evaluator, _ = make_evaluator()
    assert evaluator._normalize_doc_ids(None) == []
    assert evaluator._normalize_doc_ids("  d1  ") == ["d1"]
    assert evaluator._normalize_doc_ids("") == []
    assert evaluator._normalize_doc_ids(["d2", "d1", "d2", " ", 3]) == ["d2", "d1", "3"]
    assert evaluator._normalize_doc_ids(7) == ["7"]


def test_normalize_contexts_variants():
    evaluator, _ = make_evaluator()
    assert evaluator._normalize_contexts(None) == []
    assert evaluator._normalize_contexts(" a context ") == ["a context"]
    assert evaluator._normalize_contexts(["one", {"page_content": "two"}, ""]) == ["one", "two"]
    assert evaluator._normalize_contexts({"text": "three"}) == ["three"]


def test_parse_score_extracts_and_clamps():
    evaluator, _ = make_evaluator()
    assert evaluator._parse_score("0.8") == pytest.approx(0.8)
    assert evaluator._parse_score("Score: 0.75 (confident)") == pytest.approx(0.75)
    assert evaluator._parse_score("2") == 1.0
    assert evaluator._parse_score("-0.5") == 0.0
    assert math.isnan(evaluator._parse_score("no digits here"))


def test_load_eval_dataset_validates_and_coerces(tmp_path):
    evaluator, _ = make_evaluator()

    valid_path = tmp_path / "eval.json"
    valid_path.write_text(
        json.dumps(
            [
                {
                    "question": "Q1?",
                    "ground_truth": "A1",
                    "relevant_doc_ids": [1, "d2"],
                    "extra_field": "ignored",
                }
            ]
        ),
        encoding="utf-8",
    )
    dataset = evaluator.load_eval_dataset(str(valid_path))
    assert dataset == [
        {"question": "Q1?", "ground_truth": "A1", "relevant_doc_ids": ["1", "d2"]}
    ]

    missing_key_path = tmp_path / "missing.json"
    missing_key_path.write_text(json.dumps([{"question": "Q1?"}]), encoding="utf-8")
    with pytest.raises(ValueError, match="missing keys"):
        evaluator.load_eval_dataset(str(missing_key_path))

    not_list_path = tmp_path / "notlist.json"
    not_list_path.write_text(json.dumps({"question": "Q1?"}), encoding="utf-8")
    with pytest.raises(ValueError, match="must be a list"):
        evaluator.load_eval_dataset(str(not_list_path))

    bad_ids_path = tmp_path / "badids.json"
    bad_ids_path.write_text(
        json.dumps(
            [{"question": "Q1?", "ground_truth": "A1", "relevant_doc_ids": "d1"}]
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="relevant_doc_ids"):
        evaluator.load_eval_dataset(str(bad_ids_path))


def test_run_pipeline_evaluation_fallback_scores_with_mocked_llm():
    evaluator, llm = make_evaluator(reply="0.8")
    dataset = [
        {"question": "Q1?", "ground_truth": "GT1", "relevant_doc_ids": ["d1"]},
        {"question": "Q2?", "ground_truth": "GT2", "relevant_doc_ids": ["d9"]},
    ]
    pipeline = ScriptedPipeline(
        {
            "Q1?": {
                "answer": "Answer one",
                "contexts": ["ctx a", "ctx b"],
                "retrieved_doc_ids": ["d1", "d2"],
            },
            "Q2?": {
                "answer": "Answer two",
                "contexts": ["ctx c"],
                "retrieved_doc_ids": ["d3"],
            },
        }
    )

    results_df = evaluator.run_pipeline_evaluation(pipeline, dataset)

    assert list(results_df.columns) == RAGEvaluator.RESULT_COLUMNS
    assert len(results_df) == 2
    # 5 LLM-judged metrics x 2 questions, every score from the mocked LLM.
    assert len(llm.prompts) == 10
    for metric in RAGEvaluator.LLM_METRIC_COLUMNS:
        assert results_df[metric].tolist() == pytest.approx([0.8, 0.8])

    # Deterministic retrieval diagnostics computed from doc ids.
    assert results_df["retrieval_hit"].tolist() == [1.0, 0.0]
    assert results_df["retrieval_precision"].tolist() == pytest.approx([0.5, 0.0])
    assert results_df["retrieval_recall"].tolist() == pytest.approx([1.0, 0.0])

    # List columns are serialized as JSON strings for stable CSV round-trips.
    assert json.loads(results_df.loc[0, "contexts"]) == ["ctx a", "ctx b"]
    assert json.loads(results_df.loc[0, "retrieved_doc_ids"]) == ["d1", "d2"]
    assert json.loads(results_df.loc[1, "relevant_doc_ids"]) == ["d9"]


def test_run_pipeline_evaluation_empty_dataset_returns_empty_frame():
    evaluator, llm = make_evaluator()
    results_df = evaluator.run_pipeline_evaluation(ScriptedPipeline({}), [])
    assert list(results_df.columns) == RAGEvaluator.RESULT_COLUMNS
    assert results_df.empty
    assert llm.prompts == []


def test_run_pipeline_evaluation_rejects_non_dict_query_result():
    evaluator, _ = make_evaluator()

    class BrokenPipeline:
        def query(self, question: str):
            return "not a dict"

    with pytest.raises(TypeError, match="must return a dict"):
        evaluator.run_pipeline_evaluation(
            BrokenPipeline(),
            [{"question": "Q?", "ground_truth": "GT", "relevant_doc_ids": []}],
        )


def test_unparseable_llm_reply_yields_nan_and_none_summary_stat():
    evaluator, _ = make_evaluator(reply="I cannot rate this")
    dataset = [{"question": "Q1?", "ground_truth": "GT", "relevant_doc_ids": []}]
    pipeline = ScriptedPipeline(
        {"Q1?": {"answer": "A", "contexts": ["c"], "retrieved_doc_ids": []}}
    )

    results_df = evaluator.run_pipeline_evaluation(pipeline, dataset)
    assert math.isnan(results_df.loc[0, "faithfulness"])

    summary = evaluator.compute_summary(results_df)
    assert summary["faithfulness"]["mean"] is None
    # Retrieval metrics stay numeric even when the judge reply is unusable.
    assert summary["retrieval_recall"]["mean"] == 1.0


def test_compute_summary_statistics():
    evaluator, llm = make_evaluator(reply="0.5")
    dataset = [
        {"question": "Q1?", "ground_truth": "GT1", "relevant_doc_ids": ["d1"]},
        {"question": "Q2?", "ground_truth": "GT2", "relevant_doc_ids": ["d1"]},
    ]
    pipeline = ScriptedPipeline(
        {
            "Q1?": {"answer": "A1", "contexts": ["c"], "retrieved_doc_ids": ["d1"]},
            "Q2?": {"answer": "A2", "contexts": ["c"], "retrieved_doc_ids": []},
        }
    )
    summary = evaluator.compute_summary(
        evaluator.run_pipeline_evaluation(pipeline, dataset)
    )

    assert summary["faithfulness"]["mean"] == pytest.approx(0.5)
    assert summary["retrieval_hit"]["mean"] == pytest.approx(0.5)
    assert summary["retrieval_hit"]["min"] == 0.0
    assert summary["retrieval_hit"]["max"] == 1.0
    assert summary["retrieval_hit"]["std"] == pytest.approx(0.7071, abs=1e-4)
