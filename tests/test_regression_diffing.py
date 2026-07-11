# -*- coding: utf-8 -*-
"""Unit tests for the model-free regression diffing logic.

RegressionTester.__init__ builds a real evaluator (and therefore LLM
clients), so instances here are assembled via __new__ with only the
attributes the diffing helpers use. The end-to-end regression test run
uses a fake pipeline class and a scripted judge LLM - no model calls.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from src.evaluation_engine import RAGEvaluator
from src.regression_tester import RegressionTester

METRICS = list(RAGEvaluator.METRIC_COLUMNS)
LLM_METRICS = list(RAGEvaluator.LLM_METRIC_COLUMNS)


def make_tester(results_dir: Path | str = "./results") -> RegressionTester:
    tester = RegressionTester.__new__(RegressionTester)
    tester.eval_dataset_path = "data/eval_questions.json"
    tester.eval_dataset_display_path = "data/eval_questions.json"
    tester.results_dir = Path(results_dir)
    tester.evaluator = None  # the diffing helpers under test never touch it
    tester.metric_columns = list(METRICS)
    return tester


def make_row(
    question: str,
    scores: dict[str, float],
    retrieved: list[str] | None = None,
    relevant: list[str] | None = None,
) -> dict:
    row = {
        "question": question,
        "answer": f"answer to {question}",
        "contexts": "[]",
        "ground_truth": "gt",
        "retrieved_doc_ids": json.dumps(retrieved or []),
        "relevant_doc_ids": json.dumps(relevant or []),
    }
    for metric in METRICS:
        row[metric] = scores.get(metric, 0.5)
    return row


class TestClassifyStatus:
    def test_thresholds_are_strict(self):
        tester = make_tester()
        assert tester._classify_status({"m": -0.11}) == "degraded"
        assert tester._classify_status({"m": -0.1}) == "stable"
        assert tester._classify_status({"m": 0.1}) == "stable"
        assert tester._classify_status({"m": 0.11}) == "improved"
        assert tester._classify_status({"m": None}) == "stable"
        assert tester._classify_status({}) == "stable"

    def test_degradation_takes_precedence_over_improvement(self):
        tester = make_tester()
        assert tester._classify_status({"m1": -0.5, "m2": 0.5}) == "degraded"


class TestScalarHelpers:
    def test_compute_diff_none_propagation(self):
        tester = make_tester()
        assert tester._compute_diff(0.9, 0.3) == pytest.approx(-0.6)
        assert tester._compute_diff(None, 0.3) is None
        assert tester._compute_diff(0.9, None) is None

    def test_to_clean_float_rejects_nan_inf_and_non_numeric(self):
        tester = make_tester()
        assert tester._to_clean_float(0.5) == 0.5
        assert tester._to_clean_float("0.5") == 0.5
        assert tester._to_clean_float(float("nan")) is None
        assert tester._to_clean_float(float("inf")) is None
        assert tester._to_clean_float("not a number") is None
        assert tester._to_clean_float(None) is None

    def test_deserialize_string_list_variants(self):
        tester = make_tester()
        assert tester._deserialize_string_list('["d1", "d2"]') == ["d1", "d2"]
        assert tester._deserialize_string_list("plain-string") == ["plain-string"]
        assert tester._deserialize_string_list("42") == ["42"]
        assert tester._deserialize_string_list(["d1", 2]) == ["d1", "2"]
        assert tester._deserialize_string_list(None) == []
        assert tester._deserialize_string_list("  ") == []

    def test_slugify_and_collection_name(self):
        tester = make_tester()
        assert tester._slugify_name("Hybrid RAG (v2)!") == "Hybrid_RAG_v2"
        assert tester._slugify_name("///") == "rag"

        name = tester._build_collection_name(
            "a_very_long_pipeline_base_name_that_keeps_going",
            "v1",
            "run_20260711_123456_abcdef12",
        )
        assert "v1" in name
        assert name.startswith("a_very_long_pipeline_base_name")
        assert len(name) <= 30 + 1 + 2 + 1 + 18  # base + separators + tag + token


class TestPerQuestionDiff:
    def build_frames(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        baseline = pd.DataFrame(
            [
                make_row(
                    "Q-degraded",
                    {metric: 0.9 for metric in METRICS},
                    retrieved=["d1", "d2"],
                    relevant=["d1"],
                ),
                make_row("Q-improved", {metric: 0.4 for metric in METRICS}),
                make_row("Q-stable", {metric: 0.7 for metric in METRICS}),
            ]
        )
        updated = pd.DataFrame(
            [
                make_row(
                    "Q-degraded",
                    {**{metric: 0.9 for metric in METRICS}, "faithfulness": 0.3},
                    retrieved=["d2", "d3"],
                    relevant=["d1"],
                ),
                make_row(
                    "Q-improved",
                    {**{metric: 0.4 for metric in METRICS}, "answer_relevancy": 0.8},
                ),
                make_row("Q-stable", {metric: 0.75 for metric in METRICS}),
            ]
        )
        return baseline, updated

    def test_statuses_diffs_and_matched_doc_ids(self):
        tester = make_tester()
        baseline, updated = self.build_frames()
        diff = tester._build_per_question_diff(baseline, updated)

        assert [item["status"] for item in diff] == ["degraded", "improved", "stable"]

        degraded = diff[0]
        assert degraded["question"] == "Q-degraded"
        assert degraded["degraded_metrics"] == ["faithfulness"]
        assert degraded["diff"]["faithfulness"] == pytest.approx(-0.6)
        assert degraded["baseline_scores"]["faithfulness"] == pytest.approx(0.9)
        assert degraded["updated_scores"]["faithfulness"] == pytest.approx(0.3)
        # Retrieval bookkeeping: which retrieved ids hit the relevant set.
        assert degraded["relevant_doc_ids"] == ["d1"]
        assert degraded["baseline_matched_relevant_doc_ids"] == ["d1"]
        assert degraded["updated_matched_relevant_doc_ids"] == []

        improved = diff[1]
        assert improved["diff"]["answer_relevancy"] == pytest.approx(0.4)
        assert improved["degraded_metrics"] == []

        stable = diff[2]
        assert all(
            value == pytest.approx(0.05)
            for value in stable["diff"].values()
            if value is not None
        )

    def test_raises_on_row_count_mismatch(self):
        tester = make_tester()
        baseline, updated = self.build_frames()
        with pytest.raises(ValueError, match="different numbers of rows"):
            tester._build_per_question_diff(baseline, updated.iloc[:2])

    def test_raises_on_question_misalignment(self):
        tester = make_tester()
        baseline, updated = self.build_frames()
        updated = updated.iloc[::-1]
        with pytest.raises(ValueError, match="not aligned by question"):
            tester._build_per_question_diff(baseline, updated)

    def test_raises_on_relevant_doc_id_drift(self):
        tester = make_tester()
        baseline, updated = self.build_frames()
        updated.loc[0, "relevant_doc_ids"] = json.dumps(["d999"])
        with pytest.raises(ValueError, match="different relevant_doc_ids"):
            tester._build_per_question_diff(baseline, updated)


class TestOverallDiff:
    def test_means_and_diffs(self):
        tester = make_tester()
        baseline = pd.DataFrame(
            [
                make_row("Q1", {metric: 0.8 for metric in METRICS}),
                make_row("Q2", {metric: 1.0 for metric in METRICS}),
            ]
        )
        updated = pd.DataFrame(
            [
                make_row("Q1", {metric: 0.6 for metric in METRICS}),
                make_row("Q2", {metric: 0.8 for metric in METRICS}),
            ]
        )
        overall = tester._build_overall_diff(baseline, updated)
        assert overall["faithfulness"]["baseline"] == pytest.approx(0.9)
        assert overall["faithfulness"]["updated"] == pytest.approx(0.7)
        assert overall["faithfulness"]["diff"] == pytest.approx(-0.2)

    def test_missing_metric_column_yields_none(self):
        tester = make_tester()
        baseline = pd.DataFrame([{"question": "Q1", "faithfulness": 0.9}])
        updated = pd.DataFrame([{"question": "Q1", "faithfulness": 0.8}])
        overall = tester._build_overall_diff(baseline, updated)
        assert overall["answer_relevancy"] == {
            "baseline": None,
            "updated": None,
            "diff": None,
        }


class TestBuildPipelineIsolation:
    def test_versions_get_isolated_persist_dirs_and_collections(self, tmp_path):
        captured: list[dict] = []

        class RecordingPipeline:
            pipeline_name = "Recording Pipeline"

            def __init__(self, knowledge_base_path, collection_name, persist_dir):
                captured.append(
                    {
                        "knowledge_base_path": knowledge_base_path,
                        "collection_name": collection_name,
                        "persist_dir": persist_dir,
                    }
                )

        tester = make_tester(results_dir=tmp_path)
        run_id = tester._build_run_id(RecordingPipeline)
        for version_tag in ("v1", "v2"):
            tester._build_pipeline(
                pipeline_class=RecordingPipeline,
                knowledge_base_path=f"data/kb_{version_tag}.json",
                version_tag=version_tag,
                run_id=run_id,
                pipeline_kwargs=None,
            )

        v1, v2 = captured
        assert v1["persist_dir"] != v2["persist_dir"]
        assert v1["persist_dir"].endswith(f"{run_id}/v1")
        assert v2["persist_dir"].endswith(f"{run_id}/v2")
        assert v1["collection_name"] != v2["collection_name"]
        assert "_v1_" in v1["collection_name"]
        assert "_v2_" in v2["collection_name"]


class TestFormatDisplayPath:
    def test_project_paths_become_relative(self):
        tester = make_tester()
        project_root = Path(__file__).resolve().parents[1]
        inside = project_root / "data" / "eval_questions.json"
        assert tester._format_display_path(inside) == str(Path("data/eval_questions.json"))
        outside = Path("/somewhere/else/eval.json")
        assert tester._format_display_path(outside) == str(outside)


class ScriptedJudgeLLM:
    """Scores 0.9 when judging the v1 answer and 0.3 for the v2 answer."""

    def invoke(self, prompt: str) -> SimpleNamespace:
        if "good answer" in prompt:
            return SimpleNamespace(content="0.9")
        return SimpleNamespace(content="0.3")


class VersionedFakePipeline:
    """Answers depend on which knowledge base version was loaded."""

    pipeline_name = "VersionedFakePipeline"

    def __init__(self, knowledge_base_path, collection_name, persist_dir):
        self.is_v1 = "kb_v1" in str(knowledge_base_path)

    def query(self, question: str) -> dict:
        answer = "good answer" if self.is_v1 else "bad answer"
        return {
            "question": question,
            "answer": f"{answer} to {question}",
            "contexts": ["some context"],
            "retrieved_doc_ids": ["d1"],
        }


def test_run_regression_test_end_to_end_with_mocked_llm(tmp_path):
    eval_path = tmp_path / "eval_questions.json"
    eval_path.write_text(
        json.dumps(
            [
                {"question": "Q1?", "ground_truth": "GT1", "relevant_doc_ids": ["d1"]},
                {"question": "Q2?", "ground_truth": "GT2", "relevant_doc_ids": ["d1"]},
            ]
        ),
        encoding="utf-8",
    )

    tester = make_tester(results_dir=tmp_path / "results")
    tester.eval_dataset_path = str(eval_path)
    tester.eval_dataset_display_path = str(eval_path)
    tester.evaluator = RAGEvaluator(
        llm=ScriptedJudgeLLM(),
        embeddings=object(),
        evaluation_backend="fallback",
    )
    tester.metric_columns = list(tester.evaluator.METRIC_COLUMNS)

    result = tester.run_regression_test(
        pipeline_class=VersionedFakePipeline,
        kb_v1_path=str(tmp_path / "kb_v1.json"),
        kb_v2_path=str(tmp_path / "kb_v2.json"),
    )

    assert result["degraded_questions"] == ["Q1?", "Q2?"]
    assert result["improved_questions"] == []
    assert result["stable_questions"] == []

    for metric in LLM_METRICS:
        assert result["overall_diff"][metric]["baseline"] == pytest.approx(0.9)
        assert result["overall_diff"][metric]["updated"] == pytest.approx(0.3)
        assert result["overall_diff"][metric]["diff"] == pytest.approx(-0.6)
    # Retrieval behaviour is identical across versions -> zero diff.
    assert result["overall_diff"]["retrieval_hit"]["diff"] == pytest.approx(0.0)

    per_question = result["per_question_diff"]
    assert len(per_question) == 2
    for item in per_question:
        assert item["status"] == "degraded"
        assert sorted(item["degraded_metrics"]) == sorted(LLM_METRICS)

    report = tester.generate_report_text(result)
    assert "问题总数: 2" in report
    assert "退化 2 题" in report
    assert "Q1?" in report and "Q2?" in report


def test_generate_report_text_lists_doc_ids_for_retrieval_degradation():
    tester = make_tester()
    regression_result = {
        "overall_diff": {
            metric: {"baseline": 0.9, "updated": 0.7, "diff": -0.2} for metric in METRICS
        },
        "per_question_diff": [
            {
                "question": "Which doc answers Q1?",
                "status": "degraded",
                "degraded_metrics": ["retrieval_recall"],
                "diff": {"retrieval_recall": -0.5},
                "relevant_doc_ids": ["d1"],
                "baseline_retrieved_doc_ids": ["d1", "d2"],
                "updated_retrieved_doc_ids": ["d3"],
            }
        ],
    }
    report = tester.generate_report_text(regression_result)
    assert "退化问题列表 (1)" in report
    assert "Which doc answers Q1?" in report
    assert "retrieval_recall(-0.500)" in report
    assert "相关文档: d1" in report
    assert "Baseline 检索: d1, d2" in report
    assert "Updated 检索: d3" in report
