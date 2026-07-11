# -*- coding: utf-8 -*-
"""Regression testing helpers for comparing two RAG knowledge base versions."""

from __future__ import annotations

import json
import math
import re
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import pandas as pd

try:
    from .evaluation_engine import RAGEvaluator
    from .rag_pipelines import NaiveVectorRAG
except ImportError:
    try:
        from src.evaluation_engine import RAGEvaluator  # type: ignore
        from src.rag_pipelines import NaiveVectorRAG  # type: ignore
    except ImportError:
        from evaluation_engine import RAGEvaluator  # type: ignore
        from rag_pipelines import NaiveVectorRAG  # type: ignore


class RegressionTester:
    """Compare evaluation results before and after a knowledge base update."""

    DEGRADATION_THRESHOLD = -0.1
    IMPROVEMENT_THRESHOLD = 0.1

    def __init__(
        self,
        eval_dataset_path: str,
        results_dir: str = "./results",
    ) -> None:
        self.eval_dataset_path = str(Path(eval_dataset_path).expanduser())
        self.eval_dataset_display_path = self._format_display_path(
            Path(self.eval_dataset_path)
        )
        self.results_dir = Path(results_dir).expanduser()
        self.evaluator = RAGEvaluator()
        self.metric_columns = list(self.evaluator.METRIC_COLUMNS)

    def run_regression_test(
        self,
        pipeline_class: type,
        kb_v1_path: str,
        kb_v2_path: str,
        pipeline_kwargs: dict | None = None,
        max_questions: int | None = None,
    ) -> dict:
        """Run two evaluations and compare per-question and overall metrics."""
        eval_dataset = self.evaluator.load_eval_dataset(self.eval_dataset_path)
        if max_questions is not None:
            eval_dataset = eval_dataset[:max_questions]

        if not eval_dataset:
            raise ValueError("Evaluation dataset is empty after applying max_questions.")

        run_id = self._build_run_id(pipeline_class)
        baseline_df = self._run_single_version(
            pipeline_class=pipeline_class,
            knowledge_base_path=kb_v1_path,
            version_tag="v1",
            run_id=run_id,
            eval_dataset=eval_dataset,
            pipeline_kwargs=pipeline_kwargs,
        )
        updated_df = self._run_single_version(
            pipeline_class=pipeline_class,
            knowledge_base_path=kb_v2_path,
            version_tag="v2",
            run_id=run_id,
            eval_dataset=eval_dataset,
            pipeline_kwargs=pipeline_kwargs,
        )

        baseline_summary = self.evaluator.compute_summary(baseline_df)
        updated_summary = self.evaluator.compute_summary(updated_df)
        per_question_diff = self._build_per_question_diff(baseline_df, updated_df)
        overall_diff = self._build_overall_diff(baseline_df, updated_df)

        degraded_questions = [
            item["question"]
            for item in per_question_diff
            if item["status"] == "degraded"
        ]
        improved_questions = [
            item["question"]
            for item in per_question_diff
            if item["status"] == "improved"
        ]
        stable_questions = [
            item["question"] for item in per_question_diff if item["status"] == "stable"
        ]

        return {
            "baseline_summary": baseline_summary,
            "updated_summary": updated_summary,
            "per_question_diff": per_question_diff,
            "overall_diff": overall_diff,
            "degraded_questions": degraded_questions,
            "improved_questions": improved_questions,
            "stable_questions": stable_questions,
        }

    def generate_report_text(self, regression_result: dict) -> str:
        """Format a regression result dict into a readable text report."""
        overall_diff = regression_result.get("overall_diff", {})
        per_question_diff = regression_result.get("per_question_diff", [])
        degraded_items = [
            item for item in per_question_diff if item.get("status") == "degraded"
        ]
        improved_items = [
            item for item in per_question_diff if item.get("status") == "improved"
        ]
        stable_items = [
            item for item in per_question_diff if item.get("status") == "stable"
        ]

        lines: list[str] = []
        lines.append("RAG 知识库回归测试报告")
        lines.append("=" * 80)
        lines.append(
            f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )
        lines.append(f"评测数据集: {self.eval_dataset_display_path}")
        lines.append(f"问题总数: {len(per_question_diff)}")
        lines.append(
            "结果概览: "
            f"退化 {len(degraded_items)} 题 | "
            f"改进 {len(improved_items)} 题 | "
            f"稳定 {len(stable_items)} 题"
        )
        lines.append("")

        lines.append("总体指标对比")
        lines.append("-" * 80)
        lines.append(
            f"{'metric':<20}{'baseline':>12}{'updated':>12}{'diff':>12}"
        )
        lines.append("-" * 80)
        for metric in self.metric_columns:
            metric_diff = overall_diff.get(metric, {})
            lines.append(
                f"{metric:<20}"
                f"{self._format_score(metric_diff.get('baseline')):>12}"
                f"{self._format_score(metric_diff.get('updated')):>12}"
                f"{self._format_signed_score(metric_diff.get('diff')):>12}"
            )
        lines.append("")

        lines.append(f"退化问题列表 ({len(degraded_items)})")
        lines.append("-" * 80)
        if degraded_items:
            for index, item in enumerate(degraded_items, start=1):
                degraded_metrics = item.get("degraded_metrics", [])
                diff_text = self._format_metric_diff_details(
                    item.get("diff", {}),
                    only_metrics=degraded_metrics,
                )
                lines.append(f"{index}. {item['question']}")
                lines.append(
                    "   退化指标: "
                    + (", ".join(degraded_metrics) if degraded_metrics else "无")
                )
                lines.append(
                    "   指标变化: " + (diff_text if diff_text else "无显著退化指标")
                )
                if any(metric.startswith("retrieval_") for metric in degraded_metrics):
                    lines.append(
                        "   相关文档: "
                        + self._format_doc_ids(item.get("relevant_doc_ids", []))
                    )
                    lines.append(
                        "   Baseline 检索: "
                        + self._format_doc_ids(
                            item.get("baseline_retrieved_doc_ids", [])
                        )
                    )
                    lines.append(
                        "   Updated 检索: "
                        + self._format_doc_ids(
                            item.get("updated_retrieved_doc_ids", [])
                        )
                    )
        else:
            lines.append("无退化问题。")
        lines.append("")

        lines.append(f"改进问题列表 ({len(improved_items)})")
        lines.append("-" * 80)
        if improved_items:
            for index, item in enumerate(improved_items, start=1):
                improved_metrics = [
                    metric
                    for metric, diff_value in item.get("diff", {}).items()
                    if diff_value is not None
                    and diff_value > self.IMPROVEMENT_THRESHOLD
                ]
                diff_text = self._format_metric_diff_details(
                    item.get("diff", {}),
                    only_metrics=improved_metrics,
                )
                lines.append(f"{index}. {item['question']}")
                lines.append(
                    "   改进指标: "
                    + (", ".join(improved_metrics) if improved_metrics else "无")
                )
                lines.append(
                    "   指标变化: " + (diff_text if diff_text else "无显著改进指标")
                )
                if any(metric.startswith("retrieval_") for metric in improved_metrics):
                    lines.append(
                        "   相关文档: "
                        + self._format_doc_ids(item.get("relevant_doc_ids", []))
                    )
                    lines.append(
                        "   Baseline 检索: "
                        + self._format_doc_ids(
                            item.get("baseline_retrieved_doc_ids", [])
                        )
                    )
                    lines.append(
                        "   Updated 检索: "
                        + self._format_doc_ids(
                            item.get("updated_retrieved_doc_ids", [])
                        )
                    )
        else:
            lines.append("无显著改进问题。")
        lines.append("")

        lines.append("结论与建议")
        lines.append("-" * 80)
        for suggestion in self._build_recommendations(
            overall_diff=overall_diff,
            degraded_items=degraded_items,
            improved_items=improved_items,
            stable_items=stable_items,
        ):
            lines.append(f"- {suggestion}")

        return "\n".join(lines)

    def save_regression_report(
        self,
        regression_result: dict,
        report_text: str,
        output_dir: str,
    ) -> None:
        """Save the regression result JSON and text report with a timestamp."""
        output_path = Path(output_dir).expanduser()
        output_path.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        json_path = output_path / f"regression_report_{timestamp}.json"
        txt_path = output_path / f"regression_report_{timestamp}.txt"

        with json_path.open("w", encoding="utf-8") as file:
            json.dump(regression_result, file, ensure_ascii=False, indent=2)

        with txt_path.open("w", encoding="utf-8") as file:
            file.write(report_text)

        print(f"Saved regression JSON report to: {json_path}")
        print(f"Saved regression text report to: {txt_path}")

    def _run_single_version(
        self,
        pipeline_class: type,
        knowledge_base_path: str,
        version_tag: str,
        run_id: str,
        eval_dataset: list[dict],
        pipeline_kwargs: dict | None,
    ) -> pd.DataFrame:
        """Instantiate one pipeline version and execute the evaluation set."""
        pipeline = self._build_pipeline(
            pipeline_class=pipeline_class,
            knowledge_base_path=knowledge_base_path,
            version_tag=version_tag,
            run_id=run_id,
            pipeline_kwargs=pipeline_kwargs,
        )
        return self.evaluator.run_pipeline_evaluation(pipeline, eval_dataset)

    def _build_pipeline(
        self,
        pipeline_class: type,
        knowledge_base_path: str,
        version_tag: str,
        run_id: str,
        pipeline_kwargs: dict | None,
    ) -> Any:
        """Build a pipeline with isolated Chroma settings for this run."""
        kwargs = dict(pipeline_kwargs or {})
        pipeline_name = getattr(
            pipeline_class,
            "pipeline_name",
            getattr(pipeline_class, "__name__", "rag_pipeline"),
        )
        default_collection_name = self._slugify_name(str(pipeline_name))
        configured_collection_name = self._slugify_name(
            str(kwargs.pop("collection_name", default_collection_name))
        )
        base_persist_dir = Path(
            kwargs.pop(
                "persist_dir",
                self.results_dir / "regression_runtime" / self._slugify_name(str(pipeline_name)),
            )
        ).expanduser()
        version_persist_dir = base_persist_dir / run_id / version_tag

        kwargs["collection_name"] = self._build_collection_name(
            configured_collection_name,
            version_tag,
            run_id,
        )
        kwargs["persist_dir"] = str(version_persist_dir)

        return pipeline_class(
            knowledge_base_path=str(Path(knowledge_base_path).expanduser()),
            **kwargs,
        )

    def _build_per_question_diff(
        self,
        baseline_df: pd.DataFrame,
        updated_df: pd.DataFrame,
    ) -> list[dict]:
        """Compare metric changes question by question."""
        baseline_records = baseline_df.reset_index(drop=True).to_dict("records")
        updated_records = updated_df.reset_index(drop=True).to_dict("records")

        if len(baseline_records) != len(updated_records):
            raise ValueError(
                "Baseline and updated evaluation results have different numbers of rows."
            )

        per_question_diff: list[dict] = []
        for baseline_row, updated_row in zip(
            baseline_records,
            updated_records,
            strict=True,
        ):
            baseline_question = str(baseline_row.get("question", ""))
            updated_question = str(updated_row.get("question", ""))
            if baseline_question != updated_question:
                raise ValueError(
                    "Baseline and updated results are not aligned by question. "
                    f"Found {baseline_question!r} vs {updated_question!r}."
                )

            relevant_doc_ids = self._deserialize_string_list(
                baseline_row.get("relevant_doc_ids", [])
            )
            updated_relevant_doc_ids = self._deserialize_string_list(
                updated_row.get("relevant_doc_ids", [])
            )
            if relevant_doc_ids != updated_relevant_doc_ids:
                raise ValueError(
                    "Baseline and updated results have different relevant_doc_ids for "
                    f"question {baseline_question!r}."
                )

            baseline_retrieved_doc_ids = self._deserialize_string_list(
                baseline_row.get("retrieved_doc_ids", [])
            )
            updated_retrieved_doc_ids = self._deserialize_string_list(
                updated_row.get("retrieved_doc_ids", [])
            )
            relevant_doc_id_set = set(relevant_doc_ids)
            baseline_matched_doc_ids = [
                doc_id
                for doc_id in baseline_retrieved_doc_ids
                if doc_id in relevant_doc_id_set
            ]
            updated_matched_doc_ids = [
                doc_id
                for doc_id in updated_retrieved_doc_ids
                if doc_id in relevant_doc_id_set
            ]

            baseline_scores = {
                metric: self._to_clean_float(baseline_row.get(metric))
                for metric in self.metric_columns
            }
            updated_scores = {
                metric: self._to_clean_float(updated_row.get(metric))
                for metric in self.metric_columns
            }
            diff = {
                metric: self._compute_diff(
                    baseline_scores.get(metric),
                    updated_scores.get(metric),
                )
                for metric in self.metric_columns
            }
            degraded_metrics = [
                metric
                for metric, diff_value in diff.items()
                if diff_value is not None and diff_value < self.DEGRADATION_THRESHOLD
            ]
            status = self._classify_status(diff)

            per_question_diff.append(
                {
                    "question": baseline_question,
                    "relevant_doc_ids": relevant_doc_ids,
                    "baseline_retrieved_doc_ids": baseline_retrieved_doc_ids,
                    "updated_retrieved_doc_ids": updated_retrieved_doc_ids,
                    "baseline_matched_relevant_doc_ids": baseline_matched_doc_ids,
                    "updated_matched_relevant_doc_ids": updated_matched_doc_ids,
                    "baseline_scores": baseline_scores,
                    "updated_scores": updated_scores,
                    "diff": diff,
                    "status": status,
                    "degraded_metrics": degraded_metrics,
                }
            )

        return per_question_diff

    def _build_overall_diff(
        self,
        baseline_df: pd.DataFrame,
        updated_df: pd.DataFrame,
    ) -> dict:
        """Compute average metric changes across the whole evaluation set."""
        overall_diff: dict[str, dict[str, float | None]] = {}
        for metric in self.metric_columns:
            baseline_mean = self._series_mean(baseline_df.get(metric))
            updated_mean = self._series_mean(updated_df.get(metric))
            overall_diff[metric] = {
                "baseline": baseline_mean,
                "updated": updated_mean,
                "diff": self._compute_diff(baseline_mean, updated_mean),
            }
        return overall_diff

    def _classify_status(self, diff: dict[str, float | None]) -> str:
        """Map per-question metric changes to degraded/improved/stable."""
        diff_values = [value for value in diff.values() if value is not None]
        if any(value < self.DEGRADATION_THRESHOLD for value in diff_values):
            return "degraded"
        if any(value > self.IMPROVEMENT_THRESHOLD for value in diff_values):
            return "improved"
        return "stable"

    def _build_recommendations(
        self,
        overall_diff: dict,
        degraded_items: list[dict],
        improved_items: list[dict],
        stable_items: list[dict],
    ) -> list[str]:
        """Produce short, actionable suggestions from regression outcomes."""
        suggestions: list[str] = []
        negative_metrics = [
            (metric, values.get("diff"))
            for metric, values in overall_diff.items()
            if values.get("diff") is not None and values.get("diff") < 0
        ]
        negative_metrics.sort(key=lambda item: item[1])

        if degraded_items:
            suggestions.append(
                f"存在 {len(degraded_items)} 道退化题，优先复核这些问题的检索结果、chunk 切分和知识条目覆盖范围。"
            )
            worst_metric_names = [metric for metric, _diff in negative_metrics[:3]]
            if worst_metric_names:
                suggestions.append(
                    "整体下降最明显的指标为 "
                    + ", ".join(worst_metric_names)
                    + "，建议先定位这些指标对应的失败样例。"
                )
            if any(metric.startswith("retrieval_") for metric, _diff in negative_metrics):
                suggestions.append(
                    "检索命中指标出现下降，优先核对 relevant_doc_ids 是否仍能被召回，以及新增内容是否干扰了原有排序。"
                )
        else:
            suggestions.append("未发现超过阈值的退化问题，本次知识库更新整体稳定。")

        if improved_items:
            suggestions.append(
                f"有 {len(improved_items)} 道题显著提升，可回看新增或修改过的知识条目，确认哪些改动带来了正向效果。"
            )

        if stable_items:
            suggestions.append(
                f"共有 {len(stable_items)} 道题保持稳定，可继续扩大评测集覆盖更多边界问题。"
            )

        if not negative_metrics and improved_items:
            suggestions.append("总体均值未出现负向波动，本次更新可以作为较安全的候选版本。")

        if not suggestions:
            suggestions.append("建议结合人工抽样复核回答内容，确认自动指标与业务感知一致。")

        return suggestions

    def _build_run_id(self, pipeline_class: type) -> str:
        """Create a short unique identifier for the current regression run."""
        pipeline_name = getattr(
            pipeline_class,
            "pipeline_name",
            getattr(pipeline_class, "__name__", "rag_pipeline"),
        )
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{self._slugify_name(str(pipeline_name))}_{timestamp}_{uuid4().hex[:8]}"

    def _build_collection_name(
        self,
        base_name: str,
        version_tag: str,
        run_id: str,
    ) -> str:
        """Keep Chroma collection names short and unique for each version."""
        token = self._slugify_name(run_id)[-18:]
        normalized_base = self._slugify_name(base_name)[:30].strip("_") or "rag"
        return f"{normalized_base}_{version_tag}_{token}"

    def _series_mean(self, series: Any) -> float | None:
        """Safely compute a numeric mean from a pandas Series-like object."""
        if series is None:
            return None
        numeric_series = pd.to_numeric(series, errors="coerce")
        return self._to_clean_float(numeric_series.mean())

    def _compute_diff(
        self,
        baseline_value: float | None,
        updated_value: float | None,
    ) -> float | None:
        """Return updated - baseline when both values are numeric."""
        if baseline_value is None or updated_value is None:
            return None
        return self._to_clean_float(updated_value - baseline_value)

    def _to_clean_float(self, value: Any) -> float | None:
        """Convert pandas/numpy values into JSON-safe floats."""
        if value is None:
            return None
        try:
            numeric_value = float(value)
        except (TypeError, ValueError):
            return None
        if math.isnan(numeric_value) or math.isinf(numeric_value):
            return None
        return numeric_value

    def _format_score(self, value: Any) -> str:
        """Format a score for the text report table."""
        numeric_value = self._to_clean_float(value)
        if numeric_value is None:
            return "n/a"
        return f"{numeric_value:.3f}"

    def _format_signed_score(self, value: Any) -> str:
        """Format a signed score for the text report table."""
        numeric_value = self._to_clean_float(value)
        if numeric_value is None:
            return "n/a"
        return f"{numeric_value:+.3f}"

    def _format_metric_diff_details(
        self,
        diff: dict[str, float | None],
        only_metrics: list[str] | None = None,
    ) -> str:
        """Format selected metric diffs into a compact text fragment."""
        metrics = only_metrics if only_metrics is not None else self.metric_columns
        parts = []
        for metric in metrics:
            diff_value = diff.get(metric)
            if diff_value is None:
                continue
            parts.append(f"{metric}({diff_value:+.3f})")
        return ", ".join(parts)

    def _deserialize_string_list(self, value: Any) -> list[str]:
        """Decode a JSON-serialized string list from a dataframe cell."""
        if value is None:
            return []
        if isinstance(value, list):
            return [str(item) for item in value]
        if isinstance(value, str):
            stripped = value.strip()
            if not stripped:
                return []
            try:
                parsed = json.loads(stripped)
            except json.JSONDecodeError:
                return [stripped]
            if isinstance(parsed, list):
                return [str(item) for item in parsed]
            return [str(parsed)]
        return [str(value)]

    def _format_doc_ids(self, doc_ids: list[str]) -> str:
        """Format a doc-id list for the text report."""
        if not doc_ids:
            return "无"
        return ", ".join(doc_ids)

    def _format_display_path(self, path: Path) -> str:
        """Prefer project-relative paths in human-readable outputs."""
        resolved = path.expanduser().resolve(strict=False)
        project_root = Path(__file__).resolve().parent.parent
        try:
            return str(resolved.relative_to(project_root))
        except ValueError:
            return str(path)

    def _slugify_name(self, value: str) -> str:
        """Normalize names for filesystem paths and collection ids."""
        cleaned = re.sub(r"[^A-Za-z0-9_-]+", "_", value.strip())
        return cleaned.strip("_") or "rag"


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent
    debug_eval_dataset_path = (
        project_root / "data" / "eval_questions_regression_debug.json"
    )
    eval_dataset_path = (
        debug_eval_dataset_path
        if debug_eval_dataset_path.exists()
        else project_root / "data" / "eval_questions.json"
    )
    kb_v1_path = project_root / "data" / "knowledge_base_v1.json"
    kb_v2_path = project_root / "data" / "knowledge_base_v2.json"

    tester = RegressionTester(
        eval_dataset_path=str(eval_dataset_path),
        results_dir=str(project_root / "results"),
    )
    regression_result = tester.run_regression_test(
        pipeline_class=NaiveVectorRAG,
        kb_v1_path=str(kb_v1_path),
        kb_v2_path=str(kb_v2_path),
        max_questions=4,
    )
    report_text = tester.generate_report_text(regression_result)
    print(report_text)
