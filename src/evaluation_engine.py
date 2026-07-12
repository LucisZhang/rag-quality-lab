# -*- coding: utf-8 -*-
"""Evaluation helpers for comparing local RAG pipelines."""

from __future__ import annotations

import json
import math
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

try:
    from .rag_pipelines import NaiveVectorRAG
    from .utils import get_embeddings, get_llm
except ImportError:
    try:
        from src.rag_pipelines import NaiveVectorRAG  # type: ignore
        from src.utils import get_embeddings, get_llm  # type: ignore
    except ImportError:
        from rag_pipelines import NaiveVectorRAG  # type: ignore
        from utils import get_embeddings, get_llm  # type: ignore


def normalize_doc_ids(doc_ids: Any) -> list[str]:
    """Normalize doc ids to a unique, ordered list of non-empty strings."""
    if doc_ids is None:
        return []
    if isinstance(doc_ids, str):
        stripped = doc_ids.strip()
        return [stripped] if stripped else []
    if not isinstance(doc_ids, list):
        values = [str(doc_ids).strip()]
    else:
        values = [str(item).strip() for item in doc_ids]

    normalized: list[str] = []
    seen: set[str] = set()
    for value in values:
        if not value or value in seen:
            continue
        seen.add(value)
        normalized.append(value)
    return normalized


def compute_retrieval_metrics(
    retrieved_doc_ids: list[str],
    relevant_doc_ids: list[str],
) -> dict[str, float]:
    """Compute deterministic document-level retrieval diagnostics."""
    retrieved = normalize_doc_ids(retrieved_doc_ids)
    relevant = normalize_doc_ids(relevant_doc_ids)
    relevant_set = set(relevant)
    matched = [doc_id for doc_id in retrieved if doc_id in relevant_set]

    precision = len(matched) / len(retrieved) if retrieved else 0.0
    recall = len(matched) / len(relevant) if relevant else 1.0
    hit = 1.0 if matched else 0.0

    return {
        "retrieval_precision": precision,
        "retrieval_recall": recall,
        "retrieval_hit": hit,
    }


class RAGEvaluator:
    """Run RAG pipeline evaluations with RAGAS and an LLM-based fallback."""

    EVALUATION_BACKENDS = {"auto", "ragas", "fallback"}

    LLM_METRIC_COLUMNS = [
        "faithfulness",
        "answer_relevancy",
        "context_precision",
        "context_recall",
        "answer_correctness",
    ]
    RETRIEVAL_METRIC_COLUMNS = [
        "retrieval_precision",
        "retrieval_recall",
        "retrieval_hit",
    ]
    METRIC_COLUMNS = [*LLM_METRIC_COLUMNS, *RETRIEVAL_METRIC_COLUMNS]
    RESULT_COLUMNS = [
        "question",
        "answer",
        "contexts",
        "ground_truth",
        "retrieved_doc_ids",
        "relevant_doc_ids",
        *METRIC_COLUMNS,
    ]
    FALLBACK_METRICS = {
        "faithfulness": (
            "Whether the answer is supported by the retrieved context and avoids "
            "hallucinated claims."
        ),
        "answer_relevancy": (
            "Whether the answer directly addresses the user's question."
        ),
        "context_precision": (
            "Whether the retrieved context is focused and relevant to answering "
            "the question."
        ),
        "context_recall": (
            "Whether the retrieved context contains enough of the key facts "
            "needed for a correct answer."
        ),
        "answer_correctness": (
            "Whether the answer matches the ground truth in factual accuracy "
            "and completeness."
        ),
    }

    def __init__(
        self,
        llm: Any | None = None,
        embeddings: Any | None = None,
        evaluation_backend: str = "auto",
        ragas_timeout: int = 180,
        ragas_max_workers: int = 4,
        ragas_batch_size: int | None = None,
        ragas_max_retries: int = 1,
    ) -> None:
        if evaluation_backend not in self.EVALUATION_BACKENDS:
            raise ValueError(
                "evaluation_backend must be one of "
                f"{sorted(self.EVALUATION_BACKENDS)}, got {evaluation_backend!r}."
            )
        self.llm = llm if llm is not None else get_llm()
        self.embeddings = embeddings if embeddings is not None else get_embeddings()
        self.evaluation_backend = evaluation_backend
        self.ragas_timeout = ragas_timeout
        self.ragas_max_workers = ragas_max_workers
        self.ragas_batch_size = ragas_batch_size
        self.ragas_max_retries = ragas_max_retries

    def load_eval_dataset(self, json_path: str) -> list[dict]:
        """Load evaluation questions from a JSON file."""
        path = Path(json_path).expanduser()
        with path.open("r", encoding="utf-8") as file:
            records = json.load(file)

        if not isinstance(records, list):
            raise ValueError("Evaluation dataset JSON must be a list of objects.")

        dataset: list[dict] = []
        required_keys = {"question", "ground_truth", "relevant_doc_ids"}
        for index, record in enumerate(records, start=1):
            if not isinstance(record, dict):
                raise ValueError(
                    f"Evaluation item #{index} must be a JSON object, got {type(record)!r}."
                )
            missing_keys = required_keys.difference(record)
            if missing_keys:
                raise ValueError(
                    f"Evaluation item #{index} is missing keys: {sorted(missing_keys)}."
                )
            relevant_doc_ids = record["relevant_doc_ids"]
            if not isinstance(relevant_doc_ids, list):
                raise ValueError(
                    f"Evaluation item #{index} field 'relevant_doc_ids' must be a list."
                )

            dataset.append(
                {
                    "question": str(record["question"]),
                    "ground_truth": str(record["ground_truth"]),
                    "relevant_doc_ids": [str(doc_id) for doc_id in relevant_doc_ids],
                }
            )

        return dataset

    def run_pipeline_evaluation(
        self,
        pipeline: Any,
        eval_dataset: list[dict],
    ) -> pd.DataFrame:
        """Evaluate a pipeline across the supplied dataset."""
        if not eval_dataset:
            return pd.DataFrame(columns=self.RESULT_COLUMNS)

        collected_results: list[dict[str, Any]] = []
        total_questions = len(eval_dataset)

        for index, item in enumerate(eval_dataset, start=1):
            question = str(item["question"])
            ground_truth = str(item["ground_truth"])

            print(f"[{index}/{total_questions}] Running pipeline query: {question}")
            query_result = pipeline.query(question)
            if not isinstance(query_result, dict):
                raise TypeError(
                    "pipeline.query(question) must return a dict with 'answer' and "
                    f"'contexts', got {type(query_result)!r}."
                )

            answer_value = query_result.get("answer", "")
            answer = "" if answer_value is None else str(answer_value).strip()
            contexts = self._normalize_contexts(query_result.get("contexts", []))

            collected_results.append(
                {
                    "question": question,
                    "answer": answer,
                    "contexts": contexts,
                    "ground_truth": ground_truth,
                    "relevant_doc_ids": self._normalize_doc_ids(
                        item.get("relevant_doc_ids", [])
                    ),
                    "retrieved_doc_ids": self._normalize_doc_ids(
                        query_result.get("retrieved_doc_ids", [])
                    ),
                }
            )

        resolved_backend = self._resolve_evaluation_backend()
        if resolved_backend == "fallback":
            print(
                "ℹ️ Using fallback LLM evaluation. "
                f"Reason: {self._fallback_reason()}."
            )
            return self._evaluate_with_fallback(collected_results)

        try:
            return self._evaluate_with_ragas(collected_results)
        except Exception as exc:
            print(
                "⚠️ RAGAS evaluation failed; switching to fallback LLM scoring. "
                f"Reason: {self._format_exception(exc)}"
            )
            return self._evaluate_with_fallback(collected_results)

    def compute_summary(self, results_df: pd.DataFrame) -> dict:
        """Compute mean/min/max/std for each evaluation metric."""
        summary: dict[str, dict[str, float | None]] = {}
        for metric in self.METRIC_COLUMNS:
            if metric not in results_df.columns:
                continue

            series = pd.to_numeric(results_df[metric], errors="coerce")
            summary[metric] = {
                "mean": self._clean_stat(series.mean()),
                "min": self._clean_stat(series.min()),
                "max": self._clean_stat(series.max()),
                "std": self._clean_stat(series.std()),
            }

        return summary

    def save_results(
        self,
        results_df: pd.DataFrame,
        summary: dict,
        output_dir: str,
        pipeline_name: str,
    ) -> None:
        """Persist detailed results and summary statistics."""
        output_path = Path(output_dir).expanduser()
        output_path.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_name = self._slugify_name(pipeline_name)
        csv_path = output_path / f"{safe_name}_{timestamp}_results.csv"
        json_path = output_path / f"{safe_name}_{timestamp}_summary.json"

        results_df.to_csv(csv_path, index=False, encoding="utf-8")
        with json_path.open("w", encoding="utf-8") as file:
            json.dump(summary, file, ensure_ascii=False, indent=2)

        print(f"Saved evaluation results to: {csv_path}")
        print(f"Saved summary statistics to: {json_path}")

    def _evaluate_with_ragas(
        self,
        collected_results: list[dict[str, Any]],
    ) -> pd.DataFrame:
        """Run the primary RAGAS evaluation path."""
        from ragas import evaluate
        from ragas.dataset_schema import EvaluationDataset, SingleTurnSample
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from ragas.llms import LangchainLLMWrapper
        from ragas.metrics import (
            AnswerCorrectness,
            AnswerRelevancy,
            ContextPrecision,
            ContextRecall,
            Faithfulness,
        )
        from ragas.run_config import RunConfig

        samples = [
            SingleTurnSample(
                user_input=item["question"],
                response=item["answer"],
                retrieved_contexts=item["contexts"],
                reference=item["ground_truth"],
            )
            for item in collected_results
        ]
        eval_dataset = EvaluationDataset(samples=samples)

        wrapped_llm = LangchainLLMWrapper(self.llm)
        wrapped_embeddings = LangchainEmbeddingsWrapper(self.embeddings)
        run_config = RunConfig(
            timeout=self.ragas_timeout,
            max_workers=self.ragas_max_workers,
            max_retries=self.ragas_max_retries,
        )

        print("⏳ Running RAGAS evaluation...")
        results = evaluate(
            dataset=eval_dataset,
            metrics=[
                Faithfulness(llm=wrapped_llm),
                AnswerRelevancy(llm=wrapped_llm, embeddings=wrapped_embeddings),
                ContextPrecision(llm=wrapped_llm),
                ContextRecall(llm=wrapped_llm),
                AnswerCorrectness(llm=wrapped_llm, embeddings=wrapped_embeddings),
            ],
            run_config=run_config,
            raise_exceptions=True,
            show_progress=True,
            batch_size=self.ragas_batch_size,
        )

        scores_df = pd.DataFrame(results.scores)
        missing_columns = [
            metric
            for metric in self.LLM_METRIC_COLUMNS
            if metric not in scores_df.columns
        ]
        if missing_columns:
            raise ValueError(
                f"RAGAS result is missing expected metric columns: {missing_columns}."
            )

        numeric_scores_df = scores_df[self.LLM_METRIC_COLUMNS].apply(
            pd.to_numeric, errors="coerce"
        )
        invalid_metrics = [
            metric
            for metric in self.LLM_METRIC_COLUMNS
            if numeric_scores_df[metric].isna().any()
        ]
        if invalid_metrics:
            raise ValueError(
                "RAGAS returned incomplete metric values for: "
                f"{invalid_metrics}."
            )

        base_rows = [self._build_base_result_row(item) for item in collected_results]
        results_df = pd.DataFrame(base_rows, columns=self.RESULT_COLUMNS)

        for metric in self.LLM_METRIC_COLUMNS:
            results_df[metric] = numeric_scores_df[metric]

        return results_df[self.RESULT_COLUMNS]

    def _evaluate_with_fallback(
        self,
        collected_results: list[dict[str, Any]],
    ) -> pd.DataFrame:
        """Fallback path when RAGAS is unavailable or incompatible."""
        print("⏳ Running fallback LLM evaluation...")
        rows: list[dict[str, Any]] = []
        total_questions = len(collected_results)

        for index, item in enumerate(collected_results, start=1):
            print(f"[fallback {index}/{total_questions}] Scoring collected answer...")
            contexts_text = self._join_contexts(item["contexts"])
            metric_scores = {
                metric_name: self._score_metric_with_llm(
                    question=item["question"],
                    answer=item["answer"],
                    contexts_text=contexts_text,
                    ground_truth=item["ground_truth"],
                    metric_name=metric_name,
                    metric_description=metric_description,
                )
                for metric_name, metric_description in self.FALLBACK_METRICS.items()
            }

            rows.append(
                self._build_base_result_row(item, metric_scores=metric_scores)
            )

        return pd.DataFrame(rows, columns=self.RESULT_COLUMNS)

    def _score_metric_with_llm(
        self,
        question: str,
        answer: str,
        contexts_text: str,
        ground_truth: str,
        metric_name: str,
        metric_description: str,
    ) -> float:
        """Ask the local LLM for a simplified 0-1 metric score."""
        prompt = (
            "Rate the following on a scale of 0 to 1. Only output a number.\n"
            f"Question: {question}\n"
            f"Answer: {answer}\n"
            f"Context: {contexts_text}\n"
            f"Ground Truth: {ground_truth}\n"
            f"Metric: {metric_name} - {metric_description}\n"
            "Score:"
        )
        response_text = self._message_content_to_text(self.llm.invoke(prompt))
        return self._parse_score(response_text)

    def _normalize_contexts(self, contexts: Any) -> list[str]:
        """Normalize pipeline contexts into a list of strings."""
        if contexts is None:
            return []
        if isinstance(contexts, str):
            stripped = contexts.strip()
            return [stripped] if stripped else []
        if not isinstance(contexts, list):
            return [self._context_item_to_text(contexts)]

        normalized = [self._context_item_to_text(item) for item in contexts]
        return [item for item in normalized if item]

    def _normalize_doc_ids(self, doc_ids: Any) -> list[str]:
        """Normalize retrieved/relevant doc ids to a unique ordered string list."""
        return normalize_doc_ids(doc_ids)

    def _context_item_to_text(self, item: Any) -> str:
        """Convert a context object into plain text."""
        if isinstance(item, str):
            return item.strip()
        if isinstance(item, dict):
            for key in ("page_content", "content", "text"):
                value = item.get(key)
                if value:
                    return str(value).strip()
        page_content = getattr(item, "page_content", None)
        if page_content:
            return str(page_content).strip()
        return str(item).strip()

    def _join_contexts(self, contexts: list[str]) -> str:
        """Format contexts for LLM prompts."""
        if not contexts:
            return "No retrieved context."
        return "\n\n".join(
            f"[Context {index}] {context}"
            for index, context in enumerate(contexts, start=1)
        )

    def _stringify_contexts(self, contexts: list[str]) -> str:
        """Serialize contexts into a stable string column."""
        return json.dumps(contexts, ensure_ascii=False)

    def _stringify_string_list(self, values: list[str]) -> str:
        """Serialize a string list into a stable string column."""
        return json.dumps(values, ensure_ascii=False)

    def _build_base_result_row(
        self,
        item: dict[str, Any],
        metric_scores: dict[str, float] | None = None,
    ) -> dict[str, Any]:
        """Construct a result row with deterministic retrieval diagnostics."""
        retrieved_doc_ids = self._normalize_doc_ids(item.get("retrieved_doc_ids", []))
        relevant_doc_ids = self._normalize_doc_ids(item.get("relevant_doc_ids", []))
        row: dict[str, Any] = {
            "question": item["question"],
            "answer": item["answer"],
            "contexts": self._stringify_contexts(item["contexts"]),
            "ground_truth": item["ground_truth"],
            "retrieved_doc_ids": self._stringify_string_list(retrieved_doc_ids),
            "relevant_doc_ids": self._stringify_string_list(relevant_doc_ids),
        }
        for metric in self.METRIC_COLUMNS:
            row[metric] = None
        row.update(self._compute_retrieval_metrics(retrieved_doc_ids, relevant_doc_ids))
        if metric_scores is not None:
            row.update(metric_scores)
        return row

    def _compute_retrieval_metrics(
        self,
        retrieved_doc_ids: list[str],
        relevant_doc_ids: list[str],
    ) -> dict[str, float]:
        """Compute deterministic document-level retrieval diagnostics."""
        return compute_retrieval_metrics(retrieved_doc_ids, relevant_doc_ids)

    def _message_content_to_text(self, message: Any) -> str:
        """Normalize LangChain response payloads to plain text."""
        content = getattr(message, "content", message)
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

    def _parse_score(self, response_text: str) -> float:
        """Parse an LLM score and clamp it to [0, 1]."""
        match = re.search(r"[-+]?\d*\.?\d+", response_text)
        if match is None:
            return float("nan")

        score = float(match.group())
        return max(0.0, min(1.0, score))

    def _clean_stat(self, value: Any) -> float | None:
        """Convert pandas/numpy scalars to JSON-safe floats."""
        if value is None:
            return None
        try:
            numeric_value = float(value)
        except (TypeError, ValueError):
            return None
        if math.isnan(numeric_value) or math.isinf(numeric_value):
            return None
        return numeric_value

    def _slugify_name(self, pipeline_name: str) -> str:
        """Normalize pipeline names for filenames."""
        cleaned = re.sub(r"[^A-Za-z0-9_-]+", "_", pipeline_name.strip())
        return cleaned.strip("_") or "pipeline"

    def _resolve_evaluation_backend(self) -> str:
        """Pick the evaluation backend after applying environment heuristics."""
        if self.evaluation_backend != "auto":
            return self.evaluation_backend
        if self._uses_ollama_llm():
            return "fallback"
        return "ragas"

    def _uses_ollama_llm(self) -> bool:
        """Detect Ollama-backed chat models, which are fragile with RAGAS async eval."""
        llm_type = type(self.llm)
        module_name = getattr(llm_type, "__module__", "").lower()
        class_name = getattr(llm_type, "__name__", "").lower()
        return "ollama" in module_name or "ollama" in class_name

    def _fallback_reason(self) -> str:
        """Explain why the evaluator chose the fallback scoring path."""
        if self.evaluation_backend == "fallback":
            return "evaluation_backend='fallback'"
        if self._uses_ollama_llm():
            return (
                "detected a local Ollama judge model; "
                "RAGAS async evaluation is prone to timeout/cancellation noise in this setup"
            )
        return "RAGAS is unavailable"

    def _format_exception(self, exc: Exception) -> str:
        """Return a readable exception string, even for empty TimeoutError messages."""
        detail = str(exc).strip()
        if detail:
            return f"{type(exc).__name__}: {detail}"
        return type(exc).__name__


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent
    knowledge_base_path = project_root / "data" / "knowledge_base_v2.json"
    eval_questions_path = project_root / "data" / "eval_questions.json"

    pipeline = NaiveVectorRAG(str(knowledge_base_path))
    evaluator = RAGEvaluator()
    eval_dataset = evaluator.load_eval_dataset(str(eval_questions_path))
    results_df = evaluator.run_pipeline_evaluation(pipeline, eval_dataset[:3])
    summary = evaluator.compute_summary(results_df)

    score_columns = ["question", *RAGEvaluator.METRIC_COLUMNS]
    print("\nPer-question scores:")
    print(results_df[score_columns].to_string(index=False))
    print("\nSummary:")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
