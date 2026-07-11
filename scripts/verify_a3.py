#!/usr/bin/env python3
"""A3 verification runner for RAG Quality Lab."""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation_engine import RAGEvaluator  # noqa: E402
from src.rag_pipelines import HybridRerankRAG, NaiveVectorRAG  # noqa: E402
from src.regression_tester import RegressionTester  # noqa: E402


DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
EVIDENCE_DIR = PROJECT_ROOT / "evidence" / "verified-2026-07"
MODEL_NAMES = {"gemma4:e4b", "nomic-embed-text:latest", "nomic-embed-text"}
METRICS = [
    "faithfulness",
    "answer_relevancy",
    "context_precision",
    "context_recall",
    "answer_correctness",
]


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def count_optional_json_array(path: Path) -> int | None:
    if not path.exists():
        return None
    data = read_json(path)
    if not isinstance(data, list):
        raise ValueError(f"{path} is not a top-level JSON array")
    return len(data)


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def mean_score(summary: dict[str, Any]) -> float:
    values = []
    for metric in METRICS:
        value = summary.get(metric, {}).get("mean")
        if value is not None:
            values.append(float(value))
    if not values:
        raise ValueError("No metric means found in summary.")
    return sum(values) / len(values)


def check_ollama_models() -> dict[str, Any]:
    url = "http://127.0.0.1:11434/api/tags"
    try:
        with urllib.request.urlopen(url, timeout=5) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError) as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

    names = {str(item.get("name")) for item in payload.get("models", [])}
    has_llm = "gemma4:e4b" in names
    has_embed = bool({"nomic-embed-text:latest", "nomic-embed-text"} & names)
    return {
        "ok": has_llm and has_embed,
        "models": sorted(names),
        "has_gemma4_e4b": has_llm,
        "has_nomic_embed_text": has_embed,
    }


def deterministic_checks() -> dict[str, Any]:
    checks: dict[str, Any] = {"checked_at_utc": utc_now()}

    kb_v1 = read_json(DATA_DIR / "knowledge_base_v1.json")
    kb_v2 = read_json(DATA_DIR / "knowledge_base_v2.json")
    eval_questions = read_json(DATA_DIR / "eval_questions.json")
    large_eval_count = count_optional_json_array(DATA_DIR / "large_eval_questions.json")

    checks["data_counts"] = {
        "knowledge_base_v1": len(kb_v1),
        "knowledge_base_v2": len(kb_v2),
        "eval_questions": len(eval_questions),
        "large_eval_questions": large_eval_count,
        "large_eval_questions_present": large_eval_count is not None,
        "large_eval_questions_note": (
            "present locally"
            if large_eval_count is not None
            else "absent by design in public clones; regenerate from MS MARCO with scale_up_dataset.py"
        ),
        "large_knowledge_base_present": (DATA_DIR / "large_knowledge_base.json").exists(),
    }

    evaluator = RAGEvaluator(evaluation_backend="fallback")
    checks["eval_dataset_valid"] = len(
        evaluator.load_eval_dataset(str(DATA_DIR / "eval_questions.json"))
    )

    comparison_a = read_json(
        RESULTS_DIR
        / "dashboard_comparison"
        / "pipeline_a_12_questions_20260419_163805_summary.json"
    )
    comparison_b = read_json(
        RESULTS_DIR
        / "dashboard_comparison"
        / "pipeline_b_12_questions_20260419_163806_summary.json"
    )
    regression = read_json(
        RESULTS_DIR
        / "dashboard_regression"
        / "regression_report_20260419_190818.json"
    )

    checks["saved_metrics"] = {
        "comparison_pipeline_a_mean": mean_score(comparison_a),
        "comparison_pipeline_b_mean": mean_score(comparison_b),
        "comparison_relative_lift": (
            mean_score(comparison_b) - mean_score(comparison_a)
        )
        / mean_score(comparison_a),
        "regression_degraded": len(regression.get("degraded_questions", [])),
        "regression_improved": len(regression.get("improved_questions", [])),
        "regression_stable": len(regression.get("stable_questions", [])),
    }

    indexing = pd.read_csv(RESULTS_DIR / "scale_performance" / "indexing_benchmark_20260417_143608.csv")
    latency = pd.read_csv(RESULTS_DIR / "scale_performance" / "latency_benchmark_summary_20260417_150622.csv")
    checks["saved_scale_artifacts"] = {
        "indexing_rows": int(len(indexing)),
        "latency_rows": int(len(latency)),
        "max_documents_indexed": int(indexing["Documents"].max()),
    }

    checks["ollama"] = check_ollama_models()
    return checks


def run_pipeline(
    pipeline_class: type,
    kb_path: Path,
    collection_name: str,
    persist_dir: Path,
    evaluator: RAGEvaluator,
    eval_dataset: list[dict[str, Any]],
) -> tuple[pd.DataFrame, dict[str, Any], float]:
    start = time.perf_counter()
    pipeline = pipeline_class(
        str(kb_path),
        collection_name=collection_name,
        persist_dir=str(persist_dir),
    )
    results_df = evaluator.run_pipeline_evaluation(pipeline, eval_dataset)
    summary = evaluator.compute_summary(results_df)
    elapsed = time.perf_counter() - start
    return results_df, summary, elapsed


def run_fresh(max_questions: int) -> dict[str, Any]:
    output_dir = EVIDENCE_DIR / f"fresh-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    comparison_dir = output_dir / "comparison"
    regression_dir = output_dir / "regression"
    runtime_dir = output_dir / "runtime"
    comparison_dir.mkdir(parents=True, exist_ok=True)
    regression_dir.mkdir(parents=True, exist_ok=True)

    evaluator = RAGEvaluator(evaluation_backend="fallback")
    eval_dataset = evaluator.load_eval_dataset(str(DATA_DIR / "eval_questions.json"))[:max_questions]

    results_a, summary_a, elapsed_a = run_pipeline(
        NaiveVectorRAG,
        DATA_DIR / "knowledge_base_v2.json",
        "a3_naive_v2",
        runtime_dir / "naive_v2",
        evaluator,
        eval_dataset,
    )
    results_b, summary_b, elapsed_b = run_pipeline(
        HybridRerankRAG,
        DATA_DIR / "knowledge_base_v2.json",
        "a3_hybrid_v2",
        runtime_dir / "hybrid_v2",
        evaluator,
        eval_dataset,
    )

    results_a.to_csv(comparison_dir / "pipeline_a_results.csv", index=False)
    results_b.to_csv(comparison_dir / "pipeline_b_results.csv", index=False)
    (comparison_dir / "pipeline_a_summary.json").write_text(
        json.dumps(summary_a, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (comparison_dir / "pipeline_b_summary.json").write_text(
        json.dumps(summary_b, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    tester = RegressionTester(
        eval_dataset_path=str(DATA_DIR / "eval_questions.json"),
        results_dir=str(runtime_dir / "regression"),
    )
    regression_result = tester.run_regression_test(
        pipeline_class=HybridRerankRAG,
        kb_v1_path=str(DATA_DIR / "knowledge_base_v1.json"),
        kb_v2_path=str(DATA_DIR / "knowledge_base_v2.json"),
        pipeline_kwargs={
            "persist_dir": str(runtime_dir / "regression_hybrid"),
            "collection_name": "a3_regression_hybrid",
        },
        max_questions=max_questions,
    )
    report_text = tester.generate_report_text(regression_result)
    (regression_dir / "regression_report.json").write_text(
        json.dumps(regression_result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (regression_dir / "regression_report.txt").write_text(report_text, encoding="utf-8")

    summary = {
        "checked_at_utc": utc_now(),
        "max_questions": max_questions,
        "output_dir": str(output_dir),
        "comparison": {
            "pipeline_a_mean": mean_score(summary_a),
            "pipeline_b_mean": mean_score(summary_b),
            "pipeline_a_elapsed_seconds": elapsed_a,
            "pipeline_b_elapsed_seconds": elapsed_b,
        },
        "regression": {
            "degraded": len(regression_result.get("degraded_questions", [])),
            "improved": len(regression_result.get("improved_questions", [])),
            "stable": len(regression_result.get("stable_questions", [])),
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["deterministic", "fresh"],
        default="deterministic",
    )
    parser.add_argument("--max-questions", type=int, default=12)
    args = parser.parse_args()

    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)

    if args.mode == "deterministic":
        result = deterministic_checks()
        output_path = EVIDENCE_DIR / "deterministic-checks.json"
    else:
        result = run_fresh(max_questions=args.max_questions)
        output_path = EVIDENCE_DIR / "latest-fresh-summary.json"

    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
