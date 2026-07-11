from __future__ import annotations

import csv
import threading
import gc
import json
import math
import random
import re
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any

import chromadb
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from langchain_core.documents import Document

from src.evaluation_engine import RAGEvaluator
from src.rag_pipelines import HybridRerankRAG, NaiveVectorRAG
from src.regression_tester import RegressionTester
from src.utils import (
    EMBEDDING_MODEL_NAME,
    LLM_MODEL_NAME,
    OLLAMA_BASE_URL,
    chunk_documents,
    create_chroma_collection,
)

st.set_page_config(page_title="RAG Quality Lab", layout="wide")

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
KB_V1_PATH = str(DATA_DIR / "knowledge_base_v1.json")
KB_V2_PATH = str(DATA_DIR / "knowledge_base_v2.json")
EVAL_DATASET_PATH = str(DATA_DIR / "eval_questions.json")
LARGE_KB_PATH = str(DATA_DIR / "large_knowledge_base.json")
LARGE_EVAL_DATASET_PATH = str(DATA_DIR / "large_eval_questions.json")
BENCHMARK_ARTIFACTS_DIR = PROJECT_ROOT / "benchmark_artifacts"
BENCHMARK_CHROMA_DIR = PROJECT_ROOT / "chroma_db_benchmarks"
SCALE_RESULTS_DIR = RESULTS_DIR / "scale_performance"
BENCHMARK_DOC_SIZES = [1000, 5000, 10000, 50000]
BENCHMARK_QUERY_COUNT = 100

PIPELINE_A_LABEL = "Pipeline A (Naive Vector)"
PIPELINE_B_LABEL = "Pipeline B (Hybrid+Rerank)"
COMPARISON_METRICS = [
    "faithfulness",
    "answer_relevancy",
    "context_precision",
    "context_recall",
    "answer_correctness",
]
PIPELINE_DISPLAY_NAMES = {
    PIPELINE_A_LABEL: "Pipeline A",
    PIPELINE_B_LABEL: "Pipeline B",
}
PIPELINE_CLASS_MAP = {
    PIPELINE_A_LABEL: NaiveVectorRAG,
    PIPELINE_B_LABEL: HybridRerankRAG,
}
PIPELINE_APP_CONFIG = {
    PIPELINE_A_LABEL: {
        "collection_name": "naive_rag_streamlit_eval",
        "persist_dir": str(PROJECT_ROOT / "chroma_db_app"),
    },
    PIPELINE_B_LABEL: {
        "collection_name": "hybrid_rag_streamlit_eval",
        "persist_dir": str(PROJECT_ROOT / "chroma_db_app"),
    },
}
COMPARISON_ARTIFACT_PATTERN = re.compile(
    r"pipeline_(?P<pipeline>[ab])_(?P<max_questions>\d+)_questions_(?P<timestamp>\d{8}_\d{6})_(?P<kind>results|summary)\.(?P<extension>csv|json)$"
)
REGRESSION_REPORT_PATTERN = re.compile(
    r"regression_report_(?P<timestamp>\d{8}_\d{6})\.json$"
)
INDEXING_BENCHMARK_PATTERN = re.compile(
    r"indexing_benchmark_(?P<timestamp>\d{8}_\d{6})\.csv$"
)
LATENCY_BENCHMARK_PATTERN = re.compile(
    r"latency_benchmark_(?P<kind>summary|raw)_(?P<timestamp>\d{8}_\d{6})\.csv$"
)


class BackgroundJobManager:
    def __init__(self) -> None:
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.lock = threading.Lock()
        self.jobs: dict[str, dict[str, Any]] = {}

    def submit(self, fn: Any, *args: Any, metadata: dict[str, Any] | None = None) -> str:
        job_id = f"background-job::{uuid.uuid4().hex}"
        future = self.executor.submit(fn, *args)
        with self.lock:
            payload = {
                "future": future,
                "submitted_at": time.time(),
            }
            if metadata:
                payload.update(metadata)
            self.jobs[job_id] = payload
        return job_id

    def get(self, job_id: str | None) -> dict[str, Any] | None:
        if not job_id:
            return None
        with self.lock:
            return self.jobs.get(job_id)

    def pop(self, job_id: str | None) -> dict[str, Any] | None:
        if not job_id:
            return None
        with self.lock:
            return self.jobs.pop(job_id, None)


@st.cache_resource
def get_pipeline_a():
    from src.rag_pipelines import NaiveVectorRAG

    return NaiveVectorRAG(
        knowledge_base_path="data/knowledge_base_v1.json",
        collection_name="naive_rag_app",
        persist_dir="./chroma_db_app",
    )


@st.cache_resource
def get_pipeline_b():
    from src.rag_pipelines import HybridRerankRAG

    return HybridRerankRAG(
        knowledge_base_path="data/knowledge_base_v1.json",
        collection_name="hybrid_rag_app",
        persist_dir="./chroma_db_app",
    )


@st.cache_resource
def get_evaluator() -> RAGEvaluator:
    return RAGEvaluator()


@st.cache_resource
def get_comparison_job_manager() -> BackgroundJobManager:
    return BackgroundJobManager()


@st.cache_resource
def get_regression_job_manager() -> BackgroundJobManager:
    return BackgroundJobManager()


@st.cache_resource
def get_scale_indexing_job_manager() -> BackgroundJobManager:
    return BackgroundJobManager()


@st.cache_resource
def get_scale_latency_job_manager() -> BackgroundJobManager:
    return BackgroundJobManager()


@st.cache_data
def load_eval_dataset() -> list[dict[str, Any]]:
    return get_evaluator().load_eval_dataset(EVAL_DATASET_PATH)


@st.cache_data
def load_large_eval_dataset() -> list[dict[str, Any]]:
    return get_evaluator().load_eval_dataset(LARGE_EVAL_DATASET_PATH)


@st.cache_data
def load_knowledge_base_records(json_path: str) -> list[dict[str, Any]]:
    return json.loads(Path(json_path).expanduser().read_text(encoding="utf-8"))


@st.cache_resource
def load_large_knowledge_base_records() -> list[dict[str, Any]]:
    return json.loads(Path(LARGE_KB_PATH).expanduser().read_text(encoding="utf-8"))


@st.cache_data
def compute_knowledge_base_diff() -> dict[str, Any]:
    v1_records = {item["id"]: item for item in load_knowledge_base_records(KB_V1_PATH)}
    v2_records = {item["id"]: item for item in load_knowledge_base_records(KB_V2_PATH)}

    added = [
        {
            "id": doc_id,
            "title": v2_records[doc_id]["title"],
            "content": v2_records[doc_id]["content"],
        }
        for doc_id in sorted(v2_records.keys() - v1_records.keys())
    ]
    removed = [
        {
            "id": doc_id,
            "title": v1_records[doc_id]["title"],
            "content": v1_records[doc_id]["content"],
        }
        for doc_id in sorted(v1_records.keys() - v2_records.keys())
    ]

    modified: list[dict[str, Any]] = []
    unchanged_count = 0
    for doc_id in sorted(v1_records.keys() & v2_records.keys()):
        before = v1_records[doc_id]
        after = v2_records[doc_id]
        title_changed = before["title"] != after["title"]
        content_changed = before["content"] != after["content"]
        if not title_changed and not content_changed:
            unchanged_count += 1
            continue
        modified.append(
            {
                "id": doc_id,
                "title_before": before["title"],
                "title_after": after["title"],
                "content_before": before["content"],
                "content_after": after["content"],
                "title_changed": title_changed,
                "content_changed": content_changed,
            }
        )

    return {
        "added": added,
        "removed": removed,
        "modified": modified,
        "unchanged_count": unchanged_count,
        "summary_text": build_kb_diff_summary_text(added, modified, removed, unchanged_count),
    }


def build_kb_diff_summary_text(
    added: list[dict[str, Any]],
    modified: list[dict[str, Any]],
    removed: list[dict[str, Any]],
    unchanged_count: int,
) -> str:
    parts: list[str] = []
    if added:
        parts.append(f"{len(added)} document(s) added")
    if modified:
        parts.append(f"{len(modified)} updated")
    if removed:
        parts.append(f"{len(removed)} removed")
    if unchanged_count:
        parts.append(f"{unchanged_count} unchanged")
    if not parts:
        return "No knowledge base differences detected."
    return "Knowledge base delta: " + ", ".join(parts) + "."


@st.cache_data
def compute_large_scale_dataset_stats() -> dict[str, Any]:
    records = load_large_knowledge_base_records()
    file_path = Path(LARGE_KB_PATH)
    return {
        "document_count": len(records),
        "total_chars": sum(len(str(item.get("content", ""))) for item in records),
        "file_size_mb": file_path.stat().st_size / (1024**2),
        "eval_question_count": len(load_large_eval_dataset()),
    }


def records_to_documents(records: list[dict[str, Any]]) -> list[Document]:
    documents: list[Document] = []
    for record in records:
        document_id = str(record["id"])
        documents.append(
            Document(
                id=document_id,
                page_content=str(record["content"]),
                metadata={"id": document_id, "title": str(record["title"])},
            )
        )
    return documents


def get_large_kb_subset_records(doc_count: int) -> list[dict[str, Any]]:
    return load_large_knowledge_base_records()[:doc_count]


def ensure_benchmark_subset_file(doc_count: int) -> str:
    output_dir = BENCHMARK_ARTIFACTS_DIR / "kb_subsets"
    output_dir.mkdir(parents=True, exist_ok=True)
    subset_path = output_dir / f"large_kb_first_{doc_count}.json"
    if not subset_path.exists():
        with subset_path.open("w", encoding="utf-8") as file:
            json.dump(get_large_kb_subset_records(doc_count), file, ensure_ascii=False)
    return str(subset_path)


@st.cache_resource
def get_benchmark_pipeline_a(doc_count: int) -> NaiveVectorRAG:
    return NaiveVectorRAG(
        knowledge_base_path=ensure_benchmark_subset_file(doc_count),
        collection_name=f"naive_scale_benchmark_{doc_count}",
        persist_dir=str(BENCHMARK_CHROMA_DIR / "naive"),
    )


@st.cache_resource
def get_benchmark_pipeline_b(doc_count: int) -> HybridRerankRAG:
    return HybridRerankRAG(
        knowledge_base_path=ensure_benchmark_subset_file(doc_count),
        collection_name=f"hybrid_scale_benchmark_{doc_count}",
        persist_dir=str(BENCHMARK_CHROMA_DIR / "hybrid"),
    )


def list_collection_counts(persist_dir: Path) -> dict[str, int]:
    if not persist_dir.exists():
        return {}

    client = chromadb.PersistentClient(path=str(persist_dir))
    counts: dict[str, int] = {}
    try:
        collections = client.list_collections()
    except Exception:
        return {}

    for collection in collections:
        collection_name = getattr(collection, "name", str(collection))
        try:
            counts[str(collection_name)] = int(collection.count())
        except Exception:
            continue
    return counts


def get_collection_vector_counts() -> dict[str, int]:
    tracked_dirs = {
        "app": PROJECT_ROOT / "chroma_db_app",
        "indexing": BENCHMARK_CHROMA_DIR / "indexing",
        "latency_a": BENCHMARK_CHROMA_DIR / "naive",
        "latency_b": BENCHMARK_CHROMA_DIR / "hybrid",
    }
    counts: dict[str, int] = {}
    for category, persist_dir in tracked_dirs.items():
        for collection_name, count in list_collection_counts(persist_dir).items():
            counts[f"{category}:{collection_name}"] = count
    return counts


def summarize_collection_counts(counts: dict[str, int]) -> str:
    if not counts:
        return "Build a pipeline once to populate the collections"

    category_totals = {
        "app": 0,
        "indexing": 0,
        "latency": 0,
    }
    for name, count in counts.items():
        if name.startswith("app:"):
            category_totals["app"] += count
        elif name.startswith("indexing:"):
            category_totals["indexing"] += count
        elif name.startswith("latency_a:") or name.startswith("latency_b:"):
            category_totals["latency"] += count

    parts = []
    if category_totals["app"]:
        parts.append(f"app: {category_totals['app']:,}")
    if category_totals["indexing"]:
        parts.append(f"indexing: {category_totals['indexing']:,}")
    if category_totals["latency"]:
        parts.append(f"latency: {category_totals['latency']:,}")
    return " • ".join(parts)


def count_historical_evaluation_rows() -> int:
    total_rows = 0
    if not RESULTS_DIR.exists():
        return 0

    for csv_path in RESULTS_DIR.rglob("*.csv"):
        try:
            with csv_path.open("r", encoding="utf-8", newline="") as file:
                reader = csv.reader(file)
                next(reader, None)
                total_rows += sum(1 for _ in reader)
        except Exception:
            continue
    return total_rows


def build_latency_summary(latencies_ms: dict[str, list[float]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for pipeline_name, values in latencies_ms.items():
        series = pd.Series(values, dtype="float64")
        rows.append(
            {
                "Pipeline": pipeline_name,
                "P50 (ms)": float(series.quantile(0.50)),
                "P95 (ms)": float(series.quantile(0.95)),
                "P99 (ms)": float(series.quantile(0.99)),
                "Mean (ms)": float(series.mean()),
            }
        )
    return pd.DataFrame(rows)


def build_indexing_benchmark_chart(results_df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=results_df["Documents"],
            y=results_df["Time (s)"],
            mode="lines+markers",
            name="Index build time",
            line={"color": "#1f77b4", "width": 3},
            marker={"size": 9},
        )
    )
    fig.update_layout(
        xaxis_title="Documents Indexed",
        yaxis_title="Time (seconds)",
        margin={"l": 20, "r": 20, "t": 30, "b": 20},
    )
    return fig


def build_latency_benchmark_chart(summary_df: pd.DataFrame) -> go.Figure:
    metric_names = ["P50 (ms)", "P95 (ms)", "P99 (ms)"]
    fig = go.Figure()
    colors = {"Pipeline A": "#1f77b4", "Pipeline B": "#ff7f0e"}
    for row in summary_df.to_dict("records"):
        fig.add_trace(
            go.Bar(
                name=row["Pipeline"],
                x=metric_names,
                y=[row[metric] for metric in metric_names],
                marker_color=colors.get(row["Pipeline"], "#666666"),
            )
        )
    fig.update_layout(
        barmode="group",
        yaxis_title="Latency (ms)",
        margin={"l": 20, "r": 20, "t": 30, "b": 20},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.05, "x": 0.0},
    )
    return fig


def build_indexing_scaling_summary(results_df: pd.DataFrame) -> str:
    seconds_per_1k = results_df["Seconds per 1K Docs"]
    low = float(seconds_per_1k.min())
    high = float(seconds_per_1k.max())
    return (
        "The indexing curve is close to linear when the seconds-per-1K-documents "
        f"stays in a narrow range. This run stayed between {low:.2f}s and {high:.2f}s per 1K documents."
    )


def persist_comparison_results(
    results_a: pd.DataFrame,
    results_b: pd.DataFrame,
    summary_a: dict[str, Any],
    summary_b: dict[str, Any],
    max_questions: int,
) -> None:
    output_dir = RESULTS_DIR / "dashboard_comparison"
    evaluator = get_evaluator()
    evaluator.save_results(
        results_a,
        summary_a,
        str(output_dir),
        f"pipeline_a_{max_questions}_questions",
    )
    evaluator.save_results(
        results_b,
        summary_b,
        str(output_dir),
        f"pipeline_b_{max_questions}_questions",
    )
    list_saved_comparison_runs.clear()
    load_saved_comparison_result.clear()


def persist_regression_results(
    tester: RegressionTester,
    pipeline_label: str,
    max_questions: int,
    baseline_df: pd.DataFrame,
    updated_df: pd.DataFrame,
    regression_result: dict[str, Any],
) -> None:
    output_dir = RESULTS_DIR / "dashboard_regression"
    safe_name = PIPELINE_DISPLAY_NAMES[pipeline_label].lower().replace(" ", "_")
    tester.evaluator.save_results(
        baseline_df,
        regression_result["baseline_summary"],
        str(output_dir),
        f"{safe_name}_baseline_{max_questions}_questions",
    )
    tester.evaluator.save_results(
        updated_df,
        regression_result["updated_summary"],
        str(output_dir),
        f"{safe_name}_updated_{max_questions}_questions",
    )
    tester.save_regression_report(
        regression_result,
        tester.generate_report_text(regression_result),
        str(output_dir),
    )
    list_saved_regression_runs.clear()
    load_saved_regression_result.clear()


def persist_indexing_benchmark_results(results_df: pd.DataFrame) -> None:
    output_dir = SCALE_RESULTS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_df.to_csv(
        output_dir / f"indexing_benchmark_{timestamp}.csv",
        index=False,
        encoding="utf-8",
    )
    list_saved_indexing_runs.clear()
    load_saved_indexing_result.clear()


def persist_latency_benchmark_results(
    summary_df: pd.DataFrame,
    latencies_ms: dict[str, list[float]],
) -> None:
    output_dir = SCALE_RESULTS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_df.to_csv(
        output_dir / f"latency_benchmark_summary_{timestamp}.csv",
        index=False,
        encoding="utf-8",
    )

    raw_rows: list[dict[str, Any]] = []
    for pipeline_name, values in latencies_ms.items():
        for run_index, latency_ms in enumerate(values, start=1):
            raw_rows.append(
                {
                    "Pipeline": pipeline_name,
                    "Run": run_index,
                    "Latency (ms)": latency_ms,
                }
            )
    pd.DataFrame(raw_rows).to_csv(
        output_dir / f"latency_benchmark_raw_{timestamp}.csv",
        index=False,
        encoding="utf-8",
    )
    list_saved_latency_runs.clear()
    load_saved_latency_result.clear()

def init_session_state() -> None:
    st.session_state.setdefault("interactive_result", None)
    st.session_state.setdefault("comparison_cache", {})
    st.session_state.setdefault("comparison_result_key", None)
    st.session_state.setdefault("comparison_limit_slider", 5)
    st.session_state.setdefault("comparison_active_job_id", None)
    st.session_state.setdefault("comparison_background_error", None)
    st.session_state.setdefault("comparison_background_notice", None)
    st.session_state.setdefault("regression_cache", {})
    st.session_state.setdefault("regression_result_key", None)
    st.session_state.setdefault("regression_active_job_id", None)
    st.session_state.setdefault("regression_background_error", None)
    st.session_state.setdefault("regression_background_notice", None)
    st.session_state.setdefault("scale_indexing_benchmark", None)
    st.session_state.setdefault("scale_indexing_benchmark_persisted", False)
    st.session_state.setdefault("scale_indexing_active_job_id", None)
    st.session_state.setdefault("scale_indexing_background_error", None)
    st.session_state.setdefault("scale_indexing_background_notice", None)
    st.session_state.setdefault("scale_latency_benchmark", None)
    st.session_state.setdefault("scale_latency_benchmark_persisted", False)
    st.session_state.setdefault("scale_latency_active_job_id", None)
    st.session_state.setdefault("scale_latency_background_error", None)
    st.session_state.setdefault("scale_latency_background_notice", None)


def get_pipeline_by_label(label: str) -> Any:
    if label == PIPELINE_A_LABEL:
        return get_pipeline_a()
    return get_pipeline_b()


def humanize_metric_name(metric: str) -> str:
    return metric.replace("_", " ").title()


def clip_score(value: Any) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(numeric) or math.isinf(numeric):
        return None
    return max(0.0, min(1.0, numeric))


def sigmoid_score(value: float) -> float:
    clipped = max(-50.0, min(50.0, float(value)))
    return 1.0 / (1.0 + math.exp(-clipped))


def shorten_text(text: str, limit: int = 280) -> str:
    stripped = " ".join(str(text).split())
    if len(stripped) <= limit:
        return stripped
    return stripped[: limit - 3] + "..."


def score_cell_style(value: Any) -> str:
    score = clip_score(value)
    if score is None:
        return ""
    if score >= 0.8:
        return "background-color: rgba(46, 160, 67, 0.22); color: #1b5e20;"
    if score >= 0.5:
        return "background-color: rgba(251, 191, 36, 0.25); color: #8a5a00;"
    return "background-color: rgba(220, 38, 38, 0.22); color: #8b0000;"


def diff_cell_style(value: Any) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return ""
    if math.isnan(numeric) or math.isinf(numeric):
        return ""
    if numeric > 0:
        return "background-color: rgba(46, 160, 67, 0.18); color: #1b5e20;"
    if numeric < 0:
        return "background-color: rgba(220, 38, 38, 0.18); color: #8b0000;"
    return ""


def format_score(value: Any) -> str:
    score = clip_score(value)
    if score is None:
        return "n/a"
    return f"{score:.3f}"


def parse_string_list(value: Any) -> list[str]:
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
    if value is None:
        return []
    return [str(value)]


def parse_saved_timestamp(timestamp: str) -> datetime:
    return datetime.strptime(timestamp, "%Y%m%d_%H%M%S")


def format_saved_timestamp(timestamp: str) -> str:
    return parse_saved_timestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S")


def parse_comparison_artifact_filename(filename: str) -> dict[str, Any] | None:
    match = COMPARISON_ARTIFACT_PATTERN.fullmatch(filename)
    if not match:
        return None

    parsed = match.groupdict()
    return {
        "pipeline": str(parsed["pipeline"]),
        "max_questions": int(parsed["max_questions"]),
        "timestamp": str(parsed["timestamp"]),
        "kind": str(parsed["kind"]),
    }


@st.cache_data(show_spinner=False)
def list_saved_comparison_runs() -> list[dict[str, Any]]:
    output_dir = RESULTS_DIR / "dashboard_comparison"
    if not output_dir.exists():
        return []

    artifacts_by_limit: dict[int, dict[str, dict[str, dict[str, str]]]] = {}
    for path in output_dir.iterdir():
        if not path.is_file():
            continue

        parsed = parse_comparison_artifact_filename(path.name)
        if parsed is None:
            continue

        max_questions = int(parsed["max_questions"])
        pipeline = str(parsed["pipeline"])
        timestamp = str(parsed["timestamp"])
        kind = str(parsed["kind"])

        limit_artifacts = artifacts_by_limit.setdefault(max_questions, {"a": {}, "b": {}})
        snapshot = limit_artifacts[pipeline].setdefault(timestamp, {})
        snapshot[f"{kind}_path"] = str(path)

    runs: list[dict[str, Any]] = []
    for max_questions, pipeline_artifacts in artifacts_by_limit.items():
        pipeline_a_runs = [
            {"timestamp": timestamp, **paths}
            for timestamp, paths in pipeline_artifacts["a"].items()
            if paths.get("results_path") and paths.get("summary_path")
        ]
        pipeline_b_runs = [
            {"timestamp": timestamp, **paths}
            for timestamp, paths in pipeline_artifacts["b"].items()
            if paths.get("results_path") and paths.get("summary_path")
        ]

        pipeline_a_runs.sort(key=lambda item: item["timestamp"], reverse=True)
        pipeline_b_runs.sort(key=lambda item: item["timestamp"], reverse=True)

        for index, (run_a, run_b) in enumerate(zip(pipeline_a_runs, pipeline_b_runs), start=1):
            latest_timestamp = max(str(run_a["timestamp"]), str(run_b["timestamp"]))
            runs.append(
                {
                    "run_id": f"comparison::{max_questions}::{index}",
                    "max_questions": max_questions,
                    "timestamp": latest_timestamp,
                    "label": (
                        f"{max_questions} questions • "
                        f"{format_saved_timestamp(latest_timestamp)}"
                    ),
                    "results_a_path": str(run_a["results_path"]),
                    "summary_a_path": str(run_a["summary_path"]),
                    "results_b_path": str(run_b["results_path"]),
                    "summary_b_path": str(run_b["summary_path"]),
                }
            )

    runs.sort(key=lambda item: item["timestamp"], reverse=True)
    return runs


def get_latest_saved_comparison_run(
    max_questions: int | None = None,
) -> dict[str, Any] | None:
    for run in list_saved_comparison_runs():
        if max_questions is None or run["max_questions"] == max_questions:
            return run
    return None


@st.cache_data(show_spinner=False)
def load_saved_comparison_result(run_id: str) -> dict[str, Any] | None:
    selected_run = next(
        (run for run in list_saved_comparison_runs() if run["run_id"] == run_id),
        None,
    )
    if selected_run is None:
        return None

    results_a = pd.read_csv(selected_run["results_a_path"], keep_default_na=False)
    results_b = pd.read_csv(selected_run["results_b_path"], keep_default_na=False)
    summary_a = json.loads(Path(selected_run["summary_a_path"]).read_text(encoding="utf-8"))
    summary_b = json.loads(Path(selected_run["summary_b_path"]).read_text(encoding="utf-8"))

    return {
        "max_questions": int(selected_run["max_questions"]),
        "results_a": results_a,
        "results_b": results_b,
        "summary_a": summary_a,
        "summary_b": summary_b,
        "conclusion": build_comparison_summary_text(summary_a, summary_b),
        "source": "saved",
        "saved_timestamp": str(selected_run["timestamp"]),
        "saved_label": str(selected_run["label"]),
    }


def activate_comparison_result(result: dict[str, Any]) -> None:
    cache_key = f"comparison::{result['max_questions']}"
    st.session_state["comparison_cache"][cache_key] = result
    st.session_state["comparison_result_key"] = cache_key


def load_saved_comparison_result_into_session(run_id: str) -> bool:
    result = load_saved_comparison_result(run_id)
    if result is None:
        return False
    activate_comparison_result(result)
    return True


def parse_regression_report_filename(filename: str) -> dict[str, Any] | None:
    match = REGRESSION_REPORT_PATTERN.fullmatch(filename)
    if not match:
        return None
    return {"timestamp": str(match.group("timestamp"))}


def parse_indexing_benchmark_filename(filename: str) -> dict[str, Any] | None:
    match = INDEXING_BENCHMARK_PATTERN.fullmatch(filename)
    if not match:
        return None
    return {"timestamp": str(match.group("timestamp"))}


def parse_latency_benchmark_filename(filename: str) -> dict[str, Any] | None:
    match = LATENCY_BENCHMARK_PATTERN.fullmatch(filename)
    if not match:
        return None
    return {
        "kind": str(match.group("kind")),
        "timestamp": str(match.group("timestamp")),
    }


@st.cache_data(show_spinner=False)
def list_saved_regression_runs() -> list[dict[str, Any]]:
    output_dir = RESULTS_DIR / "dashboard_regression"
    if not output_dir.exists():
        return []

    runs: list[dict[str, Any]] = []
    for path in output_dir.iterdir():
        if not path.is_file():
            continue

        parsed = parse_regression_report_filename(path.name)
        if parsed is None:
            continue

        payload = json.loads(path.read_text(encoding="utf-8"))
        pipeline_label = str(payload.get("pipeline_label", ""))
        if pipeline_label not in PIPELINE_CLASS_MAP:
            continue

        try:
            max_questions = int(payload.get("max_questions"))
        except (TypeError, ValueError):
            continue

        timestamp = str(parsed["timestamp"])
        runs.append(
            {
                "run_id": f"regression::{pipeline_label}::{timestamp}",
                "pipeline_label": pipeline_label,
                "max_questions": max_questions,
                "timestamp": timestamp,
                "label": (
                    f"{PIPELINE_DISPLAY_NAMES[pipeline_label]} • "
                    f"{max_questions} questions • "
                    f"{format_saved_timestamp(timestamp)}"
                ),
                "report_path": str(path),
            }
        )

    runs.sort(key=lambda item: item["timestamp"], reverse=True)
    return runs


def get_latest_saved_regression_run(
    pipeline_label: str | None = None,
    max_questions: int | None = None,
) -> dict[str, Any] | None:
    for run in list_saved_regression_runs():
        if pipeline_label is not None and run["pipeline_label"] != pipeline_label:
            continue
        if max_questions is not None and run["max_questions"] != max_questions:
            continue
        return run
    return None


@st.cache_data(show_spinner=False)
def load_saved_regression_result(run_id: str) -> dict[str, Any] | None:
    selected_run = next(
        (run for run in list_saved_regression_runs() if run["run_id"] == run_id),
        None,
    )
    if selected_run is None:
        return None

    result = json.loads(Path(selected_run["report_path"]).read_text(encoding="utf-8"))
    result["source"] = "saved"
    result["saved_timestamp"] = str(selected_run["timestamp"])
    result["saved_label"] = str(selected_run["label"])
    return result


def activate_regression_result(result: dict[str, Any]) -> None:
    cache_key = f"regression::{result['pipeline_label']}::{result['max_questions']}"
    st.session_state["regression_cache"][cache_key] = result
    st.session_state["regression_result_key"] = cache_key


def load_saved_regression_result_into_session(run_id: str) -> bool:
    result = load_saved_regression_result(run_id)
    if result is None:
        return False
    activate_regression_result(result)
    return True


@st.cache_data(show_spinner=False)
def list_saved_indexing_runs() -> list[dict[str, Any]]:
    if not SCALE_RESULTS_DIR.exists():
        return []

    runs: list[dict[str, Any]] = []
    for path in SCALE_RESULTS_DIR.iterdir():
        if not path.is_file():
            continue

        parsed = parse_indexing_benchmark_filename(path.name)
        if parsed is None:
            continue

        timestamp = str(parsed["timestamp"])
        runs.append(
            {
                "run_id": f"scale-indexing::{timestamp}",
                "timestamp": timestamp,
                "label": f"Indexing benchmark • {format_saved_timestamp(timestamp)}",
                "results_path": str(path),
            }
        )

    runs.sort(key=lambda item: item["timestamp"], reverse=True)
    return runs


def get_latest_saved_indexing_run() -> dict[str, Any] | None:
    runs = list_saved_indexing_runs()
    return runs[0] if runs else None


@st.cache_data(show_spinner=False)
def load_saved_indexing_result(run_id: str) -> dict[str, Any] | None:
    selected_run = next(
        (run for run in list_saved_indexing_runs() if run["run_id"] == run_id),
        None,
    )
    if selected_run is None:
        return None

    results_df = pd.read_csv(selected_run["results_path"], keep_default_na=False)
    return {
        "results_df": results_df,
        "summary_text": build_indexing_scaling_summary(results_df),
        "source": "saved",
        "saved_timestamp": str(selected_run["timestamp"]),
        "saved_label": str(selected_run["label"]),
    }


def activate_indexing_result(result: dict[str, Any]) -> None:
    st.session_state["scale_indexing_benchmark"] = result
    st.session_state["scale_indexing_benchmark_persisted"] = True


def load_saved_indexing_result_into_session(run_id: str) -> bool:
    result = load_saved_indexing_result(run_id)
    if result is None:
        return False
    activate_indexing_result(result)
    return True


@st.cache_data(show_spinner=False)
def list_saved_latency_runs() -> list[dict[str, Any]]:
    if not SCALE_RESULTS_DIR.exists():
        return []

    artifacts: dict[str, dict[str, str]] = {}
    for path in SCALE_RESULTS_DIR.iterdir():
        if not path.is_file():
            continue

        parsed = parse_latency_benchmark_filename(path.name)
        if parsed is None:
            continue

        timestamp = str(parsed["timestamp"])
        kind = str(parsed["kind"])
        snapshot = artifacts.setdefault(timestamp, {})
        snapshot[f"{kind}_path"] = str(path)

    runs: list[dict[str, Any]] = []
    for timestamp, paths in artifacts.items():
        if not paths.get("summary_path") or not paths.get("raw_path"):
            continue
        runs.append(
            {
                "run_id": f"scale-latency::{timestamp}",
                "timestamp": timestamp,
                "label": f"Latency benchmark • {format_saved_timestamp(timestamp)}",
                "summary_path": str(paths["summary_path"]),
                "raw_path": str(paths["raw_path"]),
            }
        )

    runs.sort(key=lambda item: item["timestamp"], reverse=True)
    return runs


def get_latest_saved_latency_run() -> dict[str, Any] | None:
    runs = list_saved_latency_runs()
    return runs[0] if runs else None


@st.cache_data(show_spinner=False)
def load_saved_latency_result(run_id: str) -> dict[str, Any] | None:
    selected_run = next(
        (run for run in list_saved_latency_runs() if run["run_id"] == run_id),
        None,
    )
    if selected_run is None:
        return None

    summary_df = pd.read_csv(selected_run["summary_path"], keep_default_na=False)
    raw_df = pd.read_csv(selected_run["raw_path"], keep_default_na=False)

    latencies_ms = {
        str(pipeline_name): [
            float(value) for value in group["Latency (ms)"].tolist()
        ]
        for pipeline_name, group in raw_df.groupby("Pipeline")
    }
    sample_count = min((len(values) for values in latencies_ms.values()), default=0)

    return {
        "summary_df": summary_df,
        "sample_count": sample_count,
        "latencies_ms": latencies_ms,
        "source": "saved",
        "saved_timestamp": str(selected_run["timestamp"]),
        "saved_label": str(selected_run["label"]),
    }


def activate_latency_result(result: dict[str, Any]) -> None:
    st.session_state["scale_latency_benchmark"] = result
    st.session_state["scale_latency_benchmark_persisted"] = True


def load_saved_latency_result_into_session(run_id: str) -> bool:
    result = load_saved_latency_result(run_id)
    if result is None:
        return False
    activate_latency_result(result)
    return True


def run_demo_progress(steps: list[tuple[int, str]], initial_text: str) -> None:
    progress = st.progress(0, text=initial_text)
    for percent, label in steps:
        time.sleep(0.18)
        progress.progress(percent, text=label)
    time.sleep(0.12)
    progress.empty()


def run_demo_comparison_progress() -> None:
    run_demo_progress(
        [
            (12, "Loading evaluation dataset..."),
            (34, "Checking cached pipeline resources..."),
            (61, "Running Pipeline A..."),
            (83, "Running Pipeline B..."),
            (100, "Rendering demo result..."),
        ],
        initial_text="Preparing demo evaluation...",
    )


def run_demo_regression_progress() -> None:
    run_demo_progress(
        [
            (15, "Loading knowledge base versions..."),
            (37, "Preparing baseline pipeline..."),
            (64, "Preparing updated pipeline..."),
            (86, "Comparing per-question deltas..."),
            (100, "Rendering regression report..."),
        ],
        initial_text="Preparing demo regression run...",
    )


def run_demo_indexing_progress() -> None:
    run_demo_progress(
        [
            (18, "Sampling 1K documents..."),
            (41, "Building first index shard..."),
            (67, "Scaling to larger corpus sizes..."),
            (89, "Compiling benchmark summary..."),
            (100, "Rendering indexing results..."),
        ],
        initial_text="Preparing demo indexing benchmark...",
    )


def run_demo_latency_progress() -> None:
    run_demo_progress(
        [
            (16, "Loading benchmark questions..."),
            (39, "Checking 50K-document indices..."),
            (68, "Measuring retrieval latency..."),
            (90, "Aggregating latency percentiles..."),
            (100, "Rendering latency results..."),
        ],
        initial_text="Preparing demo latency benchmark...",
    )


def sync_completed_background_job(
    *,
    job_id_key: str,
    error_key: str,
    notice_key: str,
    manager: BackgroundJobManager,
    activate_result: Any,
    success_notice: str,
) -> bool:
    job_id = st.session_state.get(job_id_key)
    if not job_id:
        return False

    job = manager.get(job_id)
    if job is None:
        st.session_state[job_id_key] = None
        return False

    future = job["future"]
    if not future.done():
        return False

    manager.pop(job_id)
    st.session_state[job_id_key] = None

    try:
        result = future.result()
    except Exception as exc:
        st.session_state[error_key] = (
            f"{type(exc).__name__}: {exc}"
        )
        st.session_state[notice_key] = None
        return True

    activate_result(result)
    st.session_state[error_key] = None
    st.session_state[notice_key] = success_notice
    return True


def start_background_job(
    *,
    job_id_key: str,
    error_key: str,
    notice_key: str,
    manager: BackgroundJobManager,
    fn: Any,
    args: tuple[Any, ...],
    metadata: dict[str, Any],
    pending_notice: str,
) -> bool:
    if st.session_state.get(job_id_key):
        return False

    job_id = manager.submit(fn, *args, metadata=metadata)
    st.session_state[job_id_key] = job_id
    st.session_state[error_key] = None
    st.session_state[notice_key] = pending_notice
    return True


def render_background_job_status_core(
    *,
    job_id_key: str,
    manager: BackgroundJobManager,
    status_message: str,
) -> None:
    job_id = st.session_state.get(job_id_key)
    if not job_id:
        return

    job = manager.get(job_id)
    if job is None:
        return

    elapsed_seconds = max(time.time() - float(job["submitted_at"]), 0.0)
    pseudo_progress = min(95, 20 + int(elapsed_seconds * 4))
    st.info(status_message)
    st.progress(
        pseudo_progress,
        text=(
            "Background run in progress... "
            f"{elapsed_seconds:.0f}s elapsed"
        ),
    )


def sync_completed_comparison_job() -> bool:
    return sync_completed_background_job(
        job_id_key="comparison_active_job_id",
        error_key="comparison_background_error",
        notice_key="comparison_background_notice",
        manager=get_comparison_job_manager(),
        activate_result=activate_comparison_result,
        success_notice="Background evaluation finished. Showing the latest real result.",
    )


def sync_completed_regression_job() -> bool:
    return sync_completed_background_job(
        job_id_key="regression_active_job_id",
        error_key="regression_background_error",
        notice_key="regression_background_notice",
        manager=get_regression_job_manager(),
        activate_result=activate_regression_result,
        success_notice="Background regression run finished. Showing the latest real result.",
    )


def sync_completed_indexing_job() -> bool:
    return sync_completed_background_job(
        job_id_key="scale_indexing_active_job_id",
        error_key="scale_indexing_background_error",
        notice_key="scale_indexing_background_notice",
        manager=get_scale_indexing_job_manager(),
        activate_result=activate_indexing_result,
        success_notice="Background indexing benchmark finished. Showing the latest real result.",
    )


def sync_completed_latency_job() -> bool:
    return sync_completed_background_job(
        job_id_key="scale_latency_active_job_id",
        error_key="scale_latency_background_error",
        notice_key="scale_latency_background_notice",
        manager=get_scale_latency_job_manager(),
        activate_result=activate_latency_result,
        success_notice="Background latency benchmark finished. Showing the latest real result.",
    )


def start_background_comparison_job(max_questions: int) -> bool:
    return start_background_job(
        job_id_key="comparison_active_job_id",
        error_key="comparison_background_error",
        notice_key="comparison_background_notice",
        manager=get_comparison_job_manager(),
        fn=compute_comparison_evaluation,
        args=(max_questions,),
        metadata={"max_questions": max_questions},
        pending_notice="Demo progress completed. The real evaluation is still running in the background.",
    )


def start_background_regression_job(pipeline_label: str, max_questions: int) -> bool:
    return start_background_job(
        job_id_key="regression_active_job_id",
        error_key="regression_background_error",
        notice_key="regression_background_notice",
        manager=get_regression_job_manager(),
        fn=compute_regression_analysis,
        args=(pipeline_label, max_questions),
        metadata={"pipeline_label": pipeline_label, "max_questions": max_questions},
        pending_notice="Demo progress completed. The real regression run is still running in the background.",
    )


def start_background_indexing_job() -> bool:
    return start_background_job(
        job_id_key="scale_indexing_active_job_id",
        error_key="scale_indexing_background_error",
        notice_key="scale_indexing_background_notice",
        manager=get_scale_indexing_job_manager(),
        fn=compute_indexing_benchmark,
        args=(),
        metadata={},
        pending_notice="Demo progress completed. The real indexing benchmark is still running in the background.",
    )


def start_background_latency_job() -> bool:
    return start_background_job(
        job_id_key="scale_latency_active_job_id",
        error_key="scale_latency_background_error",
        notice_key="scale_latency_background_notice",
        manager=get_scale_latency_job_manager(),
        fn=compute_latency_benchmark,
        args=(),
        metadata={},
        pending_notice="Demo progress completed. The real latency benchmark is still running in the background.",
    )


@st.fragment(run_every=2)
def render_comparison_background_status() -> None:
    if sync_completed_comparison_job():
        st.rerun()
    render_background_job_status_core(
        job_id_key="comparison_active_job_id",
        manager=get_comparison_job_manager(),
        status_message="Real evaluation is running in the background for the selected question limit.",
    )


@st.fragment(run_every=2)
def render_regression_background_status() -> None:
    if sync_completed_regression_job():
        st.rerun()
    render_background_job_status_core(
        job_id_key="regression_active_job_id",
        manager=get_regression_job_manager(),
        status_message="Real regression run is running in the background for the selected pipeline.",
    )


@st.fragment(run_every=2)
def render_indexing_background_status() -> None:
    if sync_completed_indexing_job():
        st.rerun()
    render_background_job_status_core(
        job_id_key="scale_indexing_active_job_id",
        manager=get_scale_indexing_job_manager(),
        status_message="Real indexing benchmark is running in the background.",
    )


@st.fragment(run_every=2)
def render_latency_background_status() -> None:
    if sync_completed_latency_job():
        st.rerun()
    render_background_job_status_core(
        job_id_key="scale_latency_active_job_id",
        manager=get_scale_latency_job_manager(),
        status_message="Real latency benchmark is running in the background.",
    )


def is_ollama_error(exc: Exception) -> bool:
    detail = f"{type(exc).__name__}: {exc}".lower()
    signals = [
        "ollama",
        "11434",
        "connection refused",
        "connecterror",
        "failed to connect",
        "max retries exceeded",
        "all connection attempts failed",
        "timed out",
    ]
    return any(signal in detail for signal in signals)


def render_runtime_error(exc: Exception, action_label: str) -> None:
    if is_ollama_error(exc):
        st.error(
            f"Unable to {action_label} because Ollama is not reachable. "
            f"Start Ollama at `{OLLAMA_BASE_URL}` and make sure the required models are available."
        )
        st.code("ollama serve", language="bash")
        return
    st.error(f"Failed to {action_label}: {type(exc).__name__}: {exc}")


def retrieve_docs_with_scores(pipeline: Any, query: str, k: int = 4) -> list[dict[str, Any]]:
    if isinstance(pipeline, NaiveVectorRAG):
        return retrieve_vector_docs_with_scores(pipeline, query, k)
    if isinstance(pipeline, HybridRerankRAG):
        return retrieve_hybrid_docs_with_scores(pipeline, query, k)

    return [
        {"document": doc, "score": None}
        for doc in pipeline.retrieve(query, k=k)
    ]


def retrieve_vector_docs_with_scores(
    pipeline: NaiveVectorRAG,
    query: str,
    k: int,
) -> list[dict[str, Any]]:
    try:
        results = pipeline.vectorstore.similarity_search_with_relevance_scores(query, k=k)
        return [
            {"document": document, "score": clip_score(score)}
            for document, score in results
        ]
    except Exception:
        results = pipeline.vectorstore.similarity_search_with_score(query, k=k)
        normalized: list[dict[str, Any]] = []
        for document, distance in results:
            try:
                relevance = 1.0 / (1.0 + max(float(distance), 0.0))
            except (TypeError, ValueError):
                relevance = None
            normalized.append({"document": document, "score": relevance})
        return normalized


def retrieve_hybrid_docs_with_scores(
    pipeline: HybridRerankRAG,
    query: str,
    k: int,
) -> list[dict[str, Any]]:
    dense_docs = pipeline.vectorstore.similarity_search(query, k=10)
    bm25_docs = (
        pipeline.bm25.get_top_n(query.lower().split(), pipeline.chunks, n=10)
        if pipeline.bm25 is not None
        else []
    )

    merged_docs: list[Any] = []
    seen_contents: set[str] = set()
    for doc in dense_docs + bm25_docs:
        if doc.page_content in seen_contents:
            continue
        seen_contents.add(doc.page_content)
        merged_docs.append(doc)

    if not merged_docs:
        return []

    pairs = [(query, doc.page_content) for doc in merged_docs]
    scores = pipeline.reranker.predict(pairs)
    ranked_docs = sorted(
        zip(merged_docs, scores, strict=True),
        key=lambda item: float(item[1]),
        reverse=True,
    )

    return [
        {"document": doc, "score": sigmoid_score(float(score))}
        for doc, score in ranked_docs[:k]
    ]


def run_interactive_query(query: str, pipeline_label: str) -> dict[str, Any]:
    pipeline = get_pipeline_by_label(pipeline_label)
    retrieved_items = retrieve_docs_with_scores(pipeline, query, k=4)
    context_docs = [item["document"] for item in retrieved_items]
    answer = pipeline.generate(query, context_docs)

    rendered_docs = []
    for item in retrieved_items:
        document = item["document"]
        rendered_docs.append(
            {
                "title": str(document.metadata.get("title", "Untitled document")),
                "doc_id": str(document.metadata.get("id", document.id or "")),
                "content": str(document.page_content),
                "score": item.get("score"),
            }
        )

    return {
        "query": query,
        "pipeline_label": pipeline_label,
        "answer": answer,
        "docs": rendered_docs,
    }


def build_comparison_summary_text(
    summary_a: dict[str, dict[str, float | None]],
    summary_b: dict[str, dict[str, float | None]],
) -> str:
    wins_a: list[str] = []
    wins_b: list[str] = []
    tied: list[str] = []
    tolerance = 0.02

    for metric in COMPARISON_METRICS:
        mean_a = summary_a.get(metric, {}).get("mean")
        mean_b = summary_b.get(metric, {}).get("mean")
        if mean_a is None or mean_b is None:
            continue
        if mean_a - mean_b > tolerance:
            wins_a.append(humanize_metric_name(metric))
        elif mean_b - mean_a > tolerance:
            wins_b.append(humanize_metric_name(metric))
        else:
            tied.append(humanize_metric_name(metric))

    avg_a = average_means(summary_a)
    avg_b = average_means(summary_b)
    if avg_a is None or avg_b is None:
        headline = "Overall ranking is unavailable because the mean scores are incomplete."
    elif avg_a > avg_b + tolerance:
        headline = f"Pipeline A leads overall with an average score of {avg_a:.3f} versus {avg_b:.3f}."
    elif avg_b > avg_a + tolerance:
        headline = f"Pipeline B leads overall with an average score of {avg_b:.3f} versus {avg_a:.3f}."
    else:
        headline = f"Both pipelines are broadly comparable overall ({avg_a:.3f} vs {avg_b:.3f})."

    details: list[str] = []
    if wins_a:
        details.append("Pipeline A is stronger on " + ", ".join(wins_a) + ".")
    if wins_b:
        details.append("Pipeline B is stronger on " + ", ".join(wins_b) + ".")
    if tied:
        details.append("The closest metrics are " + ", ".join(tied) + ".")
    return " ".join([headline, *details]).strip()


def average_means(summary: dict[str, dict[str, float | None]]) -> float | None:
    values = [
        summary.get(metric, {}).get("mean")
        for metric in COMPARISON_METRICS
        if summary.get(metric, {}).get("mean") is not None
    ]
    if not values:
        return None
    return float(sum(values) / len(values))


def build_radar_chart(
    summary_a: dict[str, dict[str, float | None]],
    summary_b: dict[str, dict[str, float | None]],
) -> go.Figure:
    labels = [humanize_metric_name(metric) for metric in COMPARISON_METRICS]
    scores_a = [summary_a.get(metric, {}).get("mean") or 0.0 for metric in COMPARISON_METRICS]
    scores_b = [summary_b.get(metric, {}).get("mean") or 0.0 for metric in COMPARISON_METRICS]

    fig = go.Figure()
    fig.add_trace(
        go.Scatterpolar(
            r=scores_a + [scores_a[0]],
            theta=labels + [labels[0]],
            fill="toself",
            name="Pipeline A",
            line={"color": "#1f77b4", "width": 2},
            fillcolor="rgba(31, 119, 180, 0.25)",
        )
    )
    fig.add_trace(
        go.Scatterpolar(
            r=scores_b + [scores_b[0]],
            theta=labels + [labels[0]],
            fill="toself",
            name="Pipeline B",
            line={"color": "#ff7f0e", "width": 2},
            fillcolor="rgba(255, 127, 14, 0.25)",
        )
    )
    fig.update_layout(
        polar={"radialaxis": {"visible": True, "range": [0, 1]}},
        margin={"l": 20, "r": 20, "t": 30, "b": 20},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.05, "x": 0.0},
    )
    return fig


def build_grouped_bar_chart(
    summary_a: dict[str, dict[str, float | None]],
    summary_b: dict[str, dict[str, float | None]],
) -> go.Figure:
    labels = [humanize_metric_name(metric) for metric in COMPARISON_METRICS]
    scores_a = [summary_a.get(metric, {}).get("mean") or 0.0 for metric in COMPARISON_METRICS]
    scores_b = [summary_b.get(metric, {}).get("mean") or 0.0 for metric in COMPARISON_METRICS]

    fig = go.Figure()
    fig.add_trace(go.Bar(name="Pipeline A", x=labels, y=scores_a, marker_color="#1f77b4"))
    fig.add_trace(go.Bar(name="Pipeline B", x=labels, y=scores_b, marker_color="#ff7f0e"))
    fig.update_layout(
        barmode="group",
        yaxis={"range": [0, 1], "title": "Mean Score"},
        margin={"l": 20, "r": 20, "t": 30, "b": 20},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.05, "x": 0.0},
    )
    return fig


def build_comparison_table(results_a: pd.DataFrame, results_b: pd.DataFrame) -> pd.DataFrame:
    merged = results_a[["question", *COMPARISON_METRICS]].merge(
        results_b[["question", *COMPARISON_METRICS]],
        on="question",
        suffixes=("_a", "_b"),
    )

    renamed_columns = {"question": "Question"}
    for metric in COMPARISON_METRICS:
        label = humanize_metric_name(metric)
        renamed_columns[f"{metric}_a"] = f"Pipeline A | {label}"
        renamed_columns[f"{metric}_b"] = f"Pipeline B | {label}"

    return merged.rename(columns=renamed_columns)


def build_overall_regression_table(regression_result: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for metric, values in regression_result["overall_diff"].items():
        rows.append(
            {
                "Metric": humanize_metric_name(metric),
                "Baseline": values.get("baseline"),
                "Updated": values.get("updated"),
                "Diff": values.get("diff"),
            }
        )
    return pd.DataFrame(rows)


def build_regression_detail_rows(
    baseline_df: pd.DataFrame,
    updated_df: pd.DataFrame,
    per_question_diff: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    baseline_by_question = {
        str(row["question"]): row for row in baseline_df.to_dict("records")
    }
    updated_by_question = {
        str(row["question"]): row for row in updated_df.to_dict("records")
    }

    detail_rows: list[dict[str, Any]] = []
    for item in per_question_diff:
        question = item["question"]
        baseline_row = baseline_by_question[question]
        updated_row = updated_by_question[question]
        detail_rows.append(
            {
                **item,
                "baseline_answer": str(baseline_row.get("answer", "")),
                "updated_answer": str(updated_row.get("answer", "")),
                "baseline_contexts": parse_string_list(baseline_row.get("contexts")),
                "updated_contexts": parse_string_list(updated_row.get("contexts")),
            }
        )
    return detail_rows


def compute_comparison_evaluation(max_questions: int) -> dict[str, Any]:
    evaluator = get_evaluator()
    dataset = load_eval_dataset()[:max_questions]
    if not dataset:
        raise ValueError("Evaluation dataset is empty.")

    pipeline_a = get_pipeline_a()
    results_a = evaluator.run_pipeline_evaluation(pipeline_a, dataset)
    summary_a = evaluator.compute_summary(results_a)

    pipeline_b = get_pipeline_b()
    results_b = evaluator.run_pipeline_evaluation(pipeline_b, dataset)
    summary_b = evaluator.compute_summary(results_b)
    persist_comparison_results(results_a, results_b, summary_a, summary_b, max_questions)

    return {
        "max_questions": max_questions,
        "results_a": results_a,
        "results_b": results_b,
        "summary_a": summary_a,
        "summary_b": summary_b,
        "conclusion": build_comparison_summary_text(summary_a, summary_b),
        "source": "fresh",
        "completed_at": datetime.now().strftime("%Y%m%d_%H%M%S"),
    }


def compute_regression_analysis(pipeline_label: str, max_questions: int) -> dict[str, Any]:
    pipeline_class = PIPELINE_CLASS_MAP[pipeline_label]
    pipeline_config = PIPELINE_APP_CONFIG[pipeline_label]
    tester = RegressionTester(eval_dataset_path=EVAL_DATASET_PATH, results_dir=str(RESULTS_DIR))
    eval_dataset = tester.evaluator.load_eval_dataset(EVAL_DATASET_PATH)[:max_questions]
    if not eval_dataset:
        raise ValueError("Evaluation dataset is empty.")

    run_id = tester._build_run_id(pipeline_class)
    baseline_df = tester._run_single_version(
        pipeline_class=pipeline_class,
        knowledge_base_path=KB_V1_PATH,
        version_tag="v1",
        run_id=run_id,
        eval_dataset=eval_dataset,
        pipeline_kwargs=pipeline_config,
    )
    updated_df = tester._run_single_version(
        pipeline_class=pipeline_class,
        knowledge_base_path=KB_V2_PATH,
        version_tag="v2",
        run_id=run_id,
        eval_dataset=eval_dataset,
        pipeline_kwargs=pipeline_config,
    )

    per_question_diff = tester._build_per_question_diff(baseline_df, updated_df)
    overall_diff = tester._build_overall_diff(baseline_df, updated_df)
    degraded_questions = [
        item["question"] for item in per_question_diff if item["status"] == "degraded"
    ]
    improved_questions = [
        item["question"] for item in per_question_diff if item["status"] == "improved"
    ]
    stable_questions = [
        item["question"] for item in per_question_diff if item["status"] == "stable"
    ]
    result = {
        "pipeline_label": pipeline_label,
        "max_questions": max_questions,
        "baseline_summary": tester.evaluator.compute_summary(baseline_df),
        "updated_summary": tester.evaluator.compute_summary(updated_df),
        "overall_diff": overall_diff,
        "per_question_diff": per_question_diff,
        "degraded_questions": degraded_questions,
        "improved_questions": improved_questions,
        "stable_questions": stable_questions,
        "detail_rows": build_regression_detail_rows(baseline_df, updated_df, per_question_diff),
        "knowledge_base_diff": compute_knowledge_base_diff(),
    }
    persist_regression_results(
        tester=tester,
        pipeline_label=pipeline_label,
        max_questions=max_questions,
        baseline_df=baseline_df,
        updated_df=updated_df,
        regression_result=result,
    )
    result["source"] = "fresh"
    result["completed_at"] = datetime.now().strftime("%Y%m%d_%H%M%S")
    return result


def compute_indexing_benchmark() -> dict[str, Any]:
    results: list[dict[str, Any]] = []

    for doc_count in BENCHMARK_DOC_SIZES:
        subset_records = get_large_kb_subset_records(doc_count)
        start_time = time.perf_counter()
        documents = records_to_documents(subset_records)
        chunks = chunk_documents(documents, chunk_size=512, chunk_overlap=64)
        vectorstore = create_chroma_collection(
            documents=chunks,
            collection_name=f"scale_index_benchmark_{doc_count}",
            persist_directory=str(BENCHMARK_CHROMA_DIR / "indexing"),
        )
        elapsed_seconds = time.perf_counter() - start_time
        results.append(
            {
                "Documents": doc_count,
                "Vectors Indexed": len(chunks),
                "Time (s)": elapsed_seconds,
                "Seconds per 1K Docs": elapsed_seconds / (doc_count / 1000),
            }
        )
        del documents
        del chunks
        del vectorstore
        gc.collect()

    results_df = pd.DataFrame(results)
    persist_indexing_benchmark_results(results_df)
    return {
        "results_df": results_df,
        "summary_text": build_indexing_scaling_summary(results_df),
        "source": "fresh",
        "completed_at": datetime.now().strftime("%Y%m%d_%H%M%S"),
    }


def compute_latency_benchmark() -> dict[str, Any]:
    eval_dataset = load_large_eval_dataset()
    sample_count = min(BENCHMARK_QUERY_COUNT, len(eval_dataset))
    sampled_queries = [
        item["question"] for item in random.Random(42).sample(eval_dataset, sample_count)
    ]

    pipeline_a = get_benchmark_pipeline_a(50000)
    pipeline_b = get_benchmark_pipeline_b(50000)

    for warmup_query in sampled_queries[:3]:
        pipeline_a.retrieve(warmup_query, k=4)
        pipeline_b.retrieve(warmup_query, k=4)

    latencies_ms = {"Pipeline A": [], "Pipeline B": []}
    for pipeline_name, pipeline in (
        ("Pipeline A", pipeline_a),
        ("Pipeline B", pipeline_b),
    ):
        for query in sampled_queries:
            start_time = time.perf_counter()
            pipeline.retrieve(query, k=4)
            latencies_ms[pipeline_name].append((time.perf_counter() - start_time) * 1000)

    summary_df = build_latency_summary(latencies_ms)
    persist_latency_benchmark_results(summary_df, latencies_ms)
    return {
        "summary_df": summary_df,
        "sample_count": sample_count,
        "latencies_ms": latencies_ms,
        "source": "fresh",
        "completed_at": datetime.now().strftime("%Y%m%d_%H%M%S"),
    }


def render_sidebar() -> str:
    st.sidebar.header("Navigation")
    mode = st.sidebar.radio(
        "Select Mode",
        options=[
            "Interactive Query",
            "Pipeline A/B Comparison",
            "Regression Test",
            "Scale & Performance",
        ],
    )

    st.sidebar.divider()
    st.sidebar.subheader("System Info")
    st.sidebar.caption(f"LLM: `{LLM_MODEL_NAME}`")
    st.sidebar.caption(f"Embeddings: `{EMBEDDING_MODEL_NAME}`")
    st.sidebar.caption("Knowledge Base: `v1` for query/eval, `v1 vs v2` for regression")
    st.sidebar.caption("Evaluation Set: `data/eval_questions.json`")
    st.sidebar.caption(f"Ollama Endpoint: `{OLLAMA_BASE_URL}`")
    return mode


def render_interactive_query_tab() -> None:
    st.subheader("Interactive Query")
    st.caption("Ask a question, choose a retrieval pipeline, and inspect the retrieved evidence beside the generated answer.")

    query = st.text_input(
        "Question",
        placeholder="Example: Why is regression testing important for RAG systems?",
    )
    pipeline_label = st.selectbox(
        "Pipeline",
        options=[PIPELINE_A_LABEL, PIPELINE_B_LABEL],
    )

    if st.button("Run Query", type="primary", use_container_width=False):
        if not query.strip():
            st.warning("Enter a question before running the query.")
        else:
            with st.spinner(f"Running {PIPELINE_DISPLAY_NAMES[pipeline_label]}..."):
                try:
                    st.session_state["interactive_result"] = run_interactive_query(
                        query=query.strip(),
                        pipeline_label=pipeline_label,
                    )
                except Exception as exc:
                    st.session_state["interactive_result"] = None
                    render_runtime_error(exc, "run the query")

    result = st.session_state.get("interactive_result")
    if not result:
        st.info("No query has been run in this session yet.")
        return

    docs_col, answer_col = st.columns([1.15, 1.0], gap="large")

    with docs_col:
        st.markdown("#### Retrieved Documents")
        if result["docs"]:
            for index, item in enumerate(result["docs"], start=1):
                with st.container(border=True):
                    st.markdown(f"**{index}. {item['title']}**")
                    if item["doc_id"]:
                        st.caption(f"Doc ID: `{item['doc_id']}`")
                    if item["score"] is not None:
                        st.progress(
                            float(item["score"]),
                            text=f"Relevance score: {float(item['score']):.3f}",
                        )
                    st.write(shorten_text(item["content"], limit=360))
        else:
            st.warning("No documents were retrieved for this question.")

    with answer_col:
        st.markdown("#### Generated Answer")
        st.write(result["answer"] or "No answer generated.")


def render_comparison_tab() -> None:
    st.subheader("Pipeline A/B Comparison")
    st.caption("Run the same evaluation set through both pipelines and compare their average quality across five core RAG metrics.")

    sync_completed_comparison_job()

    latest_saved_run = get_latest_saved_comparison_run()

    max_questions = st.slider(
        "Question Limit",
        min_value=2,
        max_value=12,
        key="comparison_limit_slider",
    )
    cache_key = f"comparison::{max_questions}"

    action_col, load_col = st.columns(2)
    with action_col:
        run_clicked = st.button("Run Evaluation", type="primary")
    with load_col:
        load_saved_clicked = st.button(
            "Load Latest Saved Result",
            disabled=latest_saved_run is None,
        )

    if latest_saved_run is not None:
        st.caption(f"Latest saved run: {latest_saved_run['label']}")

    if load_saved_clicked and latest_saved_run is not None:
        if load_saved_comparison_result_into_session(latest_saved_run["run_id"]):
            st.rerun()
        else:
            st.warning("The latest saved comparison result could not be loaded from disk.")

    if run_clicked:
        if st.session_state.get("comparison_active_job_id"):
            st.session_state["comparison_background_notice"] = (
                "A background evaluation is already running. Waiting for that run to finish."
            )
            run_demo_comparison_progress()
            saved_run_for_limit = get_latest_saved_comparison_run(max_questions=max_questions)
            if saved_run_for_limit is not None:
                load_saved_comparison_result_into_session(saved_run_for_limit["run_id"])
            st.rerun()

        try:
            started = start_background_comparison_job(max_questions)
        except Exception as exc:
            render_runtime_error(exc, "start the background evaluation")
        else:
            if started:
                run_demo_comparison_progress()
                saved_run_for_limit = get_latest_saved_comparison_run(max_questions=max_questions)
                if saved_run_for_limit is not None:
                    load_saved_comparison_result_into_session(saved_run_for_limit["run_id"])
                st.rerun()

    result_key = st.session_state.get("comparison_result_key")

    background_error = st.session_state.get("comparison_background_error")
    if background_error:
        st.error(f"Background evaluation failed: {background_error}")

    background_notice = st.session_state.get("comparison_background_notice")
    if background_notice:
        st.caption(background_notice)

    if st.session_state.get("comparison_active_job_id"):
        render_comparison_background_status()

    if not result_key:
        st.info("Run an evaluation to populate the comparison charts and score table.")
        return

    result = st.session_state["comparison_cache"][result_key]
    if result["max_questions"] != max_questions and cache_key in st.session_state["comparison_cache"]:
        result = st.session_state["comparison_cache"][cache_key]
        st.session_state["comparison_result_key"] = cache_key
    elif result["max_questions"] != max_questions:
        saved_run_for_limit = get_latest_saved_comparison_run(max_questions=max_questions)
        if saved_run_for_limit and load_saved_comparison_result_into_session(saved_run_for_limit["run_id"]):
            result = st.session_state["comparison_cache"][cache_key]
            st.session_state["comparison_result_key"] = cache_key
        elif st.session_state.get("comparison_active_job_id"):
            st.caption(
                "Showing the last available result while the new background run is still processing."
            )
        else:
            st.caption(
                "No cached or saved comparison exists for the current question limit yet. "
                "Showing the last available result instead."
            )

    if result.get("source") == "saved":
        st.caption(
            "Showing saved evaluation results from disk"
            f" ({format_saved_timestamp(result['saved_timestamp'])})."
        )
    elif result.get("source") == "fresh" and result.get("completed_at"):
        st.caption(
            "Showing the latest real evaluation result"
            f" ({format_saved_timestamp(result['completed_at'])})."
        )
    elif cache_key in st.session_state["comparison_cache"]:
        st.caption("Showing session-cached evaluation results for the current limit.")

    chart_left, chart_right = st.columns(2, gap="large")
    with chart_left:
        st.markdown("#### Radar Chart")
        st.plotly_chart(
            build_radar_chart(result["summary_a"], result["summary_b"]),
            use_container_width=True,
        )
    with chart_right:
        st.markdown("#### Mean Score Bars")
        st.plotly_chart(
            build_grouped_bar_chart(result["summary_a"], result["summary_b"]),
            use_container_width=True,
        )

    st.markdown("#### Detailed Per-Question Scores")
    comparison_table = build_comparison_table(result["results_a"], result["results_b"])
    metric_columns = [column for column in comparison_table.columns if column != "Question"]
    styled_table = comparison_table.style.format(
        {column: "{:.3f}" for column in metric_columns},
        na_rep="n/a",
    ).map(score_cell_style, subset=metric_columns)
    st.dataframe(styled_table, use_container_width=True, hide_index=True)

    st.markdown("#### Conclusion")
    st.write(result["conclusion"])


def render_regression_tab() -> None:
    st.subheader("Regression Test")
    st.caption("Compare answer quality before and after the knowledge base update, then inspect which questions improved, degraded, or stayed stable.")

    sync_completed_regression_job()

    pipeline_label = st.selectbox(
        "Pipeline",
        options=[PIPELINE_A_LABEL, PIPELINE_B_LABEL],
        key="regression_pipeline_select",
    )
    max_questions = st.slider(
        "Question Limit",
        min_value=2,
        max_value=12,
        value=5,
        key="regression_limit_slider",
    )
    cache_key = f"regression::{pipeline_label}::{max_questions}"
    saved_run_for_selection = get_latest_saved_regression_run(
        pipeline_label=pipeline_label,
        max_questions=max_questions,
    )
    latest_saved_run = saved_run_for_selection or get_latest_saved_regression_run()

    action_col, load_col = st.columns(2)
    with action_col:
        run_clicked = st.button("Run Regression Test", type="primary")
    with load_col:
        load_saved_clicked = st.button(
            "Load Latest Saved Result",
            key="load_latest_regression_result_button",
            disabled=latest_saved_run is None,
        )

    if latest_saved_run is not None:
        st.caption(f"Latest saved run: {latest_saved_run['label']}")

    if load_saved_clicked and latest_saved_run is not None:
        if load_saved_regression_result_into_session(latest_saved_run["run_id"]):
            st.rerun()
        else:
            st.warning("The latest saved regression result could not be loaded from disk.")

    if run_clicked:
        if st.session_state.get("regression_active_job_id"):
            st.session_state["regression_background_notice"] = (
                "A background regression run is already running. Waiting for that run to finish."
            )
            run_demo_regression_progress()
            target_run = saved_run_for_selection or latest_saved_run
            if target_run is not None:
                load_saved_regression_result_into_session(target_run["run_id"])
            st.rerun()

        try:
            started = start_background_regression_job(pipeline_label, max_questions)
        except Exception as exc:
            render_runtime_error(exc, "start the background regression run")
        else:
            if started:
                run_demo_regression_progress()
                target_run = saved_run_for_selection or latest_saved_run
                if target_run is not None:
                    load_saved_regression_result_into_session(target_run["run_id"])
                st.rerun()

    result_key = st.session_state.get("regression_result_key")

    background_error = st.session_state.get("regression_background_error")
    if background_error:
        st.error(f"Background regression run failed: {background_error}")

    background_notice = st.session_state.get("regression_background_notice")
    if background_notice:
        st.caption(background_notice)

    if st.session_state.get("regression_active_job_id"):
        render_regression_background_status()

    if not result_key:
        st.info("Run a regression test to compare knowledge base V1 and V2.")
        return

    result = st.session_state["regression_cache"][result_key]
    if (
        result["pipeline_label"] != pipeline_label or result["max_questions"] != max_questions
    ) and cache_key in st.session_state["regression_cache"]:
        result = st.session_state["regression_cache"][cache_key]
        st.session_state["regression_result_key"] = cache_key
    elif result["pipeline_label"] != pipeline_label or result["max_questions"] != max_questions:
        if saved_run_for_selection and load_saved_regression_result_into_session(saved_run_for_selection["run_id"]):
            result = st.session_state["regression_cache"][cache_key]
            st.session_state["regression_result_key"] = cache_key
        elif st.session_state.get("regression_active_job_id"):
            st.caption(
                "Showing the last available regression result while the new background run is still processing."
            )
        else:
            st.caption(
                "No cached or saved regression result exists for the current selection. "
                "Showing the last available result instead."
            )

    if result.get("source") == "saved":
        st.caption(
            "Showing saved regression results from disk"
            f" ({format_saved_timestamp(result['saved_timestamp'])})."
        )
    elif result.get("source") == "fresh" and result.get("completed_at"):
        st.caption(
            "Showing the latest real regression result"
            f" ({format_saved_timestamp(result['completed_at'])})."
        )
    else:
        st.caption("Showing session-cached regression results for the current selection.")

    st.markdown("#### Overall Metric Changes")
    overall_table = build_overall_regression_table(result)
    styled_overall = overall_table.style.format(
        {"Baseline": "{:.3f}", "Updated": "{:.3f}", "Diff": "{:+.3f}"},
        na_rep="n/a",
    ).map(diff_cell_style, subset=["Diff"])
    st.dataframe(styled_overall, use_container_width=True, hide_index=True)

    improved_count = sum(1 for item in result["detail_rows"] if item["status"] == "improved")
    degraded_count = sum(1 for item in result["detail_rows"] if item["status"] == "degraded")
    stable_count = sum(1 for item in result["detail_rows"] if item["status"] == "stable")
    metric_col1, metric_col2, metric_col3 = st.columns(3)
    metric_col1.metric("Improved", improved_count)
    metric_col2.metric("Degraded", degraded_count)
    metric_col3.metric("Stable", stable_count)

    st.markdown("#### Per-Question Status Cards")
    status_priority = {"degraded": 0, "improved": 1, "stable": 2}
    for item in sorted(
        result["detail_rows"],
        key=lambda row: (status_priority.get(row["status"], 3), row["question"]),
    ):
        if item["status"] == "improved":
            icon = "🟢"
            status_label = "Improved"
        elif item["status"] == "degraded":
            icon = "🔴"
            status_label = "Degraded"
        else:
            icon = "⚪"
            status_label = "Stable"

        with st.expander(f"{icon} {status_label} | {item['question']}"):
            score_rows = []
            for metric in COMPARISON_METRICS:
                score_rows.append(
                    {
                        "Metric": humanize_metric_name(metric),
                        "V1": item["baseline_scores"].get(metric),
                        "V2": item["updated_scores"].get(metric),
                        "Diff": item["diff"].get(metric),
                    }
                )
            score_table = pd.DataFrame(score_rows)
            styled_score_table = score_table.style.format(
                {"V1": "{:.3f}", "V2": "{:.3f}", "Diff": "{:+.3f}"},
                na_rep="n/a",
            ).map(diff_cell_style, subset=["Diff"])
            st.dataframe(styled_score_table, use_container_width=True, hide_index=True)

            answer_col1, answer_col2 = st.columns(2, gap="large")
            with answer_col1:
                st.markdown("**V1 Answer**")
                st.write(item["baseline_answer"] or "No answer generated.")
                st.caption(
                    "Retrieved doc IDs: "
                    + (", ".join(item["baseline_retrieved_doc_ids"]) or "None")
                )
            with answer_col2:
                st.markdown("**V2 Answer**")
                st.write(item["updated_answer"] or "No answer generated.")
                st.caption(
                    "Retrieved doc IDs: "
                    + (", ".join(item["updated_retrieved_doc_ids"]) or "None")
                )

    st.markdown("#### Change Analysis")
    kb_diff = result["knowledge_base_diff"]
    st.write(kb_diff["summary_text"])
    diff_metric_col1, diff_metric_col2, diff_metric_col3, diff_metric_col4 = st.columns(4)
    diff_metric_col1.metric("Added", len(kb_diff["added"]))
    diff_metric_col2.metric("Modified", len(kb_diff["modified"]))
    diff_metric_col3.metric("Removed", len(kb_diff["removed"]))
    diff_metric_col4.metric("Unchanged", kb_diff["unchanged_count"])

    with st.expander("Added Documents"):
        if kb_diff["added"]:
            for item in kb_diff["added"]:
                st.write(f"- `{item['id']}` {item['title']}")
        else:
            st.write("No added documents.")

    with st.expander("Modified Documents"):
        if kb_diff["modified"]:
            for item in kb_diff["modified"]:
                title = item["title_after"] if item["title_after"] else item["title_before"]
                flags = []
                if item["title_changed"]:
                    flags.append("title")
                if item["content_changed"]:
                    flags.append("content")
                st.write(f"- `{item['id']}` {title} ({', '.join(flags)} updated)")
        else:
            st.write("No modified documents.")

    with st.expander("Removed Documents"):
        if kb_diff["removed"]:
            for item in kb_diff["removed"]:
                st.write(f"- `{item['id']}` {item['title']}")
        else:
            st.write("No removed documents.")


def render_scale_performance_tab() -> None:
    st.subheader("Scale & Performance")
    st.caption(
        "Inspect the larger 500-question workload, benchmark indexing growth, and compare retrieval latency at 50K-document scale."
    )

    sync_completed_indexing_job()
    sync_completed_latency_job()

    dataset_stats = compute_large_scale_dataset_stats()
    vector_counts = get_collection_vector_counts()
    total_vectors = sum(vector_counts.values())
    vector_delta = summarize_collection_counts(vector_counts)

    metric_col1, metric_col2, metric_col3 = st.columns(3)
    metric_col1.metric(
        "Knowledge Base Size",
        f"{dataset_stats['document_count']:,} docs",
        delta=(
            f"{dataset_stats['total_chars']:,} chars"
            f" • {dataset_stats['file_size_mb']:.1f} MB"
        ),
    )
    metric_col2.metric(
        "Evaluation Set Size",
        f"{dataset_stats['eval_question_count']:,} questions",
        delta="Loaded from large_eval_questions.json",
    )
    metric_col3.metric(
        "Total Embeddings Indexed",
        f"{total_vectors:,} vectors",
        delta=vector_delta,
    )

    st.markdown("#### Evaluation Throughput")
    st.metric(
        "Historical Result Rows",
        f"{count_historical_evaluation_rows():,}",
        delta="Counted from every CSV saved under results/",
    )
    st.caption(
        "This total grows as comparison, regression, and scale benchmark runs are saved to disk."
    )

    st.markdown("#### Indexing Benchmark")
    st.write(
        "Benchmark index construction on the first 1K, 5K, 10K, and 50K documents from the large knowledge base."
    )
    latest_saved_indexing_run = get_latest_saved_indexing_run()
    indexing_action_col, indexing_load_col = st.columns(2)
    with indexing_action_col:
        run_indexing_clicked = st.button(
            "Run Indexing Benchmark",
            type="primary",
            key="scale_indexing_benchmark_button",
        )
    with indexing_load_col:
        load_saved_indexing_clicked = st.button(
            "Load Latest Saved Result",
            key="load_latest_indexing_result_button",
            disabled=latest_saved_indexing_run is None,
        )

    if latest_saved_indexing_run is not None:
        st.caption(f"Latest saved run: {latest_saved_indexing_run['label']}")

    if load_saved_indexing_clicked and latest_saved_indexing_run is not None:
        if load_saved_indexing_result_into_session(latest_saved_indexing_run["run_id"]):
            st.rerun()
        else:
            st.warning("The latest saved indexing benchmark could not be loaded from disk.")

    if run_indexing_clicked:
        if st.session_state.get("scale_indexing_active_job_id"):
            st.session_state["scale_indexing_background_notice"] = (
                "A background indexing benchmark is already running. Waiting for that run to finish."
            )
            run_demo_indexing_progress()
            if latest_saved_indexing_run is not None:
                load_saved_indexing_result_into_session(latest_saved_indexing_run["run_id"])
            st.rerun()

        try:
            started = start_background_indexing_job()
        except Exception as exc:
            render_runtime_error(exc, "start the indexing benchmark")
        else:
            if started:
                run_demo_indexing_progress()
                if latest_saved_indexing_run is not None:
                    load_saved_indexing_result_into_session(latest_saved_indexing_run["run_id"])
                st.rerun()

    indexing_background_error = st.session_state.get("scale_indexing_background_error")
    if indexing_background_error:
        st.error(f"Background indexing benchmark failed: {indexing_background_error}")

    indexing_background_notice = st.session_state.get("scale_indexing_background_notice")
    if indexing_background_notice:
        st.caption(indexing_background_notice)

    if st.session_state.get("scale_indexing_active_job_id"):
        render_indexing_background_status()

    indexing_result = st.session_state.get("scale_indexing_benchmark")
    if indexing_result:
        st.plotly_chart(
            build_indexing_benchmark_chart(indexing_result["results_df"]),
            use_container_width=True,
        )
        st.dataframe(
            indexing_result["results_df"].style.format(
                {
                    "Documents": "{:,.0f}",
                    "Vectors Indexed": "{:,.0f}",
                    "Time (s)": "{:.2f}",
                    "Seconds per 1K Docs": "{:.2f}",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )
        if indexing_result.get("source") == "saved":
            st.caption(
                "Showing saved indexing benchmark results from disk"
                f" ({format_saved_timestamp(indexing_result['saved_timestamp'])})."
            )
        elif indexing_result.get("source") == "fresh" and indexing_result.get("completed_at"):
            st.caption(
                "Showing the latest real indexing benchmark result"
                f" ({format_saved_timestamp(indexing_result['completed_at'])})."
            )
        st.write(indexing_result["summary_text"])
    else:
        st.info("Run the indexing benchmark to see the scaling curve.")

    st.markdown("#### Retrieval Latency Benchmark")
    st.write(
        "Measure retrieval-only latency on a cached 50K-document index using 100 deterministic random questions from the large evaluation set."
    )
    latest_saved_latency_run = get_latest_saved_latency_run()
    latency_action_col, latency_load_col = st.columns(2)
    with latency_action_col:
        run_latency_clicked = st.button("Run Latency Benchmark", key="scale_latency_benchmark_button")
    with latency_load_col:
        load_saved_latency_clicked = st.button(
            "Load Latest Saved Result",
            key="load_latest_latency_result_button",
            disabled=latest_saved_latency_run is None,
        )

    if latest_saved_latency_run is not None:
        st.caption(f"Latest saved run: {latest_saved_latency_run['label']}")

    if load_saved_latency_clicked and latest_saved_latency_run is not None:
        if load_saved_latency_result_into_session(latest_saved_latency_run["run_id"]):
            st.rerun()
        else:
            st.warning("The latest saved latency benchmark could not be loaded from disk.")

    if run_latency_clicked:
        if st.session_state.get("scale_latency_active_job_id"):
            st.session_state["scale_latency_background_notice"] = (
                "A background latency benchmark is already running. Waiting for that run to finish."
            )
            run_demo_latency_progress()
            if latest_saved_latency_run is not None:
                load_saved_latency_result_into_session(latest_saved_latency_run["run_id"])
            st.rerun()

        try:
            started = start_background_latency_job()
        except Exception as exc:
            render_runtime_error(exc, "start the latency benchmark")
        else:
            if started:
                run_demo_latency_progress()
                if latest_saved_latency_run is not None:
                    load_saved_latency_result_into_session(latest_saved_latency_run["run_id"])
                st.rerun()

    latency_background_error = st.session_state.get("scale_latency_background_error")
    if latency_background_error:
        st.error(f"Background latency benchmark failed: {latency_background_error}")

    latency_background_notice = st.session_state.get("scale_latency_background_notice")
    if latency_background_notice:
        st.caption(latency_background_notice)

    if st.session_state.get("scale_latency_active_job_id"):
        render_latency_background_status()

    latency_result = st.session_state.get("scale_latency_benchmark")
    if latency_result:
        st.plotly_chart(
            build_latency_benchmark_chart(latency_result["summary_df"]),
            use_container_width=True,
        )
        st.dataframe(
            latency_result["summary_df"].style.format(
                {
                    "P50 (ms)": "{:.2f}",
                    "P95 (ms)": "{:.2f}",
                    "P99 (ms)": "{:.2f}",
                    "Mean (ms)": "{:.2f}",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )
        if latency_result.get("source") == "saved":
            st.caption(
                "Showing saved latency benchmark results from disk"
                f" ({format_saved_timestamp(latency_result['saved_timestamp'])})."
            )
        elif latency_result.get("source") == "fresh" and latency_result.get("completed_at"):
            st.caption(
                "Showing the latest real latency benchmark result"
                f" ({format_saved_timestamp(latency_result['completed_at'])})."
            )
        st.caption(
            f"Latency statistics are based on {latency_result['sample_count']} retrievals per pipeline."
        )
    else:
        st.info("Run the latency benchmark to compare P50, P95, and P99 retrieval latency.")


def main() -> None:
    init_session_state()
    mode = render_sidebar()

    st.title("🔬 RAG Quality Lab")
    if mode == "Interactive Query":
        render_interactive_query_tab()
    elif mode == "Pipeline A/B Comparison":
        render_comparison_tab()
    elif mode == "Regression Test":
        render_regression_tab()
    else:
        render_scale_performance_tab()


if __name__ == "__main__":
    main()
