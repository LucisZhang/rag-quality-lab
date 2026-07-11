from __future__ import annotations

import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import chromadb


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
BENCHMARK_CHROMA_DIR = PROJECT_ROOT / "chroma_db_benchmarks"

COMPARISON_METRICS = [
    "faithfulness",
    "answer_relevancy",
    "context_precision",
    "context_recall",
    "answer_correctness",
]
PIPELINE_DISPLAY_NAMES = {
    "Pipeline A (Naive Vector)": "Pipeline A",
    "Pipeline B (Hybrid+Rerank)": "Pipeline B",
}
COMPARISON_ARTIFACT_PATTERN = re.compile(
    r"pipeline_(?P<pipeline>[ab])_(?P<max_questions>\d+)_questions_(?P<timestamp>\d{8}_\d{6})_(?P<kind>results|summary)\.(?P<extension>csv|json)$"
)
REGRESSION_REPORT_PATTERN = re.compile(
    r"regression_report_(?P<timestamp>\d{8}_\d{6})\.json$"
)


def to_project_relative(path: Path) -> str:
    resolved = path.expanduser().resolve(strict=False)
    try:
        return str(resolved.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def humanize_metric_name(metric: str) -> str:
    return metric.replace("_", " ").title()


def parse_string_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    if value is None:
        return []
    text = str(value).strip()
    if not text:
        return []
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return [text]
    if isinstance(parsed, list):
        return [str(item) for item in parsed]
    return [str(parsed)]


def parse_float(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def shorten_text(text: str, limit: int = 220) -> str:
    collapsed = " ".join(str(text).split())
    if len(collapsed) <= limit:
        return collapsed
    return collapsed[: limit - 3] + "..."


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def build_latest_comparison_run() -> dict[str, Any]:
    output_dir = RESULTS_DIR / "dashboard_comparison"
    artifacts_by_limit: dict[int, dict[str, dict[str, dict[str, str]]]] = {}
    for path in output_dir.iterdir():
        if not path.is_file():
            continue
        match = COMPARISON_ARTIFACT_PATTERN.fullmatch(path.name)
        if match is None:
            continue
        parsed = match.groupdict()
        max_questions = int(parsed["max_questions"])
        pipeline = str(parsed["pipeline"])
        timestamp = str(parsed["timestamp"])
        kind = str(parsed["kind"])
        artifacts_by_limit.setdefault(max_questions, {"a": {}, "b": {}})
        snapshot = artifacts_by_limit[max_questions][pipeline].setdefault(timestamp, {})
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
        for run_a, run_b in zip(pipeline_a_runs, pipeline_b_runs, strict=False):
            latest_timestamp = max(str(run_a["timestamp"]), str(run_b["timestamp"]))
            runs.append(
                {
                    "max_questions": max_questions,
                    "timestamp": latest_timestamp,
                    "results_a_path": Path(run_a["results_path"]),
                    "summary_a_path": Path(run_a["summary_path"]),
                    "results_b_path": Path(run_b["results_path"]),
                    "summary_b_path": Path(run_b["summary_path"]),
                }
            )
    runs.sort(key=lambda item: item["timestamp"], reverse=True)
    if not runs:
        raise FileNotFoundError("No saved comparison runs were found.")
    return runs[0]


def build_figure4_payload(run: dict[str, Any]) -> dict[str, Any]:
    rows_a = read_csv_rows(run["results_a_path"])
    rows_b_by_question = {
        str(row["question"]): row for row in read_csv_rows(run["results_b_path"])
    }

    headers = ["Question"]
    for metric in COMPARISON_METRICS:
        label = humanize_metric_name(metric)
        headers.append(f"Pipeline A | {label}")
        headers.append(f"Pipeline B | {label}")

    rows: list[list[Any]] = []
    for row_a in rows_a:
        question = str(row_a["question"])
        row_b = rows_b_by_question.get(question)
        if row_b is None:
            continue
        merged_row: list[Any] = [question]
        for metric in COMPARISON_METRICS:
            merged_row.append(parse_float(row_a.get(metric)))
            merged_row.append(parse_float(row_b.get(metric)))
        rows.append(merged_row)

    return {
        "title": "Figure 4. Dashboard A/B Tab - Detailed Per-Question Scores",
        "subtitle": (
            f"Source: {run['results_a_path'].name} + {run['results_b_path'].name} "
            f"(latest paired saved comparison run, {run['max_questions']} questions)"
        ),
        "headers": headers,
        "rows": rows,
    }


def build_latest_regression_report() -> tuple[Path, dict[str, Any], str]:
    output_dir = RESULTS_DIR / "dashboard_regression"
    candidates: list[tuple[str, Path]] = []
    for path in output_dir.iterdir():
        if not path.is_file():
            continue
        match = REGRESSION_REPORT_PATTERN.fullmatch(path.name)
        if match is None:
            continue
        candidates.append((str(match.group("timestamp")), path))
    if not candidates:
        raise FileNotFoundError("No saved regression reports were found.")
    candidates.sort(key=lambda item: item[0], reverse=True)
    timestamp, path = candidates[0]
    return path, json.loads(path.read_text(encoding="utf-8")), timestamp


def build_figure5_payload(report_path: Path, report: dict[str, Any]) -> dict[str, Any]:
    rows: list[list[Any]] = []
    for metric, values in report["overall_diff"].items():
        rows.append(
            [
                humanize_metric_name(metric),
                values.get("baseline"),
                values.get("updated"),
                values.get("diff"),
            ]
        )
    return {
        "title": "Figure 5. Dashboard Regression Tab - Pipeline B Overall Metric Changes",
        "subtitle": (
            f"Source: {report_path.name} | "
            f"{report['pipeline_label']} | {report['max_questions']} questions"
        ),
        "pipeline_label": report["pipeline_label"],
        "headers": ["Metric", "Baseline", "Updated", "Diff"],
        "rows": rows,
    }


def build_figure6_payload(report_path: Path, report: dict[str, Any]) -> dict[str, Any]:
    status_priority = {"degraded": 0, "improved": 1, "stable": 2}
    detail_rows = sorted(
        report["detail_rows"],
        key=lambda row: (status_priority.get(str(row["status"]), 3), str(row["question"])),
    )

    cards: list[dict[str, Any]] = []
    for item in detail_rows:
        metrics = []
        for metric in COMPARISON_METRICS:
            metrics.append(
                {
                    "metric": humanize_metric_name(metric),
                    "v1": item["baseline_scores"].get(metric),
                    "v2": item["updated_scores"].get(metric),
                    "diff": item["diff"].get(metric),
                }
            )
        cards.append(
            {
                "status": str(item["status"]),
                "question": str(item["question"]),
                "relevant_doc_ids": [str(value) for value in item.get("relevant_doc_ids", [])],
                "baseline_retrieved_doc_ids": [
                    str(value) for value in item.get("baseline_retrieved_doc_ids", [])
                ],
                "updated_retrieved_doc_ids": [
                    str(value) for value in item.get("updated_retrieved_doc_ids", [])
                ],
                "degraded_metrics": [
                    humanize_metric_name(str(value))
                    for value in item.get("degraded_metrics", [])
                ],
                "baseline_answer": str(item.get("baseline_answer", "")),
                "updated_answer": str(item.get("updated_answer", "")),
                "metrics": metrics,
            }
        )

    status_counts = defaultdict(int)
    for card in cards:
        status_counts[card["status"]] += 1

    return {
        "title": "Figure 6. Dashboard Regression Tab - Per-Question Status Cards",
        "subtitle": (
            f"Source: {report_path.name} | "
            f"{report['pipeline_label']} | {report['max_questions']} questions"
        ),
        "counts": {
            "improved": int(status_counts["improved"]),
            "degraded": int(status_counts["degraded"]),
            "stable": int(status_counts["stable"]),
        },
        "cards": cards,
    }


def build_figure7_payload(report_path: Path, report: dict[str, Any]) -> dict[str, Any]:
    kb_diff = report["knowledge_base_diff"]
    added_rows = [
        [item["id"], item["title"], shorten_text(item.get("content", ""))]
        for item in kb_diff["added"]
    ]
    modified_rows = [
        [
            item["id"],
            item["title_before"],
            item["title_after"],
            ", ".join(
                flag
                for flag, changed in (
                    ("title", item.get("title_changed")),
                    ("content", item.get("content_changed")),
                )
                if changed
            )
            or "none",
        ]
        for item in kb_diff["modified"]
    ]
    removed_rows = [
        [item["id"], item["title"], shorten_text(item.get("content", ""))]
        for item in kb_diff["removed"]
    ]
    return {
        "title": "Figure 7. Dashboard Regression Tab - Change Analysis",
        "subtitle": (
            f"Source: {report_path.name} | "
            f"{report['pipeline_label']} | {report['max_questions']} questions"
        ),
        "summary_text": kb_diff["summary_text"],
        "counts": {
            "added": len(kb_diff["added"]),
            "modified": len(kb_diff["modified"]),
            "removed": len(kb_diff["removed"]),
            "unchanged": int(kb_diff["unchanged_count"]),
        },
        "added": {
            "headers": ["ID", "Title", "Content Preview"],
            "rows": added_rows,
        },
        "modified": {
            "headers": ["ID", "Title Before", "Title After", "Updated Fields"],
            "rows": modified_rows,
        },
        "removed": {
            "headers": ["ID", "Title", "Content Preview"],
            "rows": removed_rows,
        },
    }


def compute_large_scale_dataset_stats() -> dict[str, Any]:
    kb_path = DATA_DIR / "large_knowledge_base.json"
    eval_path = DATA_DIR / "large_eval_questions.json"
    kb_records = json.loads(kb_path.read_text(encoding="utf-8"))
    eval_records = json.loads(eval_path.read_text(encoding="utf-8"))
    return {
        "document_count": len(kb_records),
        "total_chars": sum(len(str(item.get("content", ""))) for item in kb_records),
        "file_size_mb": kb_path.stat().st_size / (1024**2),
        "eval_question_count": len(eval_records),
    }


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

    parts: list[str] = []
    if category_totals["app"]:
        parts.append(f"app: {category_totals['app']:,}")
    if category_totals["indexing"]:
        parts.append(f"indexing: {category_totals['indexing']:,}")
    if category_totals["latency"]:
        parts.append(f"latency: {category_totals['latency']:,}")
    return " | ".join(parts)


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


def build_figure10_payload() -> dict[str, Any]:
    dataset_stats = compute_large_scale_dataset_stats()
    vector_counts = get_collection_vector_counts()
    total_vectors = sum(vector_counts.values())

    category_counts = {
        "app": sum(
            count for name, count in vector_counts.items() if name.startswith("app:")
        ),
        "indexing": sum(
            count for name, count in vector_counts.items() if name.startswith("indexing:")
        ),
        "latency": sum(
            count
            for name, count in vector_counts.items()
            if name.startswith("latency_a:") or name.startswith("latency_b:")
        ),
    }

    return {
        "title": "Figure 10. Dashboard Scale Tab - Three Key Metrics Cards",
        "subtitle": "Source: live scale dashboard backing data and persisted benchmark collections",
        "cards": [
            {
                "label": "Knowledge Base Size",
                "value": f"{dataset_stats['document_count']:,} docs",
                "detail": (
                    f"{dataset_stats['total_chars']:,} chars | "
                    f"{dataset_stats['file_size_mb']:.1f} MB"
                ),
            },
            {
                "label": "Evaluation Set Size",
                "value": f"{dataset_stats['eval_question_count']:,} questions",
                "detail": "Loaded from large_eval_questions.json",
            },
            {
                "label": "Total Embeddings Indexed",
                "value": f"{total_vectors:,} vectors",
                "detail": summarize_collection_counts(vector_counts),
            },
        ],
        "category_summary": [
            ["app", category_counts["app"]],
            ["indexing", category_counts["indexing"]],
            ["latency", category_counts["latency"]],
        ],
        "tracked_collections": [
            [name, count] for name, count in sorted(vector_counts.items())
        ],
        "historical_result_rows": count_historical_evaluation_rows(),
    }


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("Usage: export_dashboard_word_assets.py <output_json_path>")

    output_path = Path(sys.argv[1]).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    comparison_run = build_latest_comparison_run()
    regression_report_path, regression_report, regression_timestamp = build_latest_regression_report()

    payload = {
        "metadata": {
            "project_root": to_project_relative(PROJECT_ROOT),
            "comparison_timestamp": comparison_run["timestamp"],
            "regression_timestamp": regression_timestamp,
            "comparison_sources": [
                comparison_run["results_a_path"].name,
                comparison_run["results_b_path"].name,
            ],
            "regression_source": regression_report_path.name,
        },
        "figure4": build_figure4_payload(comparison_run),
        "figure5": build_figure5_payload(regression_report_path, regression_report),
        "figure6": build_figure6_payload(regression_report_path, regression_report),
        "figure7": build_figure7_payload(regression_report_path, regression_report),
        "figure10": build_figure10_payload(),
    }

    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
