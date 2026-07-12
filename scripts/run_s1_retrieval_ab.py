#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Judge-free retrieval A/B evaluation on the EnterpriseRAG-Bench S1 subset.

Runs the lab's two pipelines retrieval-only against the adapted S1 eval set
and scores deterministic document-level metrics (retrieval precision / recall
/ hit) at dsid granularity using per-question ``relevant_doc_ids`` ground
truth. NO LLM is ever invoked: only ``pipeline.retrieve()`` is called, never
``query()``/``generate()``. On a machine without Ollama, set
``RAG_EMBEDDING_BACKEND=hf`` (plus ``RAG_HF_EMBEDDING_MODEL``) and leave the
LLM backend at its default — the LLM client object is constructed by the
pipeline but never called.

Outputs, per pipeline, under ``--output-dir/<UTC timestamp>/``: a per-question
CSV and a summary JSON (overall means + per-question-type breakdown + the
active model runtime), plus a combined ``comparison.json``.

Usage:
    python scripts/run_s1_retrieval_ab.py                 # both pipelines
    python scripts/run_s1_retrieval_ab.py --pipelines naive --max-questions 5
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation_engine import (  # noqa: E402
    compute_retrieval_metrics,
    normalize_doc_ids,
)
from src.utils import get_active_model_config  # noqa: E402

DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_KB = DATA_DIR / "knowledge_base_enterpriserag_s1.json"
DEFAULT_EVAL = DATA_DIR / "eval_questions_enterpriserag_s1.json"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "s1_retrieval_ab"
DEFAULT_PERSIST_ROOT = PROJECT_ROOT / "chroma_db_s1"
METRIC_NAMES = ("retrieval_precision", "retrieval_recall", "retrieval_hit")
PIPELINE_KINDS = ("naive", "hybrid")


def load_eval_items(path: Path) -> list[dict]:
    """Load the adapted S1 eval set, keeping per-question metadata."""
    with path.open("r", encoding="utf-8") as file:
        records = json.load(file)
    if not isinstance(records, list):
        raise ValueError(f"{path} is not a top-level JSON array")

    items: list[dict] = []
    for index, record in enumerate(records, start=1):
        missing = {"question", "relevant_doc_ids"}.difference(record)
        if missing:
            raise ValueError(f"Eval item #{index} is missing keys: {sorted(missing)}")
        items.append(
            {
                "question_id": str(record.get("question_id", f"item_{index:04d}")),
                "question_type": str(record.get("question_type", "unknown")),
                "question": str(record["question"]),
                "relevant_doc_ids": normalize_doc_ids(record["relevant_doc_ids"]),
            }
        )
    return items


def retrieved_ids_from_docs(docs: list) -> list[str]:
    """Extract normalized doc ids from retrieved documents (dsid granularity)."""
    raw_ids = [
        str(doc.metadata.get("id", getattr(doc, "id", "") or "")).strip()
        for doc in docs
    ]
    return normalize_doc_ids(raw_ids)


def evaluate_pipeline_retrieval(
    pipeline,
    eval_items: list[dict],
    k: int,
) -> tuple[pd.DataFrame, dict]:
    """Score one pipeline retrieval-only; return (per-question df, summary)."""
    rows: list[dict] = []
    total = len(eval_items)
    for index, item in enumerate(eval_items, start=1):
        print(f"[{index}/{total}] retrieve: {item['question'][:60]}")
        retrieved_docs = pipeline.retrieve(item["question"], k=k)
        retrieved_ids = retrieved_ids_from_docs(retrieved_docs)
        metrics = compute_retrieval_metrics(retrieved_ids, item["relevant_doc_ids"])
        rows.append(
            {
                "question_id": item["question_id"],
                "question_type": item["question_type"],
                "question": item["question"],
                "relevant_doc_ids": json.dumps(
                    item["relevant_doc_ids"], ensure_ascii=False
                ),
                "retrieved_doc_ids": json.dumps(retrieved_ids, ensure_ascii=False),
                **metrics,
            }
        )

    results_df = pd.DataFrame(rows)
    summary = {
        "pipeline": getattr(pipeline, "pipeline_name", type(pipeline).__name__),
        "k": k,
        "questions": total,
        "metrics_mean": {
            name: (float(results_df[name].mean()) if total else None)
            for name in METRIC_NAMES
        },
        "by_question_type": {
            question_type: {
                "questions": int(len(group)),
                **{name: float(group[name].mean()) for name in METRIC_NAMES},
            }
            for question_type, group in sorted(results_df.groupby("question_type"))
        }
        if total
        else {},
    }
    return results_df, summary


def build_pipeline(kind: str, kb_path: Path, persist_root: Path):
    """Construct a real pipeline (imports heavy deps; not used by unit tests)."""
    from src.rag_pipelines import HybridRerankRAG, NaiveVectorRAG

    if kind == "naive":
        return NaiveVectorRAG(
            str(kb_path),
            collection_name="s1_naive",
            persist_dir=str(persist_root / "naive"),
        )
    if kind == "hybrid":
        return HybridRerankRAG(
            str(kb_path),
            collection_name="s1_hybrid",
            persist_dir=str(persist_root / "hybrid"),
        )
    raise ValueError(f"Unknown pipeline kind {kind!r}; expected {PIPELINE_KINDS}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kb", type=Path, default=DEFAULT_KB)
    parser.add_argument("--eval", dest="eval_path", type=Path, default=DEFAULT_EVAL)
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument(
        "--pipelines",
        default="naive,hybrid",
        help="comma-separated subset of: naive,hybrid",
    )
    parser.add_argument("--max-questions", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--persist-root", type=Path, default=DEFAULT_PERSIST_ROOT)
    args = parser.parse_args()

    kinds = [kind.strip() for kind in args.pipelines.split(",") if kind.strip()]
    unknown_kinds = [kind for kind in kinds if kind not in PIPELINE_KINDS]
    if unknown_kinds:
        raise SystemExit(f"Unknown pipelines: {unknown_kinds}; expected {PIPELINE_KINDS}")

    eval_items = load_eval_items(args.eval_path)
    if args.max_questions is not None:
        eval_items = eval_items[: args.max_questions]

    run_dir = args.output_dir / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir.mkdir(parents=True, exist_ok=True)

    comparison: dict = {
        "run_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "kb": str(args.kb),
        "eval": str(args.eval_path),
        "k": args.k,
        "questions": len(eval_items),
        "model_runtime": get_active_model_config(),
        "pipelines": {},
    }
    for kind in kinds:
        print(f"=== building pipeline: {kind} ===")
        pipeline = build_pipeline(kind, args.kb, args.persist_root)
        results_df, summary = evaluate_pipeline_retrieval(pipeline, eval_items, args.k)
        results_df.to_csv(run_dir / f"{kind}_results.csv", index=False, encoding="utf-8")
        (run_dir / f"{kind}_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        comparison["pipelines"][kind] = summary
        print(json.dumps(summary["metrics_mean"], indent=2))

    (run_dir / "comparison.json").write_text(
        json.dumps(comparison, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Saved run outputs to: {run_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
