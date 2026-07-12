# -*- coding: utf-8 -*-
"""Unit tests for the judge-free S1 retrieval runner (no models).

Pipelines are scripted fakes exposing only ``retrieve()`` and a name — they
deliberately have NO ``llm``/``generate``/``query`` attributes, so any code
path that reached for generation would fail loudly.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from langchain_core.documents import Document

from scripts.run_s1_retrieval_ab import (
    evaluate_pipeline_retrieval,
    file_provenance,
    load_eval_items,
    pipeline_definitions,
    retrieved_ids_from_docs,
    sha256_of,
    write_output_manifest,
)

DSID_A = "dsid_" + "a" * 32
DSID_B = "dsid_" + "b" * 32
DSID_C = "dsid_" + "c" * 32


def make_doc(doc_id: str, content: str = "body") -> Document:
    return Document(
        id=doc_id,
        page_content=content,
        metadata={"id": doc_id, "title": f"Title {doc_id}"},
    )


class ScriptedRetrievalPipeline:
    """Returns fixed docs per question; retrieval-only by construction."""

    pipeline_name = "ScriptedRetrieval"

    def __init__(self, docs_by_question: dict[str, list[Document]]) -> None:
        self._docs = docs_by_question
        self.seen_k: list[int] = []

    def retrieve(self, query: str, k: int = 4) -> list[Document]:
        self.seen_k.append(k)
        return self._docs[query][:k]


def eval_item(question_id: str, question: str, relevant: list[str], qtype: str = "basic"):
    return {
        "question_id": question_id,
        "question_type": qtype,
        "question": question,
        "relevant_doc_ids": relevant,
    }


def test_retrieved_ids_dedupe_at_dsid_granularity():
    # two chunks of the same document (shared dsid) collapse to one id
    docs = [make_doc(DSID_A, "chunk 1"), make_doc(DSID_A, "chunk 2"), make_doc(DSID_B)]
    assert retrieved_ids_from_docs(docs) == [DSID_A, DSID_B]


def test_evaluate_pipeline_retrieval_metrics_and_grouping():
    pipeline = ScriptedRetrievalPipeline(
        {
            "q one": [make_doc(DSID_A), make_doc(DSID_B)],  # hit for A
            "q two": [make_doc(DSID_C)],  # miss (relevant is B)
            "q three": [make_doc(DSID_A), make_doc(DSID_C)],  # partial multi-doc
        }
    )
    items = [
        eval_item("qst_0001", "q one", [DSID_A], "basic"),
        eval_item("qst_0002", "q two", [DSID_B], "basic"),
        eval_item("qst_0003", "q three", [DSID_A, DSID_B], "semantic"),
    ]

    results_df, summary = evaluate_pipeline_retrieval(pipeline, items, k=2)

    assert pipeline.seen_k == [2, 2, 2]
    assert summary["pipeline"] == "ScriptedRetrieval"
    assert summary["k"] == 2
    assert summary["questions"] == 3

    rows = {row["question_id"]: row for row in results_df.to_dict("records")}
    assert rows["qst_0001"]["retrieval_hit"] == 1.0
    assert rows["qst_0001"]["retrieval_precision"] == 0.5
    assert rows["qst_0001"]["retrieval_recall"] == 1.0
    assert rows["qst_0002"]["retrieval_hit"] == 0.0
    assert rows["qst_0002"]["retrieval_recall"] == 0.0
    # multi-doc: retrieved {A, C}, relevant {A, B} -> p=0.5, r=0.5, hit=1
    assert rows["qst_0003"]["retrieval_precision"] == 0.5
    assert rows["qst_0003"]["retrieval_recall"] == 0.5
    assert rows["qst_0003"]["retrieval_hit"] == 1.0

    assert summary["metrics_mean"]["retrieval_hit"] == pytest.approx(2 / 3)
    by_type = summary["by_question_type"]
    assert by_type["basic"]["questions"] == 2
    assert by_type["basic"]["retrieval_hit"] == 0.5
    assert by_type["semantic"]["questions"] == 1
    assert by_type["semantic"]["retrieval_recall"] == 0.5

    # stored id lists are JSON round-trippable
    assert json.loads(rows["qst_0003"]["retrieved_doc_ids"]) == [DSID_A, DSID_C]


def test_evaluate_pipeline_retrieval_empty_dataset():
    pipeline = ScriptedRetrievalPipeline({})
    results_df, summary = evaluate_pipeline_retrieval(pipeline, [], k=4)
    assert results_df.empty
    assert summary["questions"] == 0
    assert summary["metrics_mean"]["retrieval_hit"] is None
    assert summary["by_question_type"] == {}


def test_load_eval_items_roundtrip_and_validation(tmp_path: Path):
    path = tmp_path / "eval.json"
    path.write_text(
        json.dumps(
            [
                {
                    "question_id": "qst_0001",
                    "question_type": "basic",
                    "question": "q?",
                    "ground_truth": "a.",
                    "relevant_doc_ids": [DSID_A, DSID_A, "", DSID_B],
                }
            ]
        ),
        encoding="utf-8",
    )
    items = load_eval_items(path)
    assert items[0]["question_id"] == "qst_0001"
    # normalization dedupes and drops empties
    assert items[0]["relevant_doc_ids"] == [DSID_A, DSID_B]

    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps([{"question": "only-question"}]), encoding="utf-8")
    with pytest.raises(ValueError, match="missing keys"):
        load_eval_items(bad)


def test_file_provenance_and_output_manifest_hash_completed_artifacts(tmp_path: Path):
    comparison = tmp_path / "comparison.json"
    results = tmp_path / "naive_results.csv"
    comparison.write_text('{"questions": 130}\n', encoding="utf-8")
    results.write_text("question_id,retrieval_hit\nqst_1,1.0\n", encoding="utf-8")

    provenance = file_provenance(comparison, records=130)
    assert provenance["records"] == 130
    assert provenance["bytes"] == comparison.stat().st_size
    assert provenance["sha256"] == sha256_of(comparison)

    manifest_path = write_output_manifest(tmp_path, [results, comparison])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["hash_algorithm"] == "sha256"
    assert [item["path"] for item in manifest["files"]] == [
        str(comparison),
        str(results),
    ]
    assert {item["sha256"] for item in manifest["files"]} == {
        sha256_of(comparison),
        sha256_of(results),
    }
    assert all(item["path"] != str(manifest_path) for item in manifest["files"])


def test_pipeline_definitions_capture_real_ab_contract():
    definitions = pipeline_definitions(k=4)
    assert definitions["naive"]["final_k"] == 4
    assert definitions["naive"]["retrieval"] == "dense top-k"
    assert definitions["hybrid"]["retrieval"].startswith("dense top-10 + BM25")
    assert definitions["hybrid"]["reranker_model"] == (
        "cross-encoder/ms-marco-MiniLM-L-6-v2"
    )
