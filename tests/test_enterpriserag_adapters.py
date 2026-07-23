# -*- coding: utf-8 -*-
"""Unit tests for the EnterpriseRAG-Bench S1 adapters (no models, no network).

Synthetic fixtures exercise the conversion logic; two integration checks run
against the real (gitignored) S1 mirror when it is present on this machine and
skip cleanly in CI, where the raw data does not ship.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.adapters.enterpriserag_s1_to_eval import (
    build_eval_records,
    load_kb_doc_ids,
)
from scripts.adapters.enterpriserag_s1_to_kb import (
    build_kb_records,
    extract_dsid,
    extract_title,
)

REAL_MIRROR = Path(__file__).resolve().parents[1] / "data" / "enterpriserag-bench" / "v1.0.0"

DSID_A = "dsid_" + "a" * 32
DSID_B = "dsid_" + "b" * 32
DSID_C = "dsid_" + "c" * 32


def write_doc(root: Path, source: str, dsid: str, slug: str, text: str) -> None:
    # Mirror the real layout: extracted/<source>/<source>/<file>.txt
    doc_dir = root / source / source
    doc_dir.mkdir(parents=True, exist_ok=True)
    (doc_dir / f"{dsid}__{slug}.txt").write_text(text, encoding="utf-8")


@pytest.fixture()
def extracted_tree(tmp_path: Path) -> Path:
    root = tmp_path / "extracted"
    write_doc(root, "confluence", DSID_A, "alpha-page", "Alpha Page Title\n\nBody A.")
    write_doc(root, "confluence", DSID_B, "beta-page", "  Beta   Page  \nBody B.")
    write_doc(root, "jira", DSID_C, "TCK-1-some-ticket", "Ticket title line\nDetails.")
    # duplicate dsid across sources, different content (real-data quirk)
    write_doc(root, "jira", DSID_A, "TCK-2-related", "Jira facet of alpha\nMore.")
    return root


def test_extract_dsid_accepts_real_pattern_and_rejects_others():
    assert extract_dsid(f"{DSID_A}__anything.txt") == DSID_A
    with pytest.raises(ValueError):
        extract_dsid("notadsid__x.txt")
    with pytest.raises(ValueError):
        extract_dsid("dsid_shorthex__x.txt")


def test_extract_title_first_nonempty_line_collapsed_and_capped():
    assert extract_title("\n\n  A   Title  \nrest", fallback="f") == "A Title"
    assert extract_title("", fallback="fallback-slug") == "fallback-slug"
    long_line = "x" * 500
    assert len(extract_title(long_line, fallback="f")) == 300


def test_build_kb_records_counts_order_and_duplicates(extracted_tree: Path):
    records, stats = build_kb_records(extracted_tree)

    assert stats["total_records"] == 4
    assert stats["per_source"] == {"confluence": 2, "jira": 2}
    assert stats["duplicate_ids"] == [DSID_A]
    assert stats["empty_files"] == []

    # deterministic order: confluence (sorted) then jira (sorted)
    assert [record["id"] for record in records] == [DSID_A, DSID_B, DSID_A, DSID_C]
    assert [record["source_type"] for record in records] == [
        "confluence",
        "confluence",
        "jira",
        "jira",
    ]

    alpha = records[0]
    assert alpha["title"] == "Alpha Page Title"
    assert alpha["content"].startswith("Alpha Page Title")
    # whitespace-collapsed title from a messy first line
    assert records[1]["title"] == "Beta Page"
    # the duplicated dsid keeps both records with distinct content
    assert records[2]["content"] != alpha["content"]


def test_build_kb_records_missing_source_dir_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        build_kb_records(tmp_path / "extracted")


def question_line(
    question_id: str,
    source_types: list[str],
    expected_doc_ids: list[str],
    question_type: str = "basic",
) -> str:
    return json.dumps(
        {
            "question_id": question_id,
            "question_type": question_type,
            "source_types": source_types,
            "question": f"Question {question_id}?",
            "expected_doc_ids": expected_doc_ids,
            "gold_answer": f"Answer {question_id}.",
            "answer_facts": [f"Fact {question_id}."],
        }
    )


@pytest.fixture()
def questions_file(tmp_path: Path) -> Path:
    lines = [
        question_line("qst_0002", ["confluence"], [DSID_B]),
        question_line("qst_0001", ["jira"], [DSID_C, DSID_C], "semantic"),
        question_line("qst_0003", ["github"], ["dsid_" + "d" * 32]),
        question_line("qst_0004", ["confluence", "jira"], [DSID_A, DSID_C]),
        question_line("qst_0005", ["confluence", "github"], [DSID_B]),
        question_line("qst_0006", [], []),
    ]
    path = tmp_path / "questions.jsonl"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def test_build_eval_records_filters_sorts_and_dedupes(questions_file: Path):
    records, stats = build_eval_records(questions_file)

    # S1 pool: only source_types that are non-empty subsets of {confluence, jira}
    assert [record["question_id"] for record in records] == [
        "qst_0001",
        "qst_0002",
        "qst_0004",
    ]
    assert stats["total_questions_in_file"] == 6
    assert stats["selected_records"] == 3
    assert stats["selected_by_type"] == {"basic": 2, "semantic": 1}

    by_id = {record["question_id"]: record for record in records}
    # duplicate expected id deduped (the qst_0413 real-data quirk)
    assert by_id["qst_0001"]["relevant_doc_ids"] == [DSID_C]
    assert by_id["qst_0004"]["relevant_doc_ids"] == [DSID_A, DSID_C]
    assert by_id["qst_0002"]["ground_truth"] == "Answer qst_0002."
    assert by_id["qst_0002"]["answer_facts"] == ["Fact qst_0002."]
    assert stats["multi_doc_questions"] == 1


def test_build_eval_records_validates_doc_ids(questions_file: Path):
    known = {DSID_A, DSID_B, DSID_C}
    records, _stats = build_eval_records(questions_file, known_doc_ids=known)
    assert len(records) == 3

    with pytest.raises(ValueError, match="qst_0004"):
        build_eval_records(questions_file, known_doc_ids={DSID_B, DSID_C})


def test_build_eval_records_missing_keys_raise(tmp_path: Path):
    path = tmp_path / "questions.jsonl"
    path.write_text(json.dumps({"question_id": "qst_0001"}) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing keys"):
        build_eval_records(path)


def test_load_kb_doc_ids(tmp_path: Path):
    kb_path = tmp_path / "kb.json"
    kb_path.write_text(
        json.dumps(
            [
                {"id": DSID_A, "title": "t", "content": "c"},
                {"id": DSID_B, "title": "t", "content": "c"},
            ]
        ),
        encoding="utf-8",
    )
    assert load_kb_doc_ids(kb_path) == {DSID_A, DSID_B}


@pytest.mark.skipif(
    not (REAL_MIRROR / "extracted").is_dir(),
    reason="raw S1 mirror not present (gitignored; download per DATA.md)",
)
def test_real_s1_corpus_counts_match_acquisition_evidence():
    records, stats = build_kb_records(REAL_MIRROR / "extracted")
    assert stats["total_records"] == 11309
    assert stats["per_source"] == {"confluence": 5189, "jira": 6120}
    assert stats["duplicate_ids"] == [
        "dsid_6df52fdb96ae4edcb76464738bca3340",
        "dsid_feb1e9063ebb4947bb4f935393c01f0f",
    ]
    assert len(records) == 11309


@pytest.mark.skipif(
    not (REAL_MIRROR / "questions.jsonl").is_file(),
    reason="raw S1 mirror not present (gitignored; download per DATA.md)",
)
def test_real_s1_eval_pool_matches_acquisition_evidence():
    records, stats = build_eval_records(REAL_MIRROR / "questions.jsonl")
    assert stats["total_questions_in_file"] == 500
    assert stats["selected_records"] == 130
    assert len(records) == 130
    assert all(record["relevant_doc_ids"] for record in records)
