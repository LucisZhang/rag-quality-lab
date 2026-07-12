#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Adapt EnterpriseRAG-Bench questions.jsonl into the lab's eval-set JSON.

Selects the S1-answerable pool — questions whose ``source_types`` is non-empty
and a subset of {confluence, jira} — and maps each record into the schema
``RAGEvaluator.load_eval_dataset`` requires:

- ``question``          <- ``question``
- ``ground_truth``      <- ``gold_answer``
- ``relevant_doc_ids``  <- ``expected_doc_ids``, deduplicated preserving order
                           (dsid granularity; one known question lists a dsid
                           twice — see evidence/c2-s1-mac-20260712/README.md)

``question_id``, ``question_type``, ``source_types``, and ``answer_facts`` are
carried through as extra keys: the loader ignores them, while the retrieval
runner and later judged lanes use them for per-category reporting.

All counts derive from questions.jsonl itself, never from the dataset's cards
(the HF card and GitHub quickstart disagree on per-category counts). Output is
deterministic: records are sorted by ``question_id``. When ``--kb`` is given,
every relevant doc id is validated against the adapted knowledge base and any
miss is a hard error.

Usage:
    python scripts/adapters/enterpriserag_s1_to_eval.py \
        --kb data/knowledge_base_enterpriserag_s1.json
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_QUESTIONS = REPO_ROOT / "data" / "enterpriserag-bench" / "v1.0.0" / "questions.jsonl"
DEFAULT_OUTPUT = REPO_ROOT / "data" / "eval_questions_enterpriserag_s1.json"
S1_SOURCE_TYPES = frozenset({"confluence", "jira"})
REQUIRED_KEYS = {"question_id", "question_type", "source_types", "question",
                 "expected_doc_ids", "gold_answer"}


def _dedupe_preserve_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        unique.append(value)
    return unique


def build_eval_records(
    questions_path: Path,
    allowed_sources: frozenset[str] = S1_SOURCE_TYPES,
    known_doc_ids: set[str] | None = None,
) -> tuple[list[dict], dict]:
    """Build the S1 eval set from questions.jsonl; return (records, stats)."""
    records: list[dict] = []
    total = 0
    type_counts: collections.Counter[str] = collections.Counter()

    with questions_path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue
            total += 1
            raw = json.loads(line)
            missing = REQUIRED_KEYS.difference(raw)
            if missing:
                raise ValueError(
                    f"questions.jsonl line {line_number} is missing keys: "
                    f"{sorted(missing)}"
                )
            source_types = raw["source_types"]
            if not source_types or not set(source_types) <= allowed_sources:
                continue

            relevant_doc_ids = _dedupe_preserve_order(
                [str(doc_id) for doc_id in raw["expected_doc_ids"]]
            )
            if known_doc_ids is not None:
                unknown = [d for d in relevant_doc_ids if d not in known_doc_ids]
                if unknown:
                    raise ValueError(
                        f"Question {raw['question_id']} expects doc ids missing "
                        f"from the knowledge base: {unknown}"
                    )

            type_counts[str(raw["question_type"])] += 1
            records.append(
                {
                    "question_id": str(raw["question_id"]),
                    "question_type": str(raw["question_type"]),
                    "source_types": [str(s) for s in source_types],
                    "question": str(raw["question"]),
                    "ground_truth": str(raw["gold_answer"]),
                    "relevant_doc_ids": relevant_doc_ids,
                    "answer_facts": [str(f) for f in raw.get("answer_facts", [])],
                }
            )

    records.sort(key=lambda record: record["question_id"])
    stats = {
        "total_questions_in_file": total,
        "selected_records": len(records),
        "selected_by_type": dict(sorted(type_counts.items())),
        "multi_doc_questions": sum(
            1 for record in records if len(record["relevant_doc_ids"]) > 1
        ),
    }
    return records, stats


def load_kb_doc_ids(kb_path: Path) -> set[str]:
    """Collect the set of document ids from an adapted knowledge-base JSON."""
    with kb_path.open("r", encoding="utf-8") as file:
        kb_records = json.load(file)
    if not isinstance(kb_records, list):
        raise ValueError(f"{kb_path} is not a top-level JSON array")
    return {str(record["id"]) for record in kb_records}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--kb",
        type=Path,
        default=None,
        help="adapted KB JSON to validate relevant_doc_ids against (recommended)",
    )
    args = parser.parse_args()

    known_doc_ids = load_kb_doc_ids(args.kb) if args.kb is not None else None
    records, stats = build_eval_records(args.questions, known_doc_ids=known_doc_ids)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as file:
        json.dump(records, file, ensure_ascii=False, indent=2)
        file.write("\n")

    stats["output"] = str(args.output)
    stats["doc_id_validation"] = (
        "validated against --kb" if known_doc_ids is not None else "SKIPPED (no --kb)"
    )
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
