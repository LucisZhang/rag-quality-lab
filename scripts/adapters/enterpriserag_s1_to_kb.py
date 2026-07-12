#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Convert EnterpriseRAG-Bench S1 slices into the lab's knowledge-base JSON.

Input: the extracted v1.0.0 confluence + jira slices (see DATA.md and
evidence/c2-s1-mac-20260712/ for provenance and integrity records). Every
document file is named ``dsid_<32 hex>__<slug>.txt``; the ``dsid_*`` prefix is
the document id that ``questions.jsonl`` references via ``expected_doc_ids``.

Output: a knowledge-base JSON in the lab's schema — a list of
``{"id", "title", "content", "source_type"}`` records (``load_knowledge_base``
consumes ``id``/``title``/``content`` and ignores extras). The document id IS
the dsid, so retrieval scoring happens at dsid granularity. A dsid may map to
more than one file in the raw data (two known cases, recorded in
evidence/c2-s1-mac-20260712/README.md); those files stay separate records that
share an id — they are facets/versions of one logical document.

The conversion is deterministic: sources are processed in a fixed order and
files in sorted path order, so identical input slices (hash-verified across
machines) produce byte-identical output. The output file stays OUT of git
(~90 MB); integrity travels via data/MANIFEST.json.

Usage:
    python scripts/adapters/enterpriserag_s1_to_kb.py
    python scripts/adapters/enterpriserag_s1_to_kb.py \
        --extracted-dir data/enterpriserag-bench/v1.0.0/extracted \
        --output data/knowledge_base_enterpriserag_s1.json
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_EXTRACTED_DIR = REPO_ROOT / "data" / "enterpriserag-bench" / "v1.0.0" / "extracted"
DEFAULT_OUTPUT = REPO_ROOT / "data" / "knowledge_base_enterpriserag_s1.json"
DEFAULT_SOURCE_TYPES = ("confluence", "jira")
DSID_PATTERN = re.compile(r"^(dsid_[0-9a-f]{32})__")
TITLE_MAX_CHARS = 300


def extract_dsid(filename: str) -> str:
    """Return the dsid document id encoded in an S1 filename."""
    match = DSID_PATTERN.match(filename)
    if match is None:
        raise ValueError(
            f"Filename does not match the expected 'dsid_<32 hex>__<slug>' "
            f"pattern: {filename!r}"
        )
    return match.group(1)


def extract_title(text: str, fallback: str) -> str:
    """Return the first non-empty line as the document title."""
    for line in text.splitlines():
        candidate = " ".join(line.split())
        if candidate:
            return candidate[:TITLE_MAX_CHARS]
    return fallback[:TITLE_MAX_CHARS]


def build_kb_records(
    extracted_dir: Path,
    source_types: tuple[str, ...] = DEFAULT_SOURCE_TYPES,
) -> tuple[list[dict], dict]:
    """Build KB records from the extracted slices; return (records, stats)."""
    records: list[dict] = []
    per_source: dict[str, int] = {}
    empty_files: list[str] = []

    for source in source_types:
        source_dir = extracted_dir / source
        if not source_dir.is_dir():
            raise FileNotFoundError(
                f"Expected extracted source directory is missing: {source_dir}"
            )
        files = sorted(
            (path for path in source_dir.rglob("*.txt") if path.is_file()),
            key=lambda path: path.relative_to(source_dir).as_posix(),
        )
        per_source[source] = len(files)
        for path in files:
            dsid = extract_dsid(path.name)
            text = path.read_text(encoding="utf-8", errors="replace")
            stem_slug = path.stem.split("__", 1)[1] if "__" in path.stem else path.stem
            if not text.strip():
                empty_files.append(path.name)
            records.append(
                {
                    "id": dsid,
                    "title": extract_title(text, fallback=stem_slug),
                    "content": text,
                    "source_type": source,
                }
            )

    id_counts = collections.Counter(record["id"] for record in records)
    duplicate_ids = sorted(dsid for dsid, count in id_counts.items() if count > 1)
    stats = {
        "total_records": len(records),
        "per_source": per_source,
        "unique_ids": len(id_counts),
        "duplicate_ids": duplicate_ids,
        "empty_files": empty_files,
    }
    return records, stats


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extracted-dir", type=Path, default=DEFAULT_EXTRACTED_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    records, stats = build_kb_records(args.extracted_dir)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as file:
        json.dump(records, file, ensure_ascii=False)
        file.write("\n")

    stats["output"] = str(args.output)
    stats["output_bytes"] = args.output.stat().st_size
    stats["output_sha256"] = sha256_of(args.output)
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
