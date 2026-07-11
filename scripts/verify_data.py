"""Verify the data files in this repo against data/MANIFEST.json.

Checks, per manifest entry:
- in_git files: existence, byte size, SHA-256, and top-level JSON record count.
- out-of-git MS MARCO-derived files: skipped when absent (absence is by design
  for public clones). When a file IS present (workstation / a machine holding
  regenerated data), its record count is checked and its SHA-256 + exact byte
  size are computed and printed so they can be recorded in the manifest.

Usage:
    python scripts/verify_data.py            # verify against the manifest
    python scripts/verify_data.py --write    # regenerate manifest entries for
                                             # files present on disk (descriptions
                                             # and notes for known paths are kept)

Exit code 0 = all present files match the manifest; 1 = any mismatch.
"""

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = REPO_ROOT / "data" / "MANIFEST.json"


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def record_count(path: Path) -> int:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"{path} is not a top-level JSON array")
    return len(data)


def load_manifest() -> dict:
    with MANIFEST_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def verify() -> int:
    manifest = load_manifest()
    failures = []
    for entry in manifest["files"]:
        rel = entry["path"]
        path = REPO_ROOT / rel
        if not path.exists():
            if entry.get("in_git", False):
                failures.append(f"{rel}: MISSING (tracked file expected on disk)")
            else:
                print(f"SKIP  {rel}: absent by design (out-of-git corpus)")
            continue

        actual_bytes = path.stat().st_size
        actual_sha = sha256_of(path)
        try:
            actual_records = record_count(path)
        except ValueError as e:
            failures.append(str(e))
            continue

        if entry.get("records") is not None and actual_records != entry["records"]:
            failures.append(
                f"{rel}: record count {actual_records} != manifest {entry['records']}"
            )
        if entry.get("bytes") is not None and actual_bytes != entry["bytes"]:
            failures.append(f"{rel}: size {actual_bytes} != manifest {entry['bytes']}")
        if entry.get("sha256") is not None and actual_sha != entry["sha256"]:
            failures.append(f"{rel}: sha256 {actual_sha} != manifest {entry['sha256']}")

        if entry.get("sha256") is None or entry.get("bytes") is None:
            # Out-of-git corpus present on this machine: print values to record.
            print(
                f"INFO  {rel}: present here - RECORD THESE IN THE MANIFEST: "
                f"records={actual_records} bytes={actual_bytes} sha256={actual_sha}"
            )
        else:
            print(f"OK    {rel}: records={actual_records} bytes={actual_bytes} sha256 match")

    if failures:
        print("\nFAILURES:")
        for f in failures:
            print(f"  {f}")
        return 1
    print("\nAll present data files match data/MANIFEST.json.")
    return 0


def write() -> int:
    manifest = load_manifest()
    for entry in manifest["files"]:
        path = REPO_ROOT / entry["path"]
        if not path.exists():
            continue
        entry["records"] = record_count(path)
        entry["bytes"] = path.stat().st_size
        entry["sha256"] = sha256_of(path)
    manifest["generated_at_utc"] = (
        datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    )
    manifest["generator"] = "scripts/verify_data.py --write"
    with MANIFEST_PATH.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")
    print(f"Rewrote {MANIFEST_PATH}")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="regenerate manifest values")
    args = parser.parse_args()
    sys.exit(write() if args.write else verify())
