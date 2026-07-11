#!/usr/bin/env python3
"""Run the deterministic half of scripts/verify_a3.py without heavy ML deps.

scripts/verify_a3.py imports the pipeline modules at the top, which pull in
chromadb / sentence-transformers / langchain-ollama. Deterministic mode never
instantiates a pipeline or calls a model, so CI installs lightweight stubs for
whichever of those packages are missing (none are stubbed when the full
environment is present) and then executes the script unchanged. The exit code
propagates from verify_a3.py.

Note: like verify_a3.py itself, this rewrites
evidence/verified-2026-07/deterministic-checks.json in the working tree. In CI
the workspace is ephemeral; locally, restore the file afterwards if you do not
intend to refresh the committed evidence.
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from tests.dependency_stubs import install_missing_dependency_stubs  # noqa: E402

stubbed = install_missing_dependency_stubs()
if stubbed:
    print(f"Installed dependency stubs for: {', '.join(stubbed)}")

sys.argv = ["verify_a3.py", "--mode", "deterministic"]
runpy.run_path(str(REPO_ROOT / "scripts" / "verify_a3.py"), run_name="__main__")
