# -*- coding: utf-8 -*-
"""Shared pytest setup: repo-root imports + stubs for missing heavy deps."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tests.dependency_stubs import install_missing_dependency_stubs  # noqa: E402

install_missing_dependency_stubs()
