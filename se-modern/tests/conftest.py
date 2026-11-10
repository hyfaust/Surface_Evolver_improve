"""Pytest configuration: keep `src/` on the path and share SE availability markers."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from semodern import se_bridge  # noqa: E402

requires_se = pytest.mark.skipif(not se_bridge.available(), reason="Surface Evolver executable not available")


@pytest.fixture(scope="session")
def project_root() -> Path:
    """Absolute path to the `se-modern` project directory."""
    return PROJECT_ROOT


@pytest.fixture(scope="session")
def data_se(project_root: Path) -> Path:
    """Directory holding the copied SE benchmark datafiles."""
    return project_root / "data" / "se"


@pytest.fixture
def workdir(tmp_path: Path) -> Path:
    """Per-test scratch directory for SE runs and generated files."""
    return tmp_path
