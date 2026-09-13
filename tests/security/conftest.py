from __future__ import annotations

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def cp_client(monkeypatch, tmp_path):
    """Control-plane TestClient over a throwaway sqlite DB but the REAL signed
    manifest catalog (models/manifests), per the task's security-test setup."""
    monkeypatch.setenv("MESH_CP_DB", str(tmp_path / "control.db"))
    monkeypatch.setenv("MESH_CP_MANIFEST_DIR", str(REPO_ROOT / "models" / "manifests"))

    from fastapi.testclient import TestClient
    from meshcompute_control.app import create_app

    with TestClient(create_app()) as client:
        yield client
