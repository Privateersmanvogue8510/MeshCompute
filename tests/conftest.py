"""Shared fixtures across the whole suite.

No network, no private IPs (INSTRUCTIONS.md/SECURITY.md): every gateway test points
at throwaway http://*.test hostnames and mocks httpx with respx. Real deploy files
like deploy/nodes.local.yaml (which contains the actual 10.0.0.20 pod) are never
read by tests — gateway_env below overrides NODES_LOCAL_FILE before any test touches
meshcompute_gateway.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def gateway_env(monkeypatch, tmp_path):
    """Point the gateway at a throwaway nodes-local file + fake control-plane URL,
    and clear its lru_cache'd settings/endpoint readers so each test starts clean.

    Yields {"node_id", "backend_url", "control_url"} for the test to mock against.
    """
    node_id = "nd_test_backend"
    backend_url = "http://backend.test"
    control_url = "http://control-plane.test"

    nodes_file = tmp_path / "nodes.local.yaml"
    nodes_file.write_text(yaml.safe_dump({
        "nodes": [{"node_id": node_id, "backend_url": backend_url, "backend": "lmstudio"}],
    }))

    monkeypatch.setenv("MESH_GW_CONTROL_URL", control_url)
    monkeypatch.setenv("NODES_LOCAL_FILE", str(nodes_file))
    monkeypatch.delenv("MESH_GW_DIRECT_BACKEND_URL", raising=False)

    from meshcompute_gateway import router as gw_router
    from meshcompute_gateway import settings as gw_settings

    gw_settings.get_settings.cache_clear()
    gw_settings.load_node_endpoints.cache_clear()
    gw_router._backend_cache.clear()
    try:
        yield {"node_id": node_id, "backend_url": backend_url, "control_url": control_url}
    finally:
        gw_settings.get_settings.cache_clear()
        gw_settings.load_node_endpoints.cache_clear()
        gw_router._backend_cache.clear()


@pytest.fixture
def gateway_client(gateway_env):
    from fastapi.testclient import TestClient
    from meshcompute_gateway.app import create_app

    with TestClient(create_app()) as client:
        yield client


def make_manifest(model_id="test/model", backends=("lmstudio",), size_bytes=1_000_000_000,
                   context=4096):
    """Minimal-but-valid ModelManifest for scheduler/protocol tests — not the real
    catalog entry, just enough of the frozen shape (manifest.py) to exercise it."""
    from meshcompute_protocol import (
        ChatTemplateRef,
        LicenseRef,
        ModelManifest,
        Quantization,
        Runtime,
        ShardFile,
        TokenizerRef,
        UpstreamRef,
    )

    return ModelManifest(
        id=model_id,
        architecture="test-arch",
        context_length_max=context,
        context_length_configured=context,
        upstream=UpstreamRef(provider="test", repository="test/repo", revision="deadbeef"),
        license=LicenseRef(id="Apache-2.0"),
        runtime=Runtime(backends=list(backends)),
        tokenizer=TokenizerRef(revision="deadbeef"),
        chat_template=ChatTemplateRef(revision="deadbeef"),
        quantizations=[Quantization(
            id="Q1", files=[ShardFile(rfilename="f.gguf", size_bytes=size_bytes,
                                      artifact_root_hash="a" * 16)])],
    )


@pytest.fixture
def manifest_factory():
    return make_manifest
