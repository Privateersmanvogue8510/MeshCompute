"""Gateway settings (pydantic-settings) + local deploy config.

Private worker IPs never live in committed code (SECURITY.md, INSTRUCTIONS §2.4).
Instead the gateway loads `deploy/nodes.local.yaml` (gitignored) to map a scheduler's
node_id to that worker's OpenAI-compatible backend_url. `deploy/nodes.example.yaml`
is the committed, IP-free template.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import yaml
from pydantic_settings import BaseSettings, SettingsConfigDict

# apps/gateway/meshcompute_gateway/settings.py -> repo root is 3 parents up.
_REPO_ROOT = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(extra="ignore")

    mesh_gw_bind: str = "127.0.0.1:8081"
    mesh_gw_control_url: str = "http://127.0.0.1:8080"
    # Dev/test fallback (router.py): used ONLY when the control-plane can't be reached,
    # so the gateway is verifiable before the control-plane's /schedule exists.
    mesh_gw_direct_backend_url: str | None = None
    nodes_local_file: str = "deploy/nodes.local.yaml"


@lru_cache
def get_settings() -> Settings:
    return Settings()


def _resolve(path_str: str) -> Path:
    p = Path(path_str)
    if p.is_absolute():
        return p
    cwd_candidate = Path.cwd() / p
    return cwd_candidate if cwd_candidate.exists() else _REPO_ROOT / p


@dataclass(frozen=True)
class NodeEndpoint:
    node_id: str
    backend_url: str
    backend: str = "lmstudio"


# Cached by (path, mtime) rather than @lru_cache: an operator editing
# deploy/nodes.local.yaml (adding a worker) must take effect without restarting
# the gateway. Re-reads only when the file's mtime moves.
_endpoint_cache: dict[tuple[str, float], dict[str, NodeEndpoint]] = {}


def load_node_endpoints() -> dict[str, NodeEndpoint]:
    """node_id -> NodeEndpoint from deploy/nodes.local.yaml. Empty dict if the file
    is absent (e.g. before an operator has configured deploy, or in CI)."""
    path = _resolve(get_settings().nodes_local_file)
    if not path.exists():
        return {}
    cache_key = (str(path), path.stat().st_mtime)
    cached = _endpoint_cache.get(cache_key)
    if cached is not None:
        return cached
    data = yaml.safe_load(path.read_text()) or {}
    out: dict[str, NodeEndpoint] = {}
    for n in data.get("nodes", []):
        node_id = n.get("node_id") or n.get("name")
        backend_url = n.get("backend_url")
        if not node_id or not backend_url:
            continue
        out[node_id] = NodeEndpoint(node_id=node_id, backend_url=backend_url,
                                     backend=n.get("backend", "lmstudio"))
    _endpoint_cache[cache_key] = out
    return out


def _clear_endpoint_cache() -> None:
    _endpoint_cache.clear()


# kept so callers (and tests) written against the old @lru_cache still work
load_node_endpoints.cache_clear = _clear_endpoint_cache


def dev_model_catalog() -> list[dict]:
    """Dev/test fallback for GET /v1/models when the control-plane is unreachable:
    read the same signed-manifest catalog dir the control-plane would serve from
    (models/manifests/*.yaml — committed, non-secret)."""
    manifests_dir = _REPO_ROOT / "models" / "manifests"
    out: list[dict] = []
    for f in sorted(manifests_dir.glob("*.yaml")):
        try:
            data = yaml.safe_load(f.read_text()) or {}
        except yaml.YAMLError:
            continue
        if "id" in data:
            out.append({"id": data["id"], "display_name": data.get("display_name", "")})
    return out
