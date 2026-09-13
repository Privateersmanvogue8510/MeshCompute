"""~/.mesh/config.json load/save (gateway_url, control_url, token) + env overrides."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

CONFIG_DIR = Path.home() / ".mesh"
CONFIG_PATH = CONFIG_DIR / "config.json"

DEFAULT_GATEWAY_URL = "http://127.0.0.1:8081"
DEFAULT_CONTROL_URL = "http://127.0.0.1:8080"


def load_config() -> dict[str, Any]:
    if CONFIG_PATH.exists():
        try:
            return json.loads(CONFIG_PATH.read_text())
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


def save_config(cfg: dict[str, Any]) -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.write_text(json.dumps(cfg, indent=2))
    CONFIG_PATH.chmod(0o600)


def get_gateway_url(cfg: dict[str, Any] | None = None) -> str:
    return os.environ.get("MESH_GATEWAY_URL") or (cfg or load_config()).get("gateway_url") or DEFAULT_GATEWAY_URL


def get_control_url(cfg: dict[str, Any] | None = None) -> str:
    return os.environ.get("MESH_CONTROL_URL") or (cfg or load_config()).get("control_url") or DEFAULT_CONTROL_URL


def get_token(cfg: dict[str, Any] | None = None) -> str | None:
    return (cfg or load_config()).get("token")
