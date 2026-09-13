"""`mesh manifest sign` / `mesh manifest verify` against the REAL catalog manifest.

The canonical signing scheme is the control plane's (app.py scan_manifests):
sign_json(manifest.model_dump(mode="json")) + manifest.manifest_hash(). The CLI must
produce and accept exactly that — a CLI-only scheme means operator-signed manifests
are rejected by the network (and vice versa).
"""

from __future__ import annotations

import json
from pathlib import Path

import yaml
from click.testing import CliRunner
from meshcompute_cli.main import cli
from meshcompute_protocol import ModelManifest, NodeIdentity, SignedManifest

REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_YAML = REPO_ROOT / "models" / "manifests" / "public-qwen3.8-27b-fable.yaml"


def _sign(tmp_path: Path) -> Path:
    out = tmp_path / "signed.json"
    result = CliRunner().invoke(cli, ["manifest", "sign", str(MANIFEST_YAML),
                                      "--key", str(tmp_path / "cli.key.json"),
                                      "--out", str(out)])
    assert result.exit_code == 0, result.output
    return out


def _verify(path: Path):
    return CliRunner().invoke(cli, ["manifest", "verify", str(path)])


def test_cli_signed_manifest_verifies(tmp_path):
    result = _verify(_sign(tmp_path))
    assert result.exit_code == 0, result.output
    assert "OK" in result.output


def test_tampered_manifest_body_fails_verification(tmp_path):
    signed_path = _sign(tmp_path)
    signed = json.loads(signed_path.read_text())
    signed["manifest"]["architecture"] = signed["manifest"]["architecture"] + "x"
    signed_path.write_text(json.dumps(signed))

    result = _verify(signed_path)
    assert result.exit_code == 1
    assert "FAIL" in result.output


def test_control_plane_signed_manifest_verifies_with_the_cli(tmp_path):
    """Built exactly the way apps/control-plane/.../app.py scan_manifests does."""
    manifest = ModelManifest.model_validate(yaml.safe_load(MANIFEST_YAML.read_text()))
    identity = NodeIdentity.generate()
    signed = SignedManifest(
        manifest=manifest, manifest_hash=manifest.manifest_hash(),
        public_b64=identity.public_b64,
        signature_b64=identity.sign_json(manifest.model_dump(mode="json")))

    path = tmp_path / "cp-signed.json"
    path.write_text(signed.model_dump_json(indent=2))

    result = _verify(path)
    assert result.exit_code == 0, result.output
