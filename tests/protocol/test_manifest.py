"""Signed model manifest invariants (INSTRUCTIONS §2.2, SECURITY.md model boundary)."""

from __future__ import annotations

from pathlib import Path

import yaml
from meshcompute_protocol import ModelManifest, NodeIdentity, SignedManifest, verify_json

REPO_ROOT = Path(__file__).resolve().parents[2]
QWEN_MANIFEST = REPO_ROOT / "models" / "manifests" / "public-qwen3.8-27b-fable.yaml"


def test_manifest_hash_stable_across_reloads(manifest_factory):
    manifest = manifest_factory()
    first_hash = manifest.manifest_hash()

    reloaded = ModelManifest.model_validate(manifest.model_dump(mode="json"))
    assert reloaded.manifest_hash() == first_hash


def test_signed_manifest_verifies_ok(manifest_factory):
    manifest = manifest_factory()
    ident = NodeIdentity.generate()
    body = manifest.model_dump(mode="json")
    signature = ident.sign_json(body)
    signed = SignedManifest(manifest=manifest, manifest_hash=manifest.manifest_hash(),
                            public_b64=ident.public_b64, signature_b64=signature)

    assert signed.manifest_hash == manifest.manifest_hash()
    assert verify_json(signed.public_b64, signed.manifest.model_dump(mode="json"),
                       signed.signature_b64) is True


def test_one_byte_tamper_breaks_hash_and_signature(manifest_factory):
    manifest = manifest_factory()
    original_hash = manifest.manifest_hash()
    ident = NodeIdentity.generate()
    signature = ident.sign_json(manifest.model_dump(mode="json"))

    # A single-character tamper of the manifest body...
    tampered = manifest.model_copy(update={"safety_notes": manifest.safety_notes + "x"})

    # ...must change the content-addressed hash...
    assert tampered.manifest_hash() != original_hash
    # ...and must fail verification against the original signature (tampered
    # manifest rejected, per SECURITY.md's "hash mismatch" test requirement).
    assert verify_json(ident.public_b64, tampered.model_dump(mode="json"), signature) is False


def test_real_qwen_manifest_parses_and_hashes():
    raw = yaml.safe_load(QWEN_MANIFEST.read_text())
    manifest = ModelManifest.model_validate(raw)
    assert manifest.id == "public/qwen3.8-27b-fable"
    h = manifest.manifest_hash()
    assert len(h) == 64  # BLAKE3 hex digest
    int(h, 16)  # must be valid hex
