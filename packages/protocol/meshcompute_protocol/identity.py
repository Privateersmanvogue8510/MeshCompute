"""Node identity and signing (PROTOCOL.md "Peer identity", SECURITY.md).

Each node holds an Ed25519 keypair. node_id derives from the public key. Private
keys stay on the node and are never sent to the control plane. Capability
announcements and work receipts are signed so the control plane can attribute and
verify them without trusting transport alone.

Uses `cryptography` (already present on the box) rather than pulling in PyNaCl.
"""

from __future__ import annotations

import base64
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)
from cryptography.exceptions import InvalidSignature

from .content import blake3_hex


def _b64(b: bytes) -> str:
    return base64.urlsafe_b64encode(b).decode("ascii").rstrip("=")


def _unb64(s: str) -> bytes:
    pad = "=" * (-len(s) % 4)
    return base64.urlsafe_b64decode(s + pad)


def node_id_from_public(pub: Ed25519PublicKey) -> str:
    """Stable node id derived from the public key: 'nd_' + first 32 hex of BLAKE3(pub)."""
    raw = pub.public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)
    return "nd_" + blake3_hex(raw)[:32]


@dataclass
class NodeIdentity:
    private_key: Ed25519PrivateKey

    @property
    def public_key(self) -> Ed25519PublicKey:
        return self.private_key.public_key()

    @property
    def node_id(self) -> str:
        return node_id_from_public(self.public_key)

    @property
    def public_b64(self) -> str:
        return _b64(self.public_key.public_bytes(
            serialization.Encoding.Raw, serialization.PublicFormat.Raw))

    # --- signing ---
    def sign(self, data: bytes) -> str:
        return _b64(self.private_key.sign(data))

    def sign_json(self, obj: dict[str, Any]) -> str:
        return self.sign(canonical_json(obj))

    # --- persistence (never committed; see .gitignore) ---
    @classmethod
    def generate(cls) -> "NodeIdentity":
        return cls(Ed25519PrivateKey.generate())

    @classmethod
    def load_or_create(cls, path: str | Path) -> "NodeIdentity":
        path = Path(path)
        if path.exists():
            raw = _unb64(json.loads(path.read_text())["sk"])
            return cls(Ed25519PrivateKey.from_private_bytes(raw))
        ident = cls.generate()
        sk = ident.private_key.private_bytes(
            serialization.Encoding.Raw, serialization.PrivateFormat.Raw,
            serialization.NoEncryption())
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"sk": _b64(sk), "node_id": ident.node_id}))
        path.chmod(0o600)
        return ident


def canonical_json(obj: dict[str, Any]) -> bytes:
    """Deterministic JSON for signing: sorted keys, no spaces, UTF-8."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")


def verify(public_b64: str, data: bytes, signature_b64: str) -> bool:
    try:
        pub = Ed25519PublicKey.from_public_bytes(_unb64(public_b64))
        pub.verify(_unb64(signature_b64), data)
        return True
    except (InvalidSignature, ValueError):
        return False


def verify_json(public_b64: str, obj: dict[str, Any], signature_b64: str) -> bool:
    return verify(public_b64, canonical_json(obj), signature_b64)
