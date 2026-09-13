"""Node identity/signing invariants (PROTOCOL.md "Peer identity", SECURITY.md)."""

from __future__ import annotations

from meshcompute_protocol import NodeIdentity, canonical_json, node_id_from_public, verify, verify_json


def test_sign_and_verify_roundtrip():
    ident = NodeIdentity.generate()
    data = b"hello mesh"
    sig = ident.sign(data)
    assert verify(ident.public_b64, data, sig) is True


def test_tampered_data_fails_verification():
    ident = NodeIdentity.generate()
    sig = ident.sign(b"original payload")
    assert verify(ident.public_b64, b"tampered payload", sig) is False


def test_verify_json_roundtrip_and_tamper():
    ident = NodeIdentity.generate()
    obj = {"node_id": "nd_x", "ram_free_bytes": 123}
    sig = ident.sign_json(obj)
    assert verify_json(ident.public_b64, obj, sig) is True
    assert verify_json(ident.public_b64, {**obj, "ram_free_bytes": 124}, sig) is False


def test_node_id_derives_from_public_key():
    ident = NodeIdentity.generate()
    assert ident.node_id == node_id_from_public(ident.public_key)
    assert ident.node_id.startswith("nd_")
    # a different key must derive a different id
    other = NodeIdentity.generate()
    assert other.node_id != ident.node_id


def test_canonical_json_is_stable_and_sorted():
    a = {"b": 1, "a": 2, "c": {"z": 1, "y": 2}}
    b = {"c": {"y": 2, "z": 1}, "a": 2, "b": 1}
    assert canonical_json(a) == canonical_json(b)
    assert canonical_json(a) == b'{"a":2,"b":1,"c":{"y":2,"z":1}}'


def test_load_or_create_persists_the_same_identity(tmp_path):
    key_path = tmp_path / "identity.key.json"
    first = NodeIdentity.load_or_create(key_path)
    second = NodeIdentity.load_or_create(key_path)
    assert first.node_id == second.node_id
    assert first.public_b64 == second.public_b64
