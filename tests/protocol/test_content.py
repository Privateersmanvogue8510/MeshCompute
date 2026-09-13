"""Content addressing invariants (PROTOCOL.md model chunk protocol, INSTRUCTIONS §10)."""

from __future__ import annotations

from meshcompute_protocol import chunk_id, verify_chunk


def test_chunk_id_is_deterministic():
    a = chunk_id("manifest-hash-1", "weights/file.gguf", 0, b"some chunk bytes")
    b = chunk_id("manifest-hash-1", "weights/file.gguf", 0, b"some chunk bytes")
    assert a == b


def test_verify_chunk_true_for_matching_bytes():
    expected = chunk_id("mh", "f.gguf", 4096, b"payload")
    assert verify_chunk(expected, "mh", "f.gguf", 4096, b"payload") is True


def test_verify_chunk_false_for_wrong_bytes():
    expected = chunk_id("mh", "f.gguf", 4096, b"payload")
    assert verify_chunk(expected, "mh", "f.gguf", 4096, b"tampered") is False


def test_different_offset_yields_different_id():
    a = chunk_id("mh", "f.gguf", 0, b"same bytes")
    b = chunk_id("mh", "f.gguf", 1024, b"same bytes")
    assert a != b


def test_different_manifest_hash_yields_different_id():
    a = chunk_id("mh-1", "f.gguf", 0, b"same bytes")
    b = chunk_id("mh-2", "f.gguf", 0, b"same bytes")
    assert a != b
