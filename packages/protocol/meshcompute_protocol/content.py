"""Content addressing (PROTOCOL.md "Model chunk protocol", INSTRUCTIONS §10).

Model artifacts and shards are identified by BLAKE3 hashes, not by name or claim.
A chunk id binds the manifest hash, the file path, the offset and the bytes, so the
same bytes at a different position are a different id and cannot be substituted.
"""

from __future__ import annotations

import blake3

# 1 MiB default chunk. Configurable per manifest; kept fixed for the POC.
DEFAULT_CHUNK_BYTES = 1 << 20


def blake3_hex(data: bytes) -> str:
    return blake3.blake3(data).hexdigest()


def chunk_id(manifest_hash: str, file_path: str, offset: int, chunk_bytes: bytes) -> str:
    """chunk_id = BLAKE3(manifest_hash || file_path || offset || chunk_bytes).

    Mirrors PROTOCOL.md exactly so any implementation derives the same id.
    """
    h = blake3.blake3()
    h.update(manifest_hash.encode("utf-8"))
    h.update(b"\x00")
    h.update(file_path.encode("utf-8"))
    h.update(b"\x00")
    h.update(offset.to_bytes(8, "big"))
    h.update(b"\x00")
    h.update(chunk_bytes)
    return h.hexdigest()


def verify_chunk(expected_id: str, manifest_hash: str, file_path: str, offset: int,
                 chunk_bytes: bytes) -> bool:
    """Verify a received chunk BEFORE committing it to cache (SECURITY.md model boundary)."""
    return chunk_id(manifest_hash, file_path, offset, chunk_bytes) == expected_id
