"""Content-addressed chunk store + BitTorrent-like single-file transfer.

PROTOCOL.md "Model chunk protocol", INSTRUCTIONS §10. Owns its own wire
sub-protocol for MODEL_CHUNK frames (nothing else in the repo depends on this
shape yet): request payload = offset (8 bytes, big-endian); response payload =
offset(8) + chunk_id (64 ascii-hex bytes) + chunk bytes. One Stream serves one
ShardFile.

Rarity-aware / multi-source peer selection is a documented TODO(phase-1.5) —
ShardFile only carries a whole-file artifact_root_hash today (see manifest.py),
not a published per-chunk hash list, so a leecher can't pick a rarest chunk
across sources yet. Single-source correctness (verify-before-commit, resume,
whole-file check) is real and covered below.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path

import blake3

from meshcompute_protocol import Frame, PacketClass, ShardFile
from meshcompute_protocol import chunk_id as compute_chunk_id
from meshcompute_protocol import verify_chunk

try:
    from meshcompute_node.transport_base import Stream
except ImportError:  # pragma: no cover - PYTHONPATH always has node/daemon in this repo
    Stream = object  # type: ignore[assignment,misc]

_OFFSET = struct.Struct(">Q")
_CHUNK_ID_HEX_LEN = 64  # blake3 default digest is 32 bytes -> 64 hex chars


def num_chunks_for(size_bytes: int, chunk_bytes: int) -> int:
    return -(-size_bytes // chunk_bytes) if size_bytes else 0


def _blake3_file(path: Path) -> str:
    h = blake3.blake3()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def pin_file(path: str | Path, manifest_hash: str, rfilename: str, chunk_bytes: int) -> str:
    """BLAKE3 of the whole file — the artifact_root_hash a manifest gets signed
    with. manifest_hash/rfilename/chunk_bytes aren't needed to hash the file
    (artifact_root_hash is whole-content BLAKE3, not a chunk-tree root, per
    manifest.py's ShardFile docstring); accepted here so callers that already
    have the shard context in hand don't need a second helper."""
    del manifest_hash, rfilename, chunk_bytes  # not needed for a whole-file hash (see above)
    return _blake3_file(Path(path))


class ChunkCache:
    """Content-addressed chunk store on disk. One file per chunk, named by
    chunk_id, so persistence across restarts is free (it's just files).

    ponytail: LRU order is tracked via each file's mtime (bumped on get/put)
    instead of a separate index file — one less thing that can drift out of
    sync with the directory contents after a crash. Upgrade to a real index
    (chunk_id -> size/last_access) if eviction scans over millions of chunks
    become slow.
    """

    def __init__(self, cache_dir: str | Path, max_bytes: int = 8 << 30):
        self.dir = Path(cache_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.max_bytes = max_bytes

    def _path(self, chunk_id: str) -> Path:
        return self.dir / chunk_id

    def has(self, chunk_id: str) -> bool:
        return self._path(chunk_id).is_file()

    def get(self, chunk_id: str) -> bytes | None:
        p = self._path(chunk_id)
        try:
            data = p.read_bytes()
        except FileNotFoundError:
            return None
        p.touch()  # bump LRU recency
        return data

    def put(self, chunk_id: str, data: bytes) -> None:
        p = self._path(chunk_id)
        tmp = p.with_suffix(p.suffix + ".tmp")
        tmp.write_bytes(data)
        tmp.replace(p)
        self._evict_if_needed()

    def _entries(self) -> list[Path]:
        return [f for f in self.dir.iterdir() if f.is_file() and not f.name.endswith(".tmp")]

    def total_bytes(self) -> int:
        return sum(f.stat().st_size for f in self._entries())

    def _evict_if_needed(self) -> None:
        files = self._entries()
        total = sum(f.stat().st_size for f in files)
        if total <= self.max_bytes:
            return
        for f in sorted(files, key=lambda f: f.stat().st_mtime):  # oldest-touched first
            if total <= self.max_bytes:
                break
            total -= f.stat().st_size
            f.unlink(missing_ok=True)


async def seed_file(stream: Stream, file_path: str | Path, manifest_hash: str,
                     shard: ShardFile) -> None:
    """Serve chunk requests for `shard` arriving on `stream` until the peer's
    side closes it (recv_frame raises EOFError)."""
    path = Path(file_path)
    total = shard.size_bytes
    with open(path, "rb") as f:
        while True:
            try:
                req = await stream.recv_frame(timeout=60.0)
            except EOFError:
                return
            (offset,) = _OFFSET.unpack(req.payload[:8])
            if offset < 0 or offset >= total:
                return  # stale/bad request from a confused peer; just stop serving
            length = min(shard.chunk_bytes, total - offset)
            f.seek(offset)
            data = f.read(length)
            cid = compute_chunk_id(manifest_hash, shard.rfilename, offset, data)
            payload = _OFFSET.pack(offset) + cid.encode("ascii") + data
            await stream.send_frame(Frame(packet_class=PacketClass.MODEL_CHUNK,
                                           request_id=req.request_id, payload=payload))


def _progress_path(dest_path: Path) -> Path:
    return dest_path.with_name(dest_path.name + ".mcprogress.json")


def _load_progress(dest_path: Path) -> dict[int, str]:
    p = _progress_path(dest_path)
    if not p.is_file():
        return {}
    try:
        raw = json.loads(p.read_text())
        return {int(k): v for k, v in raw.items()}
    except (json.JSONDecodeError, ValueError):
        return {}


def _save_progress(dest_path: Path, progress: dict[int, str]) -> None:
    _progress_path(dest_path).write_text(json.dumps({str(k): v for k, v in progress.items()}))


class ChunkVerifyError(RuntimeError):
    """A peer served a chunk whose bytes don't match its claimed chunk_id, and
    retries were exhausted."""


async def leech_file(
    stream: Stream,
    dest_path: str | Path,
    cache: ChunkCache,
    manifest_hash: str,
    shard: ShardFile,
    *,
    max_retries_per_chunk: int = 3,
) -> str:
    """Fetch every chunk of `shard` from the peer on `stream`.

    Resume: a sidecar `<dest>.mcprogress.json` records offset -> verified
    chunk_id as each chunk lands; a chunk already recorded there (and still in
    `cache`) is restored from cache instead of re-requested, so a restart after
    a partial run doesn't re-download what was already verified.
    ponytail: this sidecar-file resume is a Phase-1 substitute for a published
    per-chunk hash list (ShardFile only has a whole-file artifact_root_hash
    today); upgrade path is a Merkle chunk graph in the manifest (see
    manifest.py's ShardFile docstring) so a fresh peer could resume too.

    Every chunk is verify_chunk()'d BEFORE being committed to cache or written
    to disk; a mismatch is rejected and the same offset is re-requested up to
    max_retries_per_chunk times before raising ChunkVerifyError.

    Returns the reassembled file's actual BLAKE3 hex digest.
    """
    dest_path = Path(dest_path)
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    total = shard.size_bytes
    n_chunks = num_chunks_for(total, shard.chunk_bytes)
    progress = _load_progress(dest_path)

    if not dest_path.exists() or dest_path.stat().st_size != total:
        with open(dest_path, "wb") as out:
            out.truncate(total)

    req_id = 0
    with open(dest_path, "r+b") as out:
        for idx in range(n_chunks):
            offset = idx * shard.chunk_bytes
            length = min(shard.chunk_bytes, total - offset)

            cached_id = progress.get(offset)
            if cached_id is not None:
                cached_data = cache.get(cached_id)
                if cached_data is not None and len(cached_data) == length:
                    out.seek(offset)
                    out.write(cached_data)
                    continue
                progress.pop(offset, None)  # cache lost it; fall through and re-fetch

            data = None
            for attempt in range(max_retries_per_chunk):
                req_id += 1
                await stream.send_frame(Frame(packet_class=PacketClass.MODEL_CHUNK,
                                               request_id=req_id, payload=_OFFSET.pack(offset)))
                resp = await stream.recv_frame(timeout=30.0)
                resp_offset = _OFFSET.unpack(resp.payload[:8])[0]
                cid = resp.payload[8:8 + _CHUNK_ID_HEX_LEN].decode("ascii")
                chunk_data = resp.payload[8 + _CHUNK_ID_HEX_LEN:]
                if resp_offset == offset and verify_chunk(cid, manifest_hash, shard.rfilename,
                                                           offset, chunk_data):
                    data = chunk_data
                    cache.put(cid, chunk_data)
                    progress[offset] = cid
                    _save_progress(dest_path, progress)
                    break
                # reject + refetch: loop retries with a fresh request_id
            if data is None:
                raise ChunkVerifyError(
                    f"chunk at offset {offset} failed verification after "
                    f"{max_retries_per_chunk} attempts")
            out.seek(offset)
            out.write(data)

    digest = _blake3_file(dest_path)
    if not shard.artifact_root_hash.startswith("PENDING"):
        if digest != shard.artifact_root_hash:
            raise ChunkVerifyError(
                f"reassembled file blake3 {digest} != artifact_root_hash "
                f"{shard.artifact_root_hash}")
    _progress_path(dest_path).unlink(missing_ok=True)
    return digest


if __name__ == "__main__":
    import asyncio
    import os
    import sys
    import tempfile

    sys.path.insert(0, "packages/protocol")
    sys.path.insert(0, "node/daemon")
    from meshcompute_protocol import NodeIdentity  # noqa: E402
    from meshcompute_node.transport_base import PeerAddress  # noqa: E402
    from meshcompute_node.transport_quic import QuicTransport  # noqa: E402

    MANIFEST_HASH = "test-manifest-hash-0001"
    RFILENAME = "shard-000.gguf"
    CHUNK_BYTES = 1 << 20  # 1 MiB, so a 5 MiB file makes 5 chunks

    async def demo() -> None:
        tmp = Path(tempfile.mkdtemp(prefix="meshcompute-swarm-"))
        src_path = tmp / "source.bin"
        dest_path = tmp / "leeched.bin"
        cache_b = ChunkCache(tmp / "cache_b")

        size_bytes = 5 * (1 << 20) + 12345  # not a multiple of chunk size, on purpose
        src_path.write_bytes(os.urandom(size_bytes))
        root_hash = pin_file(src_path, MANIFEST_HASH, RFILENAME, CHUNK_BYTES)
        shard = ShardFile(rfilename=RFILENAME, size_bytes=size_bytes,
                           artifact_root_hash=root_hash, chunk_bytes=CHUNK_BYTES)
        print(f"seeding {size_bytes} bytes across {num_chunks_for(size_bytes, CHUNK_BYTES)} chunks")

        ident_a, ident_b = NodeIdentity.generate(), NodeIdentity.generate()
        a = QuicTransport(ident_a, bind_port=0, bind_host="127.0.0.1")
        b = QuicTransport(ident_b, bind_port=0, bind_host="127.0.0.1")
        await a.start()
        await b.start()

        async def accept_one():
            async for conn in a.accept():
                return conn

        accept_task = asyncio.ensure_future(accept_one())
        conn_b_to_a = await b.dial(PeerAddress(node_id=ident_a.node_id, host="127.0.0.1",
                                                port=a.local_quic_port))
        conn_a_side = await accept_task

        # Leecher (B) opens the stream and starts requesting; the seeder (A)
        # only sees it via accept_stream() once bytes actually hit the wire, so
        # start the leech first and race accept_stream() concurrently with it.
        leech_stream = await conn_b_to_a.open_stream()
        accept_stream_task = asyncio.ensure_future(conn_a_side.accept_stream(timeout=5.0))
        leech_task = asyncio.ensure_future(
            leech_file(leech_stream, dest_path, cache_b, MANIFEST_HASH, shard))
        seed_stream = await accept_stream_task
        seed_task = asyncio.ensure_future(seed_file(seed_stream, src_path, MANIFEST_HASH, shard))
        digest = await leech_task
        await leech_stream.close()  # EOFs the seeder's recv_frame so seed_task returns
        await seed_task

        assert digest == root_hash, f"blake3 mismatch: {digest} != {root_hash}"
        assert dest_path.read_bytes() == src_path.read_bytes(), "reassembled bytes differ"
        print(f"blake3 match OK: {digest}")
        print(f"cache_b holds {len(cache_b._entries())} chunks, "
              f"{cache_b.total_bytes()} bytes")

        # --- corrupted-chunk rejection: a seeder that always ships bit-flipped
        # bytes under an otherwise-honest chunk_id (simulates in-transit
        # corruption). leech_file must reject every attempt and raise.
        async def corrupting_seed(seed_stream) -> None:
            with open(src_path, "rb") as f:
                while True:
                    try:
                        req = await seed_stream.recv_frame(timeout=10.0)
                    except EOFError:
                        return
                    (offset,) = _OFFSET.unpack(req.payload[:8])
                    length = min(shard.chunk_bytes, size_bytes - offset)
                    f.seek(offset)
                    data = f.read(length)
                    cid = compute_chunk_id(MANIFEST_HASH, RFILENAME, offset, data)  # honest id
                    corrupted = bytes([data[0] ^ 0xFF]) + data[1:]                  # bad bytes
                    payload = _OFFSET.pack(offset) + cid.encode("ascii") + corrupted
                    await seed_stream.send_frame(
                        Frame(packet_class=PacketClass.MODEL_CHUNK, request_id=req.request_id,
                              payload=payload))

        leech_stream2 = await conn_b_to_a.open_stream()
        accept_stream_task2 = asyncio.ensure_future(conn_a_side.accept_stream(timeout=5.0))
        dest_path2 = tmp / "leeched_corrupt.bin"
        leech_task2 = asyncio.ensure_future(
            leech_file(leech_stream2, dest_path2, ChunkCache(tmp / "cache_c"),
                       MANIFEST_HASH, shard, max_retries_per_chunk=2))
        seed_stream2 = await accept_stream_task2
        corrupt_task = asyncio.ensure_future(corrupting_seed(seed_stream2))
        try:
            await leech_task2
            raise AssertionError("expected ChunkVerifyError for a permanently-corrupt chunk")
        except ChunkVerifyError as e:
            print(f"corrupted chunk correctly rejected: {e}")
        finally:
            await leech_stream2.close()
            await corrupt_task

        await conn_a_side.close()
        await conn_b_to_a.close()
        await a.stop()
        await b.stop()
        print("swarm.py self-check PASSED")

    asyncio.run(demo())
