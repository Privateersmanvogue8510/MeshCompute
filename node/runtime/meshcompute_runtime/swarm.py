"""BitTorrent-like single-file transfer over one peer stream (PROTOCOL.md
"Model chunk protocol", INSTRUCTIONS §10).

Wire sub-protocol on a Stream (both sides use Frame/PacketClass.MODEL_CHUNK
unless noted):
  1. handshake — leecher sends CONTROL {"op":"want","manifest_hash":..,"rfilename":..};
     seeder answers CONTROL {"op":"have"} and starts serving, or {"op":"nack"} and
     stops. This is what lets one inbound stream be matched to one local file.
  2. chunk loop — request payload = offset (8 bytes big-endian); response payload
     = offset(8) + chunk_id (64 ascii hex) + chunk bytes. chunk_id binds
     manifest_hash/path/offset/bytes (content.py), verified before any byte is
     kept. The leecher then checks the whole file's BLAKE3 against the manifest.

Resume: a sidecar `<dest>.mcprogress.json` records offset -> verified chunk_id
as chunks land. On restart each recorded chunk is re-hashed from the partial
file on disk and kept only if it still matches — no second copy of the model
in a chunk cache (a 24 GB model must cost 24 GB, not 48).

Rarity-aware multi-source selection is a documented TODO(phase-1.5): the
manifest carries only a whole-file artifact_root_hash today, so a leecher
can't pick the rarest chunk across peers. Single-source correctness is real.
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
HANDSHAKE_TIMEOUT_S = 15.0


class ChunkVerifyError(RuntimeError):
    """A peer served a chunk whose bytes don't match its claimed chunk_id, and
    retries were exhausted — or the reassembled file's BLAKE3 doesn't match."""


class PeerLacksFile(RuntimeError):
    """The peer answered the handshake with nack (it doesn't hold this file)."""


def num_chunks_for(size_bytes: int, chunk_bytes: int) -> int:
    return -(-size_bytes // chunk_bytes) if size_bytes else 0


def _blake3_file(path: Path) -> str:
    h = blake3.blake3()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def pin_file(path: str | Path) -> str:
    """BLAKE3 of the whole file — the artifact_root_hash a manifest is signed with."""
    return _blake3_file(Path(path))


def _shard(shard: ShardFile | dict) -> ShardFile:
    return shard if isinstance(shard, ShardFile) else ShardFile.model_validate(shard)


def _control(payload: dict, request_id: int = 0) -> Frame:
    return Frame(packet_class=PacketClass.CONTROL, request_id=request_id,
                 payload=json.dumps(payload).encode("utf-8"))


# --------------------------------------------------------------------------- seeder
async def seed_file(stream: Stream, file_path: str | Path, manifest_hash: str,
                    shard: ShardFile | dict) -> int:
    """Serve chunk requests for `shard` on `stream` until the peer closes it.
    Handshake is assumed already consumed (see serve_chunk_stream). Returns
    bytes served (for the work receipt's model_shard_bytes_served)."""
    shard = _shard(shard)
    total = shard.size_bytes
    served = 0
    with open(Path(file_path), "rb") as f:
        while True:
            try:
                req = await stream.recv_frame(timeout=60.0)
            except (EOFError, TimeoutError):
                return served
            if req.packet_class != PacketClass.MODEL_CHUNK or len(req.payload) < 8:
                return served
            (offset,) = _OFFSET.unpack(req.payload[:8])
            if offset < 0 or offset >= total or offset % shard.chunk_bytes:
                return served  # stale/bad request from a confused peer; stop serving
            length = min(shard.chunk_bytes, total - offset)
            f.seek(offset)
            data = f.read(length)
            cid = compute_chunk_id(manifest_hash, shard.rfilename, offset, data)
            await stream.send_frame(Frame(packet_class=PacketClass.MODEL_CHUNK,
                                          request_id=req.request_id,
                                          payload=_OFFSET.pack(offset) + cid.encode("ascii") + data))
            served += len(data)


async def serve_chunk_stream(stream: Stream, index: dict[tuple[str, str], tuple[Path, dict]]) -> int:
    """Seeder-side entry for one inbound stream: read the handshake, look the
    file up in `index` ((manifest_hash, rfilename) -> (path, shard)), answer
    have/nack, serve. Never raises on a misbehaving peer — just stops."""
    try:
        hello = await stream.recv_frame(timeout=HANDSHAKE_TIMEOUT_S)
        want = json.loads(hello.payload.decode("utf-8"))
        key = (str(want["manifest_hash"]), str(want["rfilename"]))
    except (EOFError, TimeoutError, ValueError, KeyError, TypeError):
        return 0
    hit = index.get(key)
    if hit is None or hello.packet_class != PacketClass.CONTROL or want.get("op") != "want":
        try:
            await stream.send_frame(_control({"op": "nack"}, hello.request_id))
        finally:
            await stream.close()
        return 0
    path, shard = hit
    await stream.send_frame(_control({"op": "have"}, hello.request_id))
    try:
        return await seed_file(stream, path, key[0], shard)
    finally:
        await stream.close()


# --------------------------------------------------------------------------- leecher
def _progress_path(dest_path: Path) -> Path:
    return dest_path.with_name(dest_path.name + ".mcprogress.json")


def _load_progress(dest_path: Path) -> dict[int, str]:
    p = _progress_path(dest_path)
    if not p.is_file():
        return {}
    try:
        return {int(k): v for k, v in json.loads(p.read_text()).items()}
    except (json.JSONDecodeError, ValueError, AttributeError):
        return {}


def _save_progress(dest_path: Path, progress: dict[int, str]) -> None:
    _progress_path(dest_path).write_text(json.dumps({str(k): v for k, v in progress.items()}))


async def leech_file(stream: Stream, dest_path: str | Path, manifest_hash: str,
                     shard: ShardFile | dict, *, max_retries_per_chunk: int = 3,
                     progress_cb=None) -> str:
    """Fetch every chunk of `shard` from the peer on `stream` into `dest_path`.

    Every chunk is verify_chunk()'d BEFORE being written; a mismatch is
    re-requested up to max_retries_per_chunk times, then ChunkVerifyError.
    Resumes from `<dest>.mcprogress.json` by re-hashing already-landed chunks
    on disk. Returns the reassembled file's BLAKE3 hex digest (raises
    ChunkVerifyError if the manifest's artifact_root_hash disagrees).
    """
    shard = _shard(shard)
    dest_path = Path(dest_path)
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    total = shard.size_bytes
    n_chunks = num_chunks_for(total, shard.chunk_bytes)

    await stream.send_frame(_control({"op": "want", "manifest_hash": manifest_hash,
                                      "rfilename": shard.rfilename}))
    reply = await stream.recv_frame(timeout=HANDSHAKE_TIMEOUT_S)
    try:
        ok = reply.packet_class == PacketClass.CONTROL and json.loads(reply.payload)["op"] == "have"
    except (ValueError, KeyError, TypeError):
        ok = False
    if not ok:
        raise PeerLacksFile(f"peer does not hold {shard.rfilename} for {manifest_hash[:16]}")

    progress = _load_progress(dest_path)
    if not dest_path.exists() or dest_path.stat().st_size != total:
        progress = {}
        with open(dest_path, "wb") as out:
            out.truncate(total)

    req_id = 0
    with open(dest_path, "r+b") as out:
        for idx in range(n_chunks):
            offset = idx * shard.chunk_bytes
            length = min(shard.chunk_bytes, total - offset)

            recorded = progress.get(offset)
            if recorded is not None:
                out.seek(offset)
                if verify_chunk(recorded, manifest_hash, shard.rfilename, offset, out.read(length)):
                    continue  # already on disk and still intact
                progress.pop(offset, None)

            data = None
            for _attempt in range(max_retries_per_chunk):
                req_id += 1
                await stream.send_frame(Frame(packet_class=PacketClass.MODEL_CHUNK,
                                              request_id=req_id, payload=_OFFSET.pack(offset)))
                resp = await stream.recv_frame(timeout=30.0)
                if resp.packet_class != PacketClass.MODEL_CHUNK or len(resp.payload) < 8 + _CHUNK_ID_HEX_LEN:
                    continue
                resp_offset = _OFFSET.unpack(resp.payload[:8])[0]
                cid = resp.payload[8:8 + _CHUNK_ID_HEX_LEN].decode("ascii", errors="replace")
                chunk_data = resp.payload[8 + _CHUNK_ID_HEX_LEN:]
                if resp_offset == offset and len(chunk_data) == length and \
                        verify_chunk(cid, manifest_hash, shard.rfilename, offset, chunk_data):
                    data = chunk_data
                    break
            if data is None:
                raise ChunkVerifyError(f"chunk at offset {offset} failed verification after "
                                       f"{max_retries_per_chunk} attempts")
            out.seek(offset)
            out.write(data)
            progress[offset] = cid
            _save_progress(dest_path, progress)
            if progress_cb is not None:
                progress_cb(idx + 1, n_chunks)

    digest = _blake3_file(dest_path)
    if not shard.artifact_root_hash.startswith("PENDING") and digest != shard.artifact_root_hash:
        raise ChunkVerifyError(f"reassembled file blake3 {digest} != artifact_root_hash "
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
    CHUNK_BYTES = 1 << 20

    async def demo() -> None:
        tmp = Path(tempfile.mkdtemp(prefix="meshcompute-swarm-"))
        src_path = tmp / "source.bin"
        size_bytes = 5 * (1 << 20) + 12345
        src_path.write_bytes(os.urandom(size_bytes))
        root_hash = pin_file(src_path)
        shard = ShardFile(rfilename=RFILENAME, size_bytes=size_bytes,
                          artifact_root_hash=root_hash, chunk_bytes=CHUNK_BYTES)
        index = {(MANIFEST_HASH, RFILENAME): (src_path, shard.model_dump())}

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

        async def seed_next() -> int:
            stream = await conn_a_side.accept_stream(timeout=5.0)
            return await serve_chunk_stream(stream, index)

        dest = tmp / "leeched.bin"
        seed_task = asyncio.ensure_future(seed_next())
        digest = await leech_file(await conn_b_to_a.open_stream(), dest, MANIFEST_HASH, shard)
        served = await seed_task
        assert digest == root_hash and dest.read_bytes() == src_path.read_bytes()
        assert served == size_bytes
        print(f"seed->leech OK: {size_bytes} bytes, blake3 {digest[:16]}…")

        # nack: asking for a file the seeder doesn't have
        seed_task = asyncio.ensure_future(seed_next())
        try:
            await leech_file(await conn_b_to_a.open_stream(), tmp / "x.bin", "other-hash", shard)
            raise AssertionError("expected PeerLacksFile")
        except PeerLacksFile as e:
            print(f"nack OK: {e}")
        await seed_task

        # resume: keep 3 verified chunks on disk + sidecar, corrupt one of them
        dest2 = tmp / "resume.bin"
        prog = {}
        with open(dest2, "wb") as out:
            out.truncate(size_bytes)
            for i in range(3):
                off = i * CHUNK_BYTES
                data = src_path.read_bytes()[off:off + CHUNK_BYTES]
                out.seek(off)
                out.write(data)
                prog[off] = compute_chunk_id(MANIFEST_HASH, RFILENAME, off, data)
        _save_progress(dest2, prog)
        with open(dest2, "r+b") as out:  # chunk 1 silently corrupted on disk
            out.seek(CHUNK_BYTES + 7)
            out.write(b"\x00\xff")
        requests = []
        seed_task = asyncio.ensure_future(seed_next())
        digest2 = await leech_file(await conn_b_to_a.open_stream(), dest2, MANIFEST_HASH, shard,
                                   progress_cb=lambda done, n: requests.append(done))
        await seed_task
        assert digest2 == root_hash
        assert len(requests) == 4, f"expected 4 re-fetched (1 corrupt + 3 missing), got {len(requests)}"
        print("resume OK: 2 intact chunks kept, corrupt chunk re-fetched")

        # corrupt-in-transit seeder: every chunk bit-flipped under an honest id
        async def corrupting_seed() -> None:
            stream = await conn_a_side.accept_stream(timeout=5.0)
            await stream.recv_frame(timeout=5.0)
            await stream.send_frame(_control({"op": "have"}))
            with open(src_path, "rb") as f:
                while True:
                    try:
                        req = await stream.recv_frame(timeout=10.0)
                    except EOFError:
                        return
                    (offset,) = _OFFSET.unpack(req.payload[:8])
                    f.seek(offset)
                    data = f.read(min(CHUNK_BYTES, size_bytes - offset))
                    cid = compute_chunk_id(MANIFEST_HASH, RFILENAME, offset, data)
                    bad = bytes([data[0] ^ 0xFF]) + data[1:]
                    await stream.send_frame(Frame(packet_class=PacketClass.MODEL_CHUNK,
                                                  request_id=req.request_id,
                                                  payload=_OFFSET.pack(offset) + cid.encode() + bad))

        corrupt_task = asyncio.ensure_future(corrupting_seed())
        s3 = await conn_b_to_a.open_stream()
        try:
            await leech_file(s3, tmp / "corrupt.bin", MANIFEST_HASH, shard, max_retries_per_chunk=2)
            raise AssertionError("expected ChunkVerifyError")
        except ChunkVerifyError as e:
            print(f"corrupt chunk rejected: {e}")
        finally:
            await s3.close()
            await corrupt_task

        await conn_a_side.close()
        await conn_b_to_a.close()
        await a.stop()
        await b.stop()
        print("swarm.py self-check PASSED")

    asyncio.run(demo())
