"""Peer model distribution over real loopback QUIC: handshake, verified chunk
transfer, nack for unknown files, resume with on-disk re-verification, and
rejection of a seeder that corrupts bytes in transit (INSTRUCTIONS §10, §23)."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

import pytest
from meshcompute_node.transport_base import PeerAddress
from meshcompute_node.transport_quic import QuicTransport
from meshcompute_protocol import Frame, NodeIdentity, PacketClass, ShardFile
from meshcompute_protocol import chunk_id as compute_chunk_id
from meshcompute_runtime import swarm

MH = "test-manifest-hash-0001"
NAME = "shard-000.gguf"
CHUNK = 1 << 16   # 64 KiB chunks keep the test fast; 6 chunks + a tail


@pytest.fixture
async def pair(tmp_path):
    ident_a, ident_b = NodeIdentity.generate(), NodeIdentity.generate()
    a = QuicTransport(ident_a, bind_port=0, bind_host="127.0.0.1")
    b = QuicTransport(ident_b, bind_port=0, bind_host="127.0.0.1")
    await a.start()
    await b.start()

    async def accept_one():
        async for conn in a.accept():
            return conn

    accept_task = asyncio.ensure_future(accept_one())
    conn_b = await b.dial(PeerAddress(node_id=ident_a.node_id, host="127.0.0.1", port=a.local_quic_port))
    conn_a = await accept_task

    src = tmp_path / "source.bin"
    size = 6 * CHUNK + 12345
    src.write_bytes(os.urandom(size))
    shard = ShardFile(rfilename=NAME, size_bytes=size, artifact_root_hash=swarm.pin_file(src),
                      chunk_bytes=CHUNK)
    index = {(MH, NAME): (src, shard.model_dump())}

    async def seed_next():
        stream = await conn_a.accept_stream(timeout=5.0)
        return await swarm.serve_chunk_stream(stream, index)

    yield {"src": src, "shard": shard, "conn_a": conn_a, "conn_b": conn_b, "seed_next": seed_next,
           "tmp": tmp_path}
    await conn_a.close()
    await conn_b.close()
    await a.stop()
    await b.stop()


async def test_seed_to_leech_verified_end_to_end(pair):
    seed = asyncio.ensure_future(pair["seed_next"]())
    dest = pair["tmp"] / "leeched.bin"
    digest = await swarm.leech_file(await pair["conn_b"].open_stream(), dest, MH, pair["shard"])
    served = await seed
    assert digest == pair["shard"].artifact_root_hash
    assert dest.read_bytes() == pair["src"].read_bytes()
    assert served == pair["shard"].size_bytes
    assert not (pair["tmp"] / "leeched.bin.mcprogress.json").exists()   # sidecar cleaned up


async def test_seeder_nacks_unknown_file(pair):
    seed = asyncio.ensure_future(pair["seed_next"]())
    with pytest.raises(swarm.PeerLacksFile):
        await swarm.leech_file(await pair["conn_b"].open_stream(), pair["tmp"] / "x.bin",
                               "some-other-manifest", pair["shard"])
    assert await seed == 0


async def test_resume_keeps_intact_chunks_and_refetches_corrupt_one(pair):
    dest = pair["tmp"] / "resume.bin"
    data = pair["src"].read_bytes()
    progress: dict[int, str] = {}
    with open(dest, "wb") as out:
        out.truncate(pair["shard"].size_bytes)
        for i in range(3):                      # 3 chunks "already landed"
            off = i * CHUNK
            out.seek(off)
            out.write(data[off:off + CHUNK])
            progress[off] = compute_chunk_id(MH, NAME, off, data[off:off + CHUNK])
    swarm._save_progress(dest, progress)
    with open(dest, "r+b") as out:            # ...one of them silently rotted on disk
        out.seek(CHUNK + 7)
        out.write(b"\x00\xff")

    fetched: list[int] = []
    seed = asyncio.ensure_future(pair["seed_next"]())
    digest = await swarm.leech_file(await pair["conn_b"].open_stream(), dest, MH, pair["shard"],
                                    progress_cb=lambda done, n: fetched.append(done))
    await seed
    assert digest == pair["shard"].artifact_root_hash
    # 7 chunks total, 2 intact kept, 1 corrupt + 4 missing re-fetched
    assert len(fetched) == 5


async def test_corrupting_seeder_is_rejected(pair):
    shard, src = pair["shard"], pair["src"]

    async def corrupting_seed():
        stream = await pair["conn_a"].accept_stream(timeout=5.0)
        await stream.recv_frame(timeout=5.0)
        await stream.send_frame(swarm._control({"op": "have"}))
        with open(src, "rb") as f:
            while True:
                try:
                    req = await stream.recv_frame(timeout=5.0)
                except (EOFError, TimeoutError):
                    return
                (offset,) = swarm._OFFSET.unpack(req.payload[:8])
                f.seek(offset)
                chunk = f.read(min(CHUNK, shard.size_bytes - offset))
                cid = compute_chunk_id(MH, NAME, offset, chunk)          # honest id...
                bad = bytes([chunk[0] ^ 0xFF]) + chunk[1:]               # ...corrupt bytes
                await stream.send_frame(Frame(packet_class=PacketClass.MODEL_CHUNK,
                                              request_id=req.request_id,
                                              payload=swarm._OFFSET.pack(offset) + cid.encode() + bad))

    task = asyncio.ensure_future(corrupting_seed())
    stream = await pair["conn_b"].open_stream()
    try:
        with pytest.raises(swarm.ChunkVerifyError):
            await swarm.leech_file(stream, pair["tmp"] / "bad.bin", MH, shard, max_retries_per_chunk=2)
    finally:
        await stream.close()
        await task


def test_pin_file_is_whole_file_blake3(tmp_path: Path):
    import blake3
    p = tmp_path / "f"
    p.write_bytes(b"hello mesh")
    assert swarm.pin_file(p) == blake3.blake3(b"hello mesh").hexdigest()
