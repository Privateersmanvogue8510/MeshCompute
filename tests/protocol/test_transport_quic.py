"""QUIC transport stream hygiene (PROTOCOL.md "Peer transport", INSTRUCTIONS §9).

Loopback only (127.0.0.1, ephemeral port) — no private IPs, no real peers, same
shape as the self-check at the bottom of transport_quic.py. Covers the two ways a
long-lived peer connection quietly rots: ping streams that are never finished, and
corrupt/undecodable frames that are dropped without a trace.
"""

from __future__ import annotations

import asyncio
import logging
import struct

import pytest
from meshcompute_node.transport_base import PeerAddress
from meshcompute_node.transport_quic import QuicTransport
from meshcompute_protocol import Frame, NodeIdentity, PacketClass


@pytest.fixture
async def peer_pair():
    """A dialed (client, server-side) connection pair over loopback QUIC."""
    ident_a, ident_b = NodeIdentity.generate(), NodeIdentity.generate()
    a = QuicTransport(ident_a, bind_port=0, bind_host="127.0.0.1")
    b = QuicTransport(ident_b, bind_port=0, bind_host="127.0.0.1")
    await a.start()
    await b.start()

    async def first_inbound():
        async for conn in b.accept():
            return conn

    accept_task = asyncio.ensure_future(first_inbound())
    conn_a = await a.dial(PeerAddress(node_id=ident_b.node_id, host="127.0.0.1",
                                      port=b.local_quic_port))
    conn_b = await asyncio.wait_for(accept_task, timeout=5.0)
    try:
        yield conn_a, conn_b
    finally:
        await conn_a.close()
        await b.stop()
        await a.stop()


async def test_repeated_ping_keeps_working(peer_pair):
    """rtt_ms() opens a stream per sample; the responder must finish its side so
    the stream is reaped. Fifty samples in a row must all measure, none hang.

    NOTE: on aioquic 1.3.0 defaults this passes with or without the write_eof()
    fix (the stream limit auto-raises and the client's own write_eof already lets
    aioquic discard the stream), so this is a smoke test, not a regression test.
    """
    conn_a, _ = peer_pair
    for _ in range(50):
        rtt = await asyncio.wait_for(conn_a.rtt_ms(), timeout=5.0)
        assert rtt > 0


async def test_corrupt_frame_is_dropped_and_logged(peer_pair, caplog):
    """A CRC mismatch must drop the stream (never surface as a real inbound
    stream) AND leave a warning — silent corruption is undebuggable."""
    conn_a, conn_b = peer_pair
    corrupt = bytearray(Frame(packet_class=PacketClass.ACTIVATION, request_id=1,
                              payload=b"activation-bytes").encode())
    corrupt[-1] ^= 0xFF  # flip a payload byte -> payload_crc16 no longer matches

    stream = await conn_a.open_stream()
    with caplog.at_level(logging.WARNING, logger="meshcompute_node.transport"):
        # send_frame() would re-encode a *valid* frame, so write the wire bytes
        # (length prefix + body) straight onto the stream.
        stream._writer.write(struct.pack(">I", len(corrupt)) + bytes(corrupt))  # noqa: SLF001
        await stream._writer.drain()  # noqa: SLF001

        with pytest.raises(TimeoutError):
            await conn_b.accept_stream(timeout=1.0)

    assert any("crc" in r.getMessage().lower() for r in caplog.records), caplog.text
    await stream.close()
