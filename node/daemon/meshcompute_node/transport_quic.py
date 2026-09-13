"""QUIC transport over aioquic — implements transport_base's ABCs.

PROTOCOL.md "Peer transport" / INSTRUCTIONS §9. NAT traversal order is:
  1. direct known-address QUIC        <- fully implemented, tested on loopback/LAN
  2. hole-punched direct QUIC         <- best-effort; needs two real NATed peers
  3. trusted relay                    <- TODO(phase-1.5) stub, see _dial_via_rendezvous

Wire framing on every stream: 4-byte big-endian length prefix + Frame.encode().
Every peer connection also gets an application-level "ping" convention used by
rtt_ms(): open a stream, send a CONTROL frame with payload b"PING", the remote
side's inbound-stream dispatcher recognizes it and echoes a CONTROL b"PONG"
frame back on the same stream without handing it to accept_stream().
"""

from __future__ import annotations

import asyncio
import contextlib
import datetime
import ssl
import struct
import time
from typing import AsyncIterator, Callable

from cryptography import x509
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.x509.oid import NameOID

from aioquic.asyncio import connect as aioquic_connect
from aioquic.asyncio import serve as aioquic_serve
from aioquic.asyncio.protocol import QuicConnectionProtocol
from aioquic.asyncio.server import QuicServer
from aioquic.quic import events as quic_events
from aioquic.quic.configuration import QuicConfiguration

from meshcompute_protocol import Frame, PacketClass

try:
    from .transport_base import PeerAddress, PeerConnection, Stream, Transport
except ImportError:  # running as a plain script (python path/to/transport_quic.py)
    from meshcompute_node.transport_base import PeerAddress, PeerConnection, Stream, Transport

ALPN = "meshcompute/1"
_LEN = struct.Struct(">I")
_MAX_FRAME_ON_WIRE = (256 << 20) + 4096  # Frame.MAX_PAYLOAD + header slack
_LOOPBACK_HOSTS = ("127.0.0.1", "::1", "localhost")


def _generate_self_signed_cert(common_name: str) -> tuple[x509.Certificate, ec.EllipticCurvePrivateKey]:
    """Per-session self-signed cert so a node needs no external PKI to speak TLS.

    TODO(phase-1.5): bind this cert (or a cert extension) to the node's Ed25519
    identity so a dialer can confirm it reached the node_id it asked for, instead
    of trusting whoever answers the UDP port. Today verify_mode=CERT_NONE means
    transport encryption is real but peer *authentication* happens one layer up
    (rendezvous tickets / signed manifests), not at the TLS handshake.
    """
    key = ec.generate_private_key(ec.SECP256R1())
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, common_name[:64])])
    now = datetime.datetime.now(datetime.timezone.utc)
    cert = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - datetime.timedelta(minutes=5))
        .not_valid_after(now + datetime.timedelta(days=365))
        .sign(key, hashes.SHA256())
    )
    return cert, key


async def _read_frame(reader: asyncio.StreamReader) -> Frame:
    try:
        header = await reader.readexactly(4)
    except asyncio.IncompleteReadError:
        raise EOFError("stream closed") from None
    (length,) = _LEN.unpack(header)
    if length > _MAX_FRAME_ON_WIRE:
        raise ValueError(f"frame length {length} exceeds bound {_MAX_FRAME_ON_WIRE}")
    try:
        body = await reader.readexactly(length)
    except asyncio.IncompleteReadError:
        raise EOFError("stream closed mid-frame") from None
    return Frame.decode(body)


async def _write_frame(writer: asyncio.StreamWriter, frame: Frame) -> None:
    data = frame.encode()
    writer.write(_LEN.pack(len(data)) + data)
    await writer.drain()


class QuicStream(Stream):
    """One QUIC stream, framed with a 4-byte length prefix around Frame.encode().

    Backpressure: we never buffer more than one frame ahead of the reader (we
    read the length, then read exactly that many bytes) and reject any length
    prefix bigger than Frame's own MAX_PAYLOAD bound before allocating for it —
    that's the DoS/backpressure bound; per-stream byte-level flow control below
    that is aioquic's own QUIC stream flow control (max_stream_data).
    """

    def __init__(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter,
                 preloaded: Frame | None = None):
        self._reader = reader
        self._writer = writer
        self._preloaded = preloaded

    async def send_frame(self, frame: Frame) -> None:
        await _write_frame(self._writer, frame)

    async def recv_frame(self, timeout: float | None = None) -> Frame:
        if self._preloaded is not None:
            frame, self._preloaded = self._preloaded, None
            return frame
        try:
            return await asyncio.wait_for(_read_frame(self._reader), timeout)
        except asyncio.TimeoutError:
            raise TimeoutError("recv_frame timed out") from None

    async def close(self) -> None:
        with contextlib.suppress(Exception):
            self._writer.write_eof()
        with contextlib.suppress(Exception):
            await self._writer.drain()


class _MeshProtocol(QuicConnectionProtocol):
    """One QUIC connection. Owns inbound-stream dispatch (ping auto-reply +
    queueing everything else for accept_stream()) and remembers the remote addr
    (aioquic doesn't expose it on the protocol itself)."""

    def __init__(self, *args, on_handshake: Callable[["_MeshProtocol"], None] | None = None,
                 **kwargs):
        kwargs.pop("stream_handler", None)  # we install our own below, always
        super().__init__(*args, **kwargs)
        self._stream_handler = self._on_new_stream
        self.inbound_streams: asyncio.Queue[Stream] = asyncio.Queue()
        self.remote_addr: tuple[str, int] | None = None
        self._on_handshake = on_handshake
        self._handshake_announced = False

    def datagram_received(self, data, addr) -> None:  # noqa: D102 (base has no docstring either)
        if self.remote_addr is None:
            self.remote_addr = addr
        super().datagram_received(data, addr)

    def quic_event_received(self, event: quic_events.QuicEvent) -> None:
        super().quic_event_received(event)
        if isinstance(event, quic_events.HandshakeCompleted) and not self._handshake_announced:
            self._handshake_announced = True
            if self._on_handshake is not None:
                self._on_handshake(self)

    def _on_new_stream(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        asyncio.ensure_future(self._dispatch_stream(reader, writer))

    async def _dispatch_stream(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            frame = await _read_frame(reader)
        except (EOFError, ValueError):
            return
        if frame.packet_class == PacketClass.CONTROL and frame.payload == b"PING":
            pong = Frame(packet_class=PacketClass.CONTROL, request_id=frame.request_id, payload=b"PONG")
            with contextlib.suppress(Exception):
                await _write_frame(writer, pong)
            return
        self.inbound_streams.put_nowait(QuicStream(reader, writer, preloaded=frame))


def _path_type_for(host: str) -> str:
    return "loopback" if host in _LOOPBACK_HOSTS else "direct"


class QuicPeerConnection(PeerConnection):
    def __init__(self, protocol: _MeshProtocol, remote_addr: tuple[str, int],
                 path_type: str = "direct", close_stack: contextlib.AsyncExitStack | None = None):
        self._protocol = protocol
        self._remote_addr = remote_addr
        self._path_type = path_type
        self._close_stack = close_stack

    async def open_stream(self) -> Stream:
        reader, writer = await self._protocol.create_stream()
        return QuicStream(reader, writer)

    async def accept_stream(self, timeout: float | None = None) -> Stream:
        """Extension beyond the PeerConnection ABC: retrieve the next stream the
        remote side opened on us. The base ABC only defines outbound
        open_stream(); something has to let the accepting side see inbound
        streams (used by the loopback test and by swarm seeding/leeching)."""
        coro = self._protocol.inbound_streams.get()
        if timeout is None:
            return await coro
        try:
            return await asyncio.wait_for(coro, timeout)
        except asyncio.TimeoutError:
            raise TimeoutError("no inbound stream within timeout") from None

    async def rtt_ms(self) -> float:
        stream = await self.open_stream()
        try:
            start = time.monotonic()
            await stream.send_frame(Frame(packet_class=PacketClass.CONTROL, payload=b"PING"))
            pong = await stream.recv_frame(timeout=10.0)
            elapsed_ms = (time.monotonic() - start) * 1000.0
        finally:
            await stream.close()
        if pong.packet_class != PacketClass.CONTROL or pong.payload != b"PONG":
            raise RuntimeError("peer did not answer ping with a pong")
        return elapsed_ms

    async def close(self) -> None:
        if self._close_stack is not None:
            with contextlib.suppress(Exception):
                await self._close_stack.aclose()
        else:
            self._protocol.close()
            with contextlib.suppress(Exception):
                await self._protocol.wait_closed()

    @property
    def path_type(self) -> str:
        return self._path_type


class QuicTransport(Transport):
    """Node-level QUIC transport. Prefers outbound dials (see transport_base
    docstring); accept() serves whoever dials us."""

    def __init__(self, identity, bind_port: int = 0, rendezvous_client=None,
                 bind_host: str = "0.0.0.0"):
        self._identity = identity
        self._bind_port = bind_port
        self._bind_host = bind_host
        self._rendezvous = rendezvous_client
        self._server: QuicServer | None = None
        self._accept_queue: asyncio.Queue[PeerConnection] = asyncio.Queue()
        node_label = getattr(identity, "node_id", None) or "meshcompute-node"
        self._cert, self._key = _generate_self_signed_cert(node_label)
        self._bound_port = 0

    async def start(self) -> None:
        cfg = QuicConfiguration(is_client=False, alpn_protocols=[ALPN])
        cfg.certificate = self._cert
        cfg.private_key = self._key
        self._server = await aioquic_serve(
            self._bind_host, self._bind_port, configuration=cfg,
            create_protocol=self._make_server_protocol,
        )
        sockname = self._server._transport.get_extra_info("sockname")  # noqa: SLF001
        self._bound_port = sockname[1]

    async def stop(self) -> None:
        if self._server is not None:
            self._server.close()
            self._server = None

    def _make_server_protocol(self, quic, stream_handler=None) -> _MeshProtocol:
        return _MeshProtocol(quic, on_handshake=self._on_inbound_handshake)

    def _on_inbound_handshake(self, protocol: _MeshProtocol) -> None:
        host, port = protocol.remote_addr[0], protocol.remote_addr[1]
        conn = QuicPeerConnection(protocol, (host, port), path_type=_path_type_for(host))
        self._accept_queue.put_nowait(conn)

    async def accept(self) -> AsyncIterator[PeerConnection]:
        while True:
            yield await self._accept_queue.get()

    async def dial(self, addr: PeerAddress, *, timeout: float = 15.0) -> PeerConnection:
        try:
            return await asyncio.wait_for(self._dial_direct(addr), timeout=timeout)
        except Exception as direct_err:
            if self._rendezvous is None:
                raise
            return await self._dial_via_rendezvous(addr, direct_err, timeout=timeout)

    async def _dial_direct(self, addr: PeerAddress) -> QuicPeerConnection:
        cfg = QuicConfiguration(is_client=True, alpn_protocols=[ALPN])
        cfg.verify_mode = ssl.CERT_NONE  # see _generate_self_signed_cert TODO
        stack = contextlib.AsyncExitStack()
        protocol = await stack.enter_async_context(
            aioquic_connect(addr.host, addr.port, configuration=cfg,
                             create_protocol=lambda q, stream_handler=None: _MeshProtocol(q))
        )
        path_type = addr.path_type if addr.path_type != "direct" else _path_type_for(addr.host)
        return QuicPeerConnection(protocol, (addr.host, addr.port), path_type=path_type,
                                   close_stack=stack)

    async def _dial_via_rendezvous(self, addr: PeerAddress, direct_err: Exception, *,
                                    timeout: float) -> PeerConnection:
        """Best-effort NAT traversal via the control-plane tracker.

        TODO(phase-1.5): real simultaneous-open hole punching needs the *callee*
        coordinated at the same moment (e.g. the control plane pushing this
        ticket to them so both sides fire UDP packets at each other's reflexive
        addr inside the same NAT binding window). We only drive our own side
        here — this cannot be fully exercised without two real NATed machines,
        which is explicitly out of scope for this pass. Relay tunneling
        (GET /api/v1/rendezvous/relay/{session}) is likewise not implemented yet.
        """
        ticket = await self._rendezvous.connect(self._identity.node_id, addr.node_id)
        if ticket.peer and ticket.peer.reflexive_addr:
            host, port_s = ticket.peer.reflexive_addr.rsplit(":", 1)
            punch_addr = PeerAddress(node_id=addr.node_id, host=host, port=int(port_s),
                                      path_type="holepunched")
            try:
                return await asyncio.wait_for(self._dial_direct(punch_addr), timeout=timeout)
            except Exception:
                pass  # fall through to relay
        if ticket.relay_url:
            raise ConnectionError(
                f"relay fallback not implemented yet (relay_url={ticket.relay_url}); "
                f"see TODO(phase-1.5) in _dial_via_rendezvous"
            ) from direct_err
        raise ConnectionError("rendezvous gave no reflexive addr and no relay_url") from direct_err

    @property
    def local_quic_port(self) -> int:
        return self._bound_port


if __name__ == "__main__":
    import sys

    sys.path.insert(0, "packages/protocol")
    from meshcompute_protocol import Dtype, NodeIdentity, PacketClass

    async def demo() -> None:
        ident_a = NodeIdentity.generate()
        ident_b = NodeIdentity.generate()
        a = QuicTransport(ident_a, bind_port=0, bind_host="127.0.0.1")
        b = QuicTransport(ident_b, bind_port=0, bind_host="127.0.0.1")
        await a.start()
        await b.start()
        print(f"A listening on 127.0.0.1:{a.local_quic_port}")
        print(f"B listening on 127.0.0.1:{b.local_quic_port}")

        accept_task = asyncio.ensure_future(anext_conn(b))

        conn_a = await a.dial(PeerAddress(node_id=ident_b.node_id, host="127.0.0.1",
                                           port=b.local_quic_port))
        print(f"A->B dial path_type={conn_a.path_type}")
        conn_b = await accept_task
        print(f"B accepted path_type={conn_b.path_type}")

        stream_a = await conn_a.open_stream()
        sent = Frame(packet_class=PacketClass.ACTIVATION, request_id=42, session_id="s1",
                     step=1, tensor_id=7, dtype=Dtype.F32, shape=(2, 3),
                     payload=b"hello-mesh-activation")
        await stream_a.send_frame(sent)

        stream_b = await conn_b.accept_stream(timeout=5.0)
        received = await stream_b.recv_frame(timeout=5.0)
        assert received.payload == sent.payload, "payload roundtrip failed"
        assert received.shape == sent.shape, "shape roundtrip failed"
        assert received.packet_class == PacketClass.ACTIVATION
        print(f"frame roundtrip OK: payload={received.payload!r} shape={received.shape}")

        rtt = await conn_a.rtt_ms()
        print(f"measured rtt_ms={rtt:.3f}")
        assert rtt > 0

        await stream_a.close()
        await conn_a.close()
        await conn_b.close()
        await a.stop()
        await b.stop()
        print("transport_quic.py self-check PASSED")

    async def anext_conn(transport: QuicTransport) -> PeerConnection:
        async for conn in transport.accept():
            return conn
        raise RuntimeError("accept() exhausted")

    asyncio.run(demo())
