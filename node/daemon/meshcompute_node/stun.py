"""Minimal STUN client (RFC 5389) — hand-rolled Binding Request/Response.

No `stun` library dependency, by design: we only need one thing (a reflexive
"ip:port"), so this implements just the Binding Request/Response pair and the
XOR-MAPPED-ADDRESS attribute (falling back to the older MAPPED-ADDRESS for
servers that don't send it). See PROTOCOL.md "NAT traversal" / INSTRUCTIONS §9.
"""

from __future__ import annotations

import asyncio
import os
import socket
import struct

MAGIC_COOKIE = 0x2112A442
_BINDING_REQUEST = 0x0001
_BINDING_SUCCESS = 0x0101
_MAPPED_ADDRESS = 0x0001
_XOR_MAPPED_ADDRESS = 0x0020
_FAMILY_IPV4 = 0x01

_HEADER = struct.Struct(">HHI12s")  # type, length, magic cookie, transaction id

DEFAULT_STUN_SERVERS = [
    "stun.l.google.com:19302",
    "stun1.l.google.com:19302",
]


def _build_binding_request(txn_id: bytes) -> bytes:
    return _HEADER.pack(_BINDING_REQUEST, 0, MAGIC_COOKIE, txn_id)


def _parse_binding_response(data: bytes, txn_id: bytes) -> str | None:
    """Return "ip:port" from a Binding Success Response, or None if unparseable
    / not a match for our transaction."""
    if len(data) < _HEADER.size:
        return None
    msg_type, msg_len, cookie, resp_txn = _HEADER.unpack_from(data, 0)
    if msg_type != _BINDING_SUCCESS or cookie != MAGIC_COOKIE or resp_txn != txn_id:
        return None
    body = data[_HEADER.size:_HEADER.size + msg_len]
    mapped: str | None = None
    xor_mapped: str | None = None
    off = 0
    while off + 4 <= len(body):
        atype, alen = struct.unpack_from(">HH", body, off)
        aval = body[off + 4:off + 4 + alen]
        if len(aval) < 8 or aval[1] != _FAMILY_IPV4:
            off += 4 + alen + (-alen % 4)  # attributes are padded to a 4-byte boundary
            continue
        port_raw, addr_raw = aval[2:4], aval[4:8]
        if atype == _XOR_MAPPED_ADDRESS:
            port = struct.unpack(">H", port_raw)[0] ^ (MAGIC_COOKIE >> 16)
            addr = bytes(b ^ c for b, c in zip(addr_raw, data[4:8]))  # XOR w/ cookie bytes
            xor_mapped = f"{socket.inet_ntoa(addr)}:{port}"
        elif atype == _MAPPED_ADDRESS:
            port = struct.unpack(">H", port_raw)[0]
            mapped = f"{socket.inet_ntoa(addr_raw)}:{port}"
        off += 4 + alen + (-alen % 4)
    return xor_mapped or mapped


class _StunClientProtocol(asyncio.DatagramProtocol):
    def __init__(self) -> None:
        self.response: asyncio.Future[bytes] = asyncio.get_event_loop().create_future()

    def datagram_received(self, data: bytes, addr) -> None:
        if not self.response.done():
            self.response.set_result(data)

    def error_received(self, exc: Exception) -> None:
        if not self.response.done():
            self.response.set_exception(exc)


async def _query_one(local_port: int, host: str, port: int, timeout: float) -> str | None:
    loop = asyncio.get_running_loop()
    txn_id = os.urandom(12)
    transport, protocol = await loop.create_datagram_endpoint(
        _StunClientProtocol, local_addr=("0.0.0.0", local_port), remote_addr=(host, port))
    try:
        transport.sendto(_build_binding_request(txn_id))
        data = await asyncio.wait_for(protocol.response, timeout)
        return _parse_binding_response(data, txn_id)
    except (TimeoutError, OSError):
        return None
    finally:
        transport.close()


async def discover_reflexive(
    local_port: int, servers: list[str] | None = None, timeout: float = 3.0
) -> str | None:
    """Ask STUN servers in turn for our reflexive "ip:port"; None if all fail.

    Graceful failure by design (offline sandbox, blocked UDP, server down): a
    caller with no network path to any STUN server just gets None back, never
    an exception.
    """
    for server in servers or DEFAULT_STUN_SERVERS:
        host, _, port_s = server.partition(":")
        try:
            result = await _query_one(local_port, host, int(port_s or 3478), timeout)
        except Exception:
            result = None
        if result:
            return result
    return None


if __name__ == "__main__":
    # Self-contained check: no real network needed. Spin up a fake STUN server on
    # loopback that replies with a known XOR-MAPPED-ADDRESS, and confirm
    # discover_reflexive() decodes it correctly.
    async def _fake_stun_server(sock: socket.socket) -> None:
        loop = asyncio.get_running_loop()
        data, addr = await loop.sock_recvfrom(sock, 2048)
        msg_type, _, cookie, txn_id = _HEADER.unpack_from(data, 0)
        assert msg_type == _BINDING_REQUEST and cookie == MAGIC_COOKIE
        want_ip, want_port = "203.0.113.42", 51820
        xport = want_port ^ (MAGIC_COOKIE >> 16)
        cookie_bytes = struct.pack(">I", MAGIC_COOKIE)
        xaddr = bytes(b ^ c for b, c in zip(socket.inet_aton(want_ip), cookie_bytes))
        attr_val = bytes([0, _FAMILY_IPV4]) + struct.pack(">H", xport) + xaddr
        attr = struct.pack(">HH", _XOR_MAPPED_ADDRESS, len(attr_val)) + attr_val
        resp = _HEADER.pack(_BINDING_SUCCESS, len(attr), MAGIC_COOKIE, txn_id) + attr
        await loop.sock_sendto(sock, resp, addr)

    async def demo() -> None:
        srv_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        srv_sock.bind(("127.0.0.1", 0))
        srv_sock.setblocking(False)
        srv_port = srv_sock.getsockname()[1]
        server_task = asyncio.ensure_future(_fake_stun_server(srv_sock))
        try:
            result = await discover_reflexive(0, [f"127.0.0.1:{srv_port}"], timeout=2.0)
            print("reflexive:", result)
            assert result == "203.0.113.42:51820", result
            await server_task
        finally:
            srv_sock.close()

        # graceful-failure path: nothing listening on this port.
        none_result = await discover_reflexive(0, ["127.0.0.1:1"], timeout=0.3)
        assert none_result is None
        print("graceful failure OK (no server -> None)")
        print("stun.py self-check PASSED")

    asyncio.run(demo())
