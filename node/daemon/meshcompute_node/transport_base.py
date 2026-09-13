"""Transport abstraction (PROTOCOL.md layers, INSTRUCTIONS §9).

Separates transport from inference semantics so a local pod can later swap in
NCCL / RDMA / shared memory while WAN peers use QUIC. Phase 1 ships the QUIC
implementation (internet-native, no VPN); this ABC is what the swarm and the
inference session code depend on.

Every stream MUST expose cancellation, timeout, bounded queues and backpressure
(PROTOCOL.md "Backpressure"; INSTRUCTIONS rule 12).
"""

from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import AsyncIterator

from meshcompute_protocol import Frame


@dataclass
class PeerAddress:
    node_id: str
    host: str
    port: int
    path_type: str = "direct"     # direct | holepunched | relay | loopback


class Stream(abc.ABC):
    """A multiplexed, ordered, reliable byte/frame stream over a peer connection."""

    @abc.abstractmethod
    async def send_frame(self, frame: Frame) -> None: ...

    @abc.abstractmethod
    async def recv_frame(self, timeout: float | None = None) -> Frame:
        """Raises TimeoutError on timeout, EOFError on clean close."""

    @abc.abstractmethod
    async def close(self) -> None: ...


class PeerConnection(abc.ABC):
    @abc.abstractmethod
    async def open_stream(self) -> Stream: ...

    @abc.abstractmethod
    async def rtt_ms(self) -> float:
        """Measured round-trip latency on this connection (feeds the topology graph)."""

    @abc.abstractmethod
    async def close(self) -> None: ...

    @property
    @abc.abstractmethod
    def path_type(self) -> str: ...


class Transport(abc.ABC):
    """Node-level transport: listen for inbound peers, dial outbound peers.

    Prefer OUTBOUND connections so contributors need no inbound firewall rules
    (API.md "Node API"). NAT traversal is coordinated by the rendezvous server.
    """

    @abc.abstractmethod
    async def start(self) -> None: ...

    @abc.abstractmethod
    async def stop(self) -> None: ...

    @abc.abstractmethod
    async def dial(self, addr: PeerAddress, *, timeout: float = 15.0) -> PeerConnection:
        """Establish a direct encrypted connection, hole-punching if needed."""

    @abc.abstractmethod
    def accept(self) -> AsyncIterator[PeerConnection]:
        """Yield inbound peer connections."""

    @property
    @abc.abstractmethod
    def local_quic_port(self) -> int: ...
