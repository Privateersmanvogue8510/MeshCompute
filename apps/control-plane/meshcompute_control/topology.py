"""Topology graph (INSTRUCTIONS §8, ARCHITECTURE.md).

Vertices = workers (or local multi-GPU pods). Edges = measured RTT, throughput,
packet loss, NAT path type, relay requirement, historical stability. The scheduler
reads this graph; the daemon feeds it with measured link metrics.

This is deliberately in-memory for Phase 1 (rebuilt from the node registry +
heartbeat samples). Durable edge history is a Phase-1.5 refinement.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class LinkMetrics:
    rtt_ms: float = 1.0            # round-trip latency, measured over the QUIC link
    throughput_mbps: float = 1000  # measured usable bandwidth
    packet_loss: float = 0.0       # 0..1
    # how the direct internet path was established (no VPN involved):
    #   direct       - both reachable, no NAT trickery needed
    #   holepunched  - UDP hole-punch via rendezvous succeeded
    #   relay        - fell back to a relay (adds a hop, higher rtt)
    #   loopback/pcie- same host / same pod
    path_type: str = "direct"
    stability: float = 1.0         # 0..1, historical (1 = never dropped)

    def is_lan_class(self) -> bool:
        # Sub-5ms, high bandwidth, direct: safe for latency-sensitive splits.
        return self.rtt_ms <= 5.0 and self.throughput_mbps >= 500 and self.path_type in (
            "direct", "loopback", "pcie")


@dataclass
class Vertex:
    node_id: str
    # link from *this* vertex to another node_id
    links: dict[str, LinkMetrics] = field(default_factory=dict)


class TopologyGraph:
    def __init__(self) -> None:
        self._vertices: dict[str, Vertex] = {}

    def upsert_node(self, node_id: str) -> Vertex:
        return self._vertices.setdefault(node_id, Vertex(node_id))

    def set_link(self, a: str, b: str, metrics: LinkMetrics) -> None:
        self.upsert_node(a).links[b] = metrics
        # links are measured per-direction; default the reverse to the same until
        # the reverse sample arrives (asymmetry is common on consumer WAN uplinks).
        self.upsert_node(b).links.setdefault(a, metrics)

    def link(self, a: str, b: str) -> LinkMetrics:
        if a == b:
            return LinkMetrics(rtt_ms=0.05, throughput_mbps=100000, path_type="loopback")
        v = self._vertices.get(a)
        if v and b in v.links:
            return v.links[b]
        # Unknown link: assume a pessimistic WAN edge so the scheduler does not
        # optimistically split across an unmeasured path.
        return LinkMetrics(rtt_ms=120.0, throughput_mbps=100, path_type="relay", stability=0.7)

    def nodes(self) -> list[str]:
        return list(self._vertices)
