"""In-memory rendezvous + relay (INSTRUCTIONS §2.1, §9 — public rendezvous +
NAT hole-punch + relay fallback, no VPN/tailnet).

The control plane brokers *introduction* only: announce() lets peers discover
others seeding the same content, connect() hands out a signed ticket telling two
peers how to reach each other. The relay (used only when hole-punching can't
work) carries opaque bytes for that one session — it is never the default
activation data path (INSTRUCTIONS §2.1).

Pure in-memory, rebuilt from announces — same philosophy as topology.py for
Phase 1 (no durable peer history yet). Timestamps are passed in by the caller
(app.py), never read from the clock in here.
"""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass, field

from meshcompute_protocol import (
    ConnectRequest,
    ConnectTicket,
    NodeIdentity,
    PeerCandidate,
    RendezvousAnnounce,
    RendezvousPeers,
    verify_json,
)

RELAY_TICKET_TTL_S = 30.0


@dataclass
class _Peer:
    candidate: PeerCandidate
    seeding: set[str]
    last_announce: float


@dataclass
class RelayPair:
    """Two-sided queue pair for one relay session. Side "a" writes go to `to_b`
    and side "a" reads from `to_a` (i.e. each queue is named for its reader)."""

    to_a: asyncio.Queue = field(default_factory=asyncio.Queue)
    to_b: asyncio.Queue = field(default_factory=asyncio.Queue)
    _claimed: set[str] = field(default_factory=set)

    def claim_side(self) -> str | None:
        """First caller becomes "a", second becomes "b". A third caller (or a
        reconnect) is refused — ponytail: no reconnect semantics yet, add if a
        relay needs to survive one side dropping and coming back."""
        for side in ("a", "b"):
            if side not in self._claimed:
                self._claimed.add(side)
                return side
        return None

    def queues_for(self, side: str) -> tuple[asyncio.Queue, asyncio.Queue]:
        """Returns (inbox, outbox): inbox is what this side reads (bytes sent by
        the other side), outbox is where this side writes (delivered to the
        other side)."""
        return (self.to_a, self.to_b) if side == "a" else (self.to_b, self.to_a)


class Rendezvous:
    def __init__(self, identity: NodeIdentity) -> None:
        self._identity = identity
        self._peers: dict[str, _Peer] = {}
        self._relays: dict[str, RelayPair] = {}

    # --- announce --------------------------------------------------------------
    def announce(self, req: RendezvousAnnounce, source_addr: str, now: float) -> RendezvousPeers:
        """Verify the announce signature, upsert the peer, return peers seeding
        overlapping content plus the server-observed source addr (STUN-like
        reflexive fallback for peers that couldn't self-discover one)."""
        payload = req.model_dump(mode="json", exclude={"signature_b64"})
        if not verify_json(req.public_b64, payload, req.signature_b64):
            raise ValueError("bad announce signature")

        reflexive = req.reflexive_addr or source_addr or None
        candidate = PeerCandidate(
            node_id=req.node_id,
            public_b64=req.public_b64,
            local_addrs=req.local_addrs,
            reflexive_addr=reflexive,
            quic_port=req.quic_port,
            nat_type=req.nat_type,
        )
        seeding = set(req.seeding_manifest_hashes)
        self._peers[req.node_id] = _Peer(candidate, seeding, now)

        others = [
            p.candidate
            for nid, p in self._peers.items()
            if nid != req.node_id and (p.seeding & seeding)
        ]
        return RendezvousPeers(ok=True, peers=others, your_reflexive_addr=reflexive)

    # --- connect -----------------------------------------------------------------
    def connect(self, req: ConnectRequest, now: float,
                ttl_s: float = RELAY_TICKET_TTL_S) -> ConnectTicket:
        """Issue a signed ticket for from_node_id to reach to_node_id. Caller
        (app.py) is responsible for verifying req.signature_b64 against
        from_node_id's registered key before calling this."""
        target = self._peers.get(req.to_node_id)
        if target is None:
            raise ValueError(f"unknown peer {req.to_node_id} (has it announced?)")
        source = self._peers.get(req.from_node_id)

        both_reflexive = bool(source and source.candidate.reflexive_addr) and bool(
            target.candidate.reflexive_addr
        )
        method = "holepunch" if both_reflexive else "relay"
        session_id = "sess_" + uuid.uuid4().hex[:24]

        ticket = ConnectTicket(
            ok=True,
            session_id=session_id,
            peer=target.candidate,
            method=method,
            relay_url=f"/api/v1/rendezvous/relay/{session_id}" if method == "relay" else None,
            expires_at=now + ttl_s,
        )
        signable = ticket.model_dump(mode="json", exclude={"control_signature_b64"})
        ticket.control_signature_b64 = self._identity.sign_json(signable)
        return ticket

    # --- relay plumbing, used by app.py's websocket endpoint --------------------
    def relay_session(self, session_id: str) -> RelayPair:
        """Get-or-create the queue pair for a session. Created lazily by
        whichever peer's websocket connects first — connect() only decides
        *whether* relay is needed, not when the pair is allocated."""
        return self._relays.setdefault(session_id, RelayPair())

    def drop_relay(self, session_id: str) -> None:
        self._relays.pop(session_id, None)


def _demo() -> None:
    """ponytail self-check: signature verification + peer matching + relay
    pairing, without a running server."""
    ann_key = NodeIdentity.generate()
    other_key = NodeIdentity.generate()
    cp_identity = NodeIdentity.generate()
    rv = Rendezvous(cp_identity)

    def _sign_announce(ident: NodeIdentity, node_id: str, seeding: list[str]) -> RendezvousAnnounce:
        a = RendezvousAnnounce(
            node_id=node_id, public_b64=ident.public_b64,
            local_addrs=["10.0.0.5:4000"], quic_port=4000,
            seeding_manifest_hashes=seeding,
        )
        payload = a.model_dump(mode="json", exclude={"signature_b64"})
        a.signature_b64 = ident.sign_json(payload)
        return a

    a1 = _sign_announce(ann_key, "nd_a", ["hash1"])
    peers1 = rv.announce(a1, "1.2.3.4:9999", now=1.0)
    assert peers1.ok and peers1.peers == [] and peers1.your_reflexive_addr == "1.2.3.4:9999"

    a2 = _sign_announce(other_key, "nd_b", ["hash1"])
    peers2 = rv.announce(a2, "5.6.7.8:8888", now=2.0)
    assert len(peers2.peers) == 1 and peers2.peers[0].node_id == "nd_a"

    # tampered payload must fail verification
    bad = _sign_announce(ann_key, "nd_c", ["hash1"])
    bad.node_id = "nd_evil"
    try:
        rv.announce(bad, "0.0.0.0:1", now=3.0)
        raise AssertionError("expected bad signature to raise")
    except ValueError:
        pass

    req = ConnectRequest(from_node_id="nd_a", to_node_id="nd_b")
    ticket = rv.connect(req, now=10.0)
    assert ticket.ok and ticket.method == "holepunch"  # both announced reflexive addrs
    assert verify_json(
        cp_identity.public_b64,
        ticket.model_dump(mode="json", exclude={"control_signature_b64"}),
        ticket.control_signature_b64,
    )

    pair = rv.relay_session("sess_x")
    assert pair.claim_side() == "a"
    assert pair.claim_side() == "b"
    assert pair.claim_side() is None
    rv.drop_relay("sess_x")
    print("rendezvous.py self-check OK")


if __name__ == "__main__":
    _demo()
