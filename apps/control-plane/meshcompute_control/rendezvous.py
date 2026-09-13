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
    wanted: set[str] = field(default_factory=set)
    wanted_since: dict[str, float] = field(default_factory=dict)   # hash -> first announce


@dataclass
class RelayPair:
    """Two-sided queue pair for one relay session. Side "a" writes go to `to_b`
    and side "a" reads from `to_a` (i.e. each queue is named for its reader)."""

    # bounded: a peer that never drains cannot make the control plane buffer
    # unboundedly (SECURITY.md "Denial of service": bounded queues)
    to_a: asyncio.Queue = field(default_factory=lambda: asyncio.Queue(maxsize=64))
    to_b: asyncio.Queue = field(default_factory=lambda: asyncio.Queue(maxsize=64))
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
        self._issued: dict[str, float] = {}   # session_id -> expires_at (from connect())

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
        wanted = set(req.wanted_manifest_hashes)
        prev = self._peers.get(req.node_id)
        since = {h: (prev.wanted_since.get(h, now) if prev else now) for h in wanted}
        self._peers[req.node_id] = _Peer(candidate, seeding, now, wanted, since)

        # Roles are relative to what the announcer WANTS: a peer is a "seeder"
        # only if it holds a wanted hash, a "leecher" if it is fetching one right
        # now (so it will seed soon — wait for it rather than stampede origin).
        # With nothing wanted, any content overlap is reported (plain discovery).
        others: list[PeerCandidate] = []
        for nid, p in self._peers.items():
            if nid == req.node_id:
                continue
            if wanted and (p.seeding & wanted):
                others.append(p.candidate.model_copy(update={"role": "seeder"}))
            elif wanted and (p.wanted & wanted):
                others.append(p.candidate.model_copy(update={"role": "leecher"}))
            elif not wanted and (p.seeding & seeding):
                others.append(p.candidate)
        # Origin leader for the wanted content: among everyone wanting it (the
        # announcer included) and nobody seeding it, the earliest starter wins.
        # Deterministic on the tracker's clock, so a wave of nodes agrees on ONE
        # origin downloader without any node-to-node coordination.
        leader = None
        if wanted and not any(p.role == "seeder" for p in others):
            h = sorted(wanted)[0]
            wanting = [(p.wanted_since[h], nid) for nid, p in self._peers.items()
                       if h in p.wanted and h not in p.seeding]
            leader = min(wanting)[1] if wanting else None
        return RendezvousPeers(ok=True, peers=others, your_reflexive_addr=reflexive,
                               origin_leader=leader)

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
        self._issued[session_id] = now + ttl_s
        for sid in [sid for sid, exp in self._issued.items() if exp < now - 3600]:
            self._issued.pop(sid, None)
        return ticket

    def session_known(self, session_id: str) -> bool:
        """Only sessions handed out by connect() may open a relay."""
        return session_id in self._issued

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

    def _sign_announce(ident: NodeIdentity, node_id: str, seeding: list[str],
                       wanted: list[str] | None = None) -> RendezvousAnnounce:
        a = RendezvousAnnounce(
            node_id=node_id, public_b64=ident.public_b64,
            local_addrs=["10.0.0.5:4000"], quic_port=4000,
            seeding_manifest_hashes=seeding, wanted_manifest_hashes=wanted or [],
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

    # two leechers of hash2 that both still SEED hash1 (an update wave): each must
    # see the other as a "leecher" of hash2, never as a seeder via the old hash
    k1, k2 = NodeIdentity.generate(), NodeIdentity.generate()
    l1 = rv.announce(_sign_announce(k1, "nd_l1", ["hash1"], ["hash2"]), "1.1.1.1:1", now=4.0)
    assert l1.peers == [] and l1.origin_leader == "nd_l1"        # first to want it -> origin
    l2 = rv.announce(_sign_announce(k2, "nd_l2", ["hash1"], ["hash2"]), "2.2.2.2:2", now=5.0)
    assert [(p.node_id, p.role) for p in l2.peers] == [("nd_l1", "leecher")]
    assert l2.origin_leader == "nd_l1"                            # nd_l2 waits
    # nd_l1 re-announcing while it downloads keeps its original "since"
    again = rv.announce(_sign_announce(k1, "nd_l1", ["hash1"], ["hash2"]), "1.1.1.1:1", now=5.5)
    assert again.origin_leader == "nd_l1"
    # once nd_l1 has it, nd_l2 sees a seeder and no leader is needed
    rv.announce(_sign_announce(k1, "nd_l1", ["hash1", "hash2"]), "1.1.1.1:1", now=6.0)
    l2b = rv.announce(_sign_announce(k2, "nd_l2", ["hash1"], ["hash2"]), "2.2.2.2:2", now=7.0)
    assert [(p.node_id, p.role) for p in l2b.peers] == [("nd_l1", "seeder")] and l2b.origin_leader is None

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

    assert not rv.session_known("sess_x") and rv.session_known(ticket.session_id)
    pair = rv.relay_session("sess_x")
    assert pair.claim_side() == "a"
    assert pair.claim_side() == "b"
    assert pair.claim_side() is None
    rv.drop_relay("sess_x")
    print("rendezvous.py self-check OK")


if __name__ == "__main__":
    _demo()
