"""An update wave must not stampede the origin: the tracker reports who is
already fetching, and a node waits for a lower-id leecher instead of racing.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from meshcompute_control.rendezvous import Rendezvous
from meshcompute_protocol import NodeIdentity, PeerCandidate, RendezvousAnnounce, RendezvousPeers


def _announce(rv, ident, seeding, wanted, now):
    a = RendezvousAnnounce(node_id=ident.node_id, public_b64=ident.public_b64, quic_port=1,
                           seeding_manifest_hashes=seeding, wanted_manifest_hashes=wanted)
    a.signature_b64 = ident.sign_json(a.model_dump(mode="json", exclude={"signature_b64"}))
    return rv.announce(a, "1.2.3.4:5", now)


def test_tracker_roles_and_origin_leader_are_relative_to_wanted_hash():
    rv = Rendezvous(NodeIdentity.generate())
    a, b = NodeIdentity.generate(), NodeIdentity.generate()
    # both seed OLD, both now want NEW (the update wave); b happens to poll first
    first = _announce(rv, b, ["OLD"], ["NEW"], 1.0)
    assert first.peers == [] and first.origin_leader == b.node_id      # b goes to origin
    second = _announce(rv, a, ["OLD"], ["NEW"], 2.0)
    assert [(p.node_id, p.role) for p in second.peers] == [(b.node_id, "leecher")]   # not seeder via OLD
    assert second.origin_leader == b.node_id                           # a waits, whatever its id
    _announce(rv, b, ["OLD", "NEW"], [], 3.0)                          # b finished + re-announced
    done = _announce(rv, a, ["OLD"], ["NEW"], 4.0)
    assert [(p.node_id, p.role) for p in done.peers] == [(b.node_id, "seeder")]
    assert done.origin_leader is None


async def test_fetch_waits_for_origin_leader_then_fetches_from_it(tmp_path, monkeypatch):
    from meshcompute_node import daemon

    me = NodeIdentity.generate()
    leader_id = "nd_ffffffffffffffffffffffffffffffff"   # a HIGHER id than ours can still lead
    calls = {"announce": 0, "seed": 0}
    state = {"leader_done": False}

    class _Transport:
        local_quic_port = 1

    swarm = daemon.PeerSwarm(me, "http://cp.test", _Transport())
    swarm.WAIT_POLL_S = 0.01

    async def fake_announce(wanted=None):
        calls["announce"] += 1
        if calls["announce"] >= 3:
            state["leader_done"] = True
        role = "seeder" if state["leader_done"] else "leecher"
        return RendezvousPeers(ok=True, peers=[PeerCandidate(node_id=leader_id, public_b64="x",
                                                             local_addrs=["127.0.0.1:1"], role=role)],
                               origin_leader=None if state["leader_done"] else leader_id)

    async def fake_try_seeders(seeders, dest, manifest_hash, shard):
        calls["seed"] += 1
        assert all(p.role == "seeder" for p in seeders)
        Path(dest).write_bytes(b"model")
        return True

    monkeypatch.setattr(swarm, "announce", fake_announce)
    monkeypatch.setattr(swarm, "_try_seeders", fake_try_seeders)
    ok = await swarm.fetch(tmp_path / "m.gguf", "NEW", {"rfilename": "m.gguf", "size_bytes": 5})
    assert ok and calls["seed"] == 1 and calls["announce"] >= 3   # waited, then fetched P2P


async def test_fetch_goes_to_origin_when_it_is_the_leader(tmp_path, monkeypatch):
    from meshcompute_node import daemon

    me = NodeIdentity.generate()
    other = "nd_0000000000000000000000000000000"   # a LOWER id that started later does not lead

    class _Transport:
        local_quic_port = 1

    swarm = daemon.PeerSwarm(me, "http://cp.test", _Transport())

    async def fake_announce(wanted=None):
        return RendezvousPeers(ok=True, peers=[PeerCandidate(node_id=other, public_b64="x",
                                                             role="leecher")],
                               origin_leader=me.node_id)

    monkeypatch.setattr(swarm, "announce", fake_announce)
    t0 = asyncio.get_running_loop().time()
    ok = await swarm.fetch(tmp_path / "m.gguf", "NEW", {"rfilename": "m.gguf", "size_bytes": 5})
    assert ok is False                                        # origin, immediately
    assert asyncio.get_running_loop().time() - t0 < 1.0
