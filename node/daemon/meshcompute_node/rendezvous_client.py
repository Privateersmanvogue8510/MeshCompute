"""Rendezvous HTTP client (PROTOCOL.md "Rendezvous", INSTRUCTIONS §9).

Talks to the control-plane's public tracker endpoints (api_models.py is the
frozen contract) so a node can announce itself — BitTorrent-tracker style —
and ask for an introduction ticket to another peer. This module only speaks
HTTP; it does not touch sockets or QUIC (see transport_quic.py for that).
"""

from __future__ import annotations

import httpx

from meshcompute_protocol import (
    ConnectRequest,
    ConnectTicket,
    NodeIdentity,
    RendezvousAnnounce,
    RendezvousPeers,
)


class RendezvousClient:
    def __init__(self, base_url: str, *, timeout: float = 10.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    async def announce(
        self,
        identity: NodeIdentity,
        local_addrs: list[str],
        reflexive_addr: str | None,
        quic_port: int,
        nat_type: str = "unknown",
        seeding_hashes: list[str] | None = None,
        wanted_hashes: list[str] | None = None,
    ) -> RendezvousPeers:
        """Sign and POST a RendezvousAnnounce; return the peers seeding content
        we seed or want (BitTorrent-tracker style)."""
        announce = RendezvousAnnounce(
            node_id=identity.node_id,
            public_b64=identity.public_b64,
            local_addrs=local_addrs,
            reflexive_addr=reflexive_addr,
            quic_port=quic_port,
            nat_type=nat_type,
            seeding_manifest_hashes=seeding_hashes or [],
            wanted_manifest_hashes=wanted_hashes or [],
        )
        body = announce.model_dump(mode="json")
        body.pop("signature_b64", None)
        body["signature_b64"] = identity.sign_json(body)
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(f"{self.base_url}/api/v1/rendezvous/announce", json=body)
            resp.raise_for_status()
        return RendezvousPeers.model_validate(resp.json())

    async def connect(self, identity: NodeIdentity, to_id: str, plan_id: str = "") -> ConnectTicket:
        """Ask the tracker to introduce us to `to_id`; returns a short-lived ticket
        describing how to reach them (holepunch coordinates or a relay URL).
        Signed with our identity — the control plane rejects unsigned requests."""
        req = ConnectRequest(from_node_id=identity.node_id, to_node_id=to_id, plan_id=plan_id)
        body = req.model_dump(mode="json", exclude={"signature_b64"})
        body["signature_b64"] = identity.sign_json(body)
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(f"{self.base_url}/api/v1/rendezvous/connect", json=body)
            resp.raise_for_status()
        return ConnectTicket.model_validate(resp.json())


if __name__ == "__main__":
    # Self-check without a real control plane: a tiny local HTTP stub that just
    # echoes back what an announce/connect exchange should look like, so we
    # confirm the signing + request/response shapes are wired correctly.
    import asyncio
    import threading
    from http.server import BaseHTTPRequestHandler, HTTPServer

    from meshcompute_protocol import verify_json

    captured: dict = {}

    class _Stub(BaseHTTPRequestHandler):
        def log_message(self, *a):  # silence
            pass

        def do_POST(self):
            length = int(self.headers["Content-Length"])
            body = self.rfile.read(length)
            import json
            payload = json.loads(body)
            if self.path.endswith("/announce"):
                captured["announce"] = payload
                resp = {"ok": True, "peers": [], "your_reflexive_addr": "198.51.100.7:4000"}
            else:
                captured["connect"] = payload
                resp = {"ok": True, "session_id": "s1", "peer": None, "method": "direct",
                        "relay_url": None, "expires_at": 0.0, "control_signature_b64": ""}
            data = json.dumps(resp).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    async def demo() -> None:
        server = HTTPServer(("127.0.0.1", 0), _Stub)
        port = server.server_address[1]
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            ident = NodeIdentity.generate()
            client = RendezvousClient(f"http://127.0.0.1:{port}")
            peers = await client.announce(ident, ["10.0.0.5:5000"], "198.51.100.7:4000", 5000)
            assert peers.ok and peers.your_reflexive_addr == "198.51.100.7:4000"
            sig = captured["announce"].pop("signature_b64")
            assert verify_json(ident.public_b64, captured["announce"], sig), "signature must verify"
            print("announce signed + verified OK")

            ticket = await client.connect(ident, "nd_other", plan_id="p1")
            assert ticket.ok and ticket.session_id == "s1"
            assert captured["connect"]["from_node_id"] == ident.node_id
            sig = captured["connect"].pop("signature_b64")
            assert verify_json(ident.public_b64, captured["connect"], sig), "connect must be signed"
            print("connect round-trip OK")
            print("rendezvous_client.py self-check PASSED")
        finally:
            server.shutdown()
            thread.join(timeout=2)

    asyncio.run(demo())
