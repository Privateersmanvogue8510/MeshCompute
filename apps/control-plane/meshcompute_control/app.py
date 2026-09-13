"""Control-plane FastAPI app — implements every endpoint in
meshcompute_protocol.api_models (THE FROZEN CONTRACT) exactly.

Responsibilities per INSTRUCTIONS §2.1: auth, peer registration, rendezvous,
model catalog, scheduling metadata, credit accounting. This process is NOT the
activation data path — it never sees prompt bodies and holds no tool
credentials (INSTRUCTIONS §2.4, §18).

Run with: uvicorn meshcompute_control.app:app --port 8080
"""

from __future__ import annotations

import asyncio
import base64
import logging
import secrets
import time
from pathlib import Path

import yaml
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from pydantic_settings import BaseSettings, SettingsConfigDict

from meshcompute_protocol import (
    CapabilityRecord,
    ConnectRequest,
    ConnectTicket,
    ExecutionPlan,
    HeartbeatRequest,
    HeartbeatResponse,
    ModelManifest,
    ModelView,
    NodeIdentity,
    NodeView,
    RegisterRequest,
    RegisterResponse,
    RendezvousAnnounce,
    RendezvousPeers,
    ScheduleRequest,
    SignedCapability,
    SignedManifest,
    SignedReceipt,
    node_id_from_public,
    verify_json,
)

from .rendezvous import Rendezvous
from .scheduler import Scheduler
from .store import Store
from .topology import LinkMetrics, TopologyGraph

logger = logging.getLogger("meshcompute_control")

# A node considered offline if it hasn't registered/heartbeat within this window.
HEARTBEAT_TIMEOUT_S = 90.0
# Idle timeout for an unused relay side — never block forever (INSTRUCTIONS §26.12).
RELAY_IDLE_TIMEOUT_S = 300.0


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="MESH_CP_")
    bind: str = "127.0.0.1:8080"
    db: str = "var/control.db"
    manifest_dir: str = "models/manifests"


def _pubkey_from_b64(public_b64: str) -> Ed25519PublicKey:
    pad = "=" * (-len(public_b64) % 4)
    raw = base64.urlsafe_b64decode(public_b64 + pad)
    return Ed25519PublicKey.from_public_bytes(raw)


def _node_id_from_public_b64(public_b64: str) -> str:
    try:
        return node_id_from_public(_pubkey_from_b64(public_b64))
    except ValueError:
        return ""


def _load_measured_rtts() -> dict[str, float]:
    """Best-effort read of deploy/nodes.local.yaml (gitignored) to seed known
    node_id -> RTT-from-here (ms). Missing/malformed file is not fatal — the
    scheduler falls back to TopologyGraph's own pessimistic WAN default."""
    path = Path("deploy/nodes.local.yaml")
    if not path.exists():
        return {}
    try:
        data = yaml.safe_load(path.read_text()) or {}
    except Exception:
        logger.warning("could not parse %s; ignoring", path)
        return {}
    out: dict[str, float] = {}
    for n in data.get("nodes", []) or []:
        nid = n.get("node_id")
        rtt = n.get("measured_rtt_ms_from_here")
        if nid and rtt is not None:
            out[str(nid)] = float(rtt)
    return out


def create_app() -> FastAPI:
    settings = Settings()
    app = FastAPI(title="MeshCompute Control Plane")

    store = Store(settings.db)
    identity = NodeIdentity.load_or_create("deploy/identities/control-plane.key.json")
    rendezvous = Rendezvous(identity)
    scheduler = Scheduler()
    measured_rtts = _load_measured_rtts()
    # Latest heartbeat per node, kept in memory only. The signed capability is
    # the trust-anchored snapshot; heartbeats are frequent/unsigned liveness +
    # freshness pings layered on top (see heartbeat() and schedule() below).
    live_heartbeats: dict[str, HeartbeatRequest] = {}

    app.state.store = store
    app.state.identity = identity

    @app.on_event("startup")
    async def _load_manifests() -> None:
        manifest_dir = Path(settings.manifest_dir)
        if not manifest_dir.is_dir():
            logger.warning("manifest dir %s does not exist", manifest_dir)
            return
        for path in sorted(manifest_dir.glob("*.yaml")):
            try:
                raw = yaml.safe_load(path.read_text())
                manifest = ModelManifest.model_validate(raw)
                manifest_hash = manifest.manifest_hash()
                signature = identity.sign_json(manifest.model_dump(mode="json"))
                signed = SignedManifest(
                    manifest=manifest, manifest_hash=manifest_hash,
                    public_b64=identity.public_b64, signature_b64=signature,
                )
                store.put_manifest(manifest.id, manifest_hash, signed.model_dump(mode="json"))
                logger.info("loaded manifest %s (%s)", manifest.id, manifest_hash[:12])
            except Exception:
                logger.exception("failed to load manifest %s", path)

    # ---- nodes --------------------------------------------------------------
    @app.post("/api/v1/nodes/register", response_model=RegisterResponse)
    async def register_node(req: RegisterRequest) -> RegisterResponse:
        store.upsert_node(req.node_id, req.public_b64, req.pool_ids, time.time(), online=True)
        return RegisterResponse(
            ok=True, node_id=req.node_id, challenge_nonce=secrets.token_hex(16),
            heartbeat_interval_s=30, rendezvous_url="/api/v1/rendezvous",
        )

    @app.post("/api/v1/nodes/{node_id}/capabilities")
    async def post_capabilities(node_id: str, cap: SignedCapability) -> dict:
        node = store.get_node(node_id)
        if node is None:
            raise HTTPException(404, f"unknown node {node_id}")
        derived = _node_id_from_public_b64(cap.public_b64)
        if derived != node_id or cap.record.node_id != node_id:
            raise HTTPException(400, "public key/record does not derive the claimed node_id")
        if cap.public_b64 != node["public_b64"]:
            raise HTTPException(400, "public key does not match the key registered for this node")
        if not verify_json(cap.public_b64, cap.record.model_dump(mode="json"), cap.signature_b64):
            raise HTTPException(400, "bad capability signature")
        store.put_capability(node_id, cap.model_dump(mode="json"), time.time())
        return {"ok": True}

    @app.post("/api/v1/nodes/{node_id}/heartbeat", response_model=HeartbeatResponse)
    async def heartbeat(node_id: str, req: HeartbeatRequest) -> HeartbeatResponse:
        if store.get_node(node_id) is None:
            raise HTTPException(404, f"unknown node {node_id}")
        live_heartbeats[node_id] = req
        store.touch_node(node_id, time.time(), online=True)
        return HeartbeatResponse(ok=True, challenge_nonce=secrets.token_hex(16))

    @app.get("/api/v1/nodes", response_model=list[NodeView])
    async def list_nodes() -> list[NodeView]:
        now = time.time()
        views = []
        for n in store.list_nodes():
            cap_raw = store.get_capability(n["node_id"])
            cap = SignedCapability.model_validate(cap_raw) if cap_raw else None
            online = n["online"] and (now - n["last_seen"]) < HEARTBEAT_TIMEOUT_S
            views.append(NodeView(
                node_id=n["node_id"], online=online, pool_ids=n["pool_ids"],
                capability=cap, last_seen=n["last_seen"],
            ))
        return views

    # ---- model catalog --------------------------------------------------------
    @app.get("/api/v1/models", response_model=list[ModelView])
    async def list_models() -> list[ModelView]:
        out = []
        for m in store.list_manifests():
            manifest = m["signed_manifest"]["manifest"]
            out.append(ModelView(
                id=m["alias"], manifest_hash=m["manifest_hash"],
                display_name=manifest.get("display_name", ""),
                context_length=manifest.get("context_length_configured", 0),
                signed=True,
            ))
        return out

    @app.get("/api/v1/models/{alias}", response_model=SignedManifest)
    async def get_model(alias: str) -> SignedManifest:
        raw = store.get_manifest(alias)
        if raw is None:
            raise HTTPException(404, f"unknown model {alias}")
        return SignedManifest.model_validate(raw)

    # ---- scheduling -------------------------------------------------------------
    @app.post("/api/v1/schedule", response_model=ExecutionPlan)
    async def schedule(req: ScheduleRequest) -> ExecutionPlan:
        raw = store.get_manifest(req.model_id) or store.get_manifest_by_hash(req.model_id)
        if raw is None:
            raise HTTPException(404, f"unknown model {req.model_id}")
        manifest = ModelManifest.model_validate(raw["manifest"])

        now = time.time()
        graph = TopologyGraph()
        nodes: dict[str, CapabilityRecord] = {}
        for n in store.list_nodes():
            if not n["online"] or (now - n["last_seen"]) > HEARTBEAT_TIMEOUT_S:
                continue
            if req.pool_id not in n["pool_ids"]:
                continue
            cap_raw = store.get_capability(n["node_id"])
            if cap_raw is None:
                continue
            record = SignedCapability.model_validate(cap_raw).record

            hb = live_heartbeats.get(n["node_id"])
            if hb is not None:
                # ponytail: heartbeat's aggregate RAM figure is fresher than the
                # (less frequently re-signed) capability snapshot; override just
                # that field. Per-GPU VRAM redistribution from one aggregate
                # heartbeat number is unneeded precision for the POC.
                record = record.model_copy(update={"ram_free_bytes": hb.ram_free_bytes})
            nodes[n["node_id"]] = record

            if n["node_id"] in measured_rtts:
                graph.set_link(req.client_node_id, n["node_id"],
                                LinkMetrics(rtt_ms=measured_rtts[n["node_id"]], path_type="direct"))
            # else: leave unset. TopologyGraph.link() already defaults unknown
            # links pessimistically (WAN) and true loopback (client == node)
            # automatically — no need to re-encode that heuristic here.

        # current queue depth per node, straight from the latest heartbeat — this is
        # what lets the scheduler's replica routing prefer a less-busy node.
        load = {nid: live_heartbeats[nid].queue_depth for nid in nodes if nid in live_heartbeats}
        try:
            plan = scheduler.plan(
                model=manifest, nodes=nodes, graph=graph, client_node_id=req.client_node_id,
                context_length=req.context_length or manifest.context_length_configured,
                prompt_tokens=req.prompt_tokens, gen_tokens=req.gen_tokens, load=load,
                plan_id="plan_" + secrets.token_hex(8),
            )
        except ValueError as e:
            raise HTTPException(500, str(e))

        logger.info("plan %s selected %s over %s: %s",
                    plan.plan_id, plan.strategy.value, plan.peer_chain, plan.decision_trace[0])
        store.put_session(plan.plan_id, plan.model_dump(mode="json"), now)
        return plan

    # ---- work receipts / credit ledger -------------------------------------------
    @app.post("/api/v1/work-receipts")
    async def submit_receipt(sr: SignedReceipt) -> dict:
        node = store.get_node(sr.receipt.node_id)
        if node is None:
            raise HTTPException(404, f"unknown node {sr.receipt.node_id}")
        if sr.public_b64 != node["public_b64"]:
            raise HTTPException(400, "public key does not match the key registered for this node")
        if not verify_json(sr.public_b64, sr.receipt.model_dump(mode="json"), sr.signature_b64):
            raise HTTPException(400, "bad receipt signature")

        ok = store.put_receipt(
            sr.receipt.receipt_id, sr.receipt.node_id, sr.receipt.plan_id,
            sr.receipt.challenge_nonce, sr.model_dump(mode="json"),
        )
        if not ok:
            # receipt_id reused, or (node_id, challenge_nonce) already seen —
            # either way this is a replay, never trust it twice.
            raise HTTPException(409, "duplicate receipt_id or replayed challenge_nonce")

        # Simple, auditable work-unit credit (INSTRUCTIONS §11): GPU-active
        # milliseconds is the measured input; demand/reliability/scarcity
        # multipliers are a Phase-2 refinement over this same ledger row shape.
        credited = round(sr.receipt.measured_work.gpu_active_ms / 1000.0, 4)
        store.append_ledger(
            sr.receipt.node_id, credited, "work_receipt", sr.receipt.receipt_id, time.time(),
        )
        return {"ok": True, "credited": credited}

    # ---- rendezvous (internet-native NAT traversal, no VPN) ----------------------
    @app.post("/api/v1/rendezvous/announce", response_model=RendezvousPeers)
    async def rendezvous_announce(req: RendezvousAnnounce, request: Request) -> RendezvousPeers:
        source_addr = f"{request.client.host}:{request.client.port}" if request.client else ""
        try:
            return rendezvous.announce(req, source_addr, time.time())
        except ValueError as e:
            raise HTTPException(400, str(e))

    @app.post("/api/v1/rendezvous/connect", response_model=ConnectTicket)
    async def rendezvous_connect(req: ConnectRequest) -> ConnectTicket:
        node = store.get_node(req.from_node_id)
        if node is None:
            raise HTTPException(404, f"unknown node {req.from_node_id}")
        payload = req.model_dump(mode="json", exclude={"signature_b64"})
        if not verify_json(node["public_b64"], payload, req.signature_b64):
            raise HTTPException(400, "bad connect signature")
        try:
            return rendezvous.connect(req, time.time())
        except ValueError as e:
            raise HTTPException(404, str(e))

    @app.websocket("/api/v1/rendezvous/relay/{session}")
    async def rendezvous_relay(ws: WebSocket, session: str) -> None:
        """Opaque-byte relay between the two peers of one session — fallback
        path only, used when hole-punching can't establish a direct link
        (INSTRUCTIONS §2.1: the control plane is not the default data path)."""
        await ws.accept()
        pair = rendezvous.relay_session(session)
        side = pair.claim_side()
        if side is None:
            await ws.close(code=1013, reason="relay session full")
            return
        inbox, outbox = pair.queues_for(side)

        async def pump_in() -> None:
            try:
                while True:
                    data = await ws.receive_bytes()
                    await outbox.put(data)
            except WebSocketDisconnect:
                pass

        async def pump_out() -> None:
            while True:
                data = await asyncio.wait_for(inbox.get(), timeout=RELAY_IDLE_TIMEOUT_S)
                await ws.send_bytes(data)

        tasks = [asyncio.create_task(pump_in()), asyncio.create_task(pump_out())]
        try:
            await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        except (asyncio.TimeoutError, WebSocketDisconnect):
            pass
        finally:
            for t in tasks:
                t.cancel()
            rendezvous.drop_relay(session)

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    host, _, port = Settings().bind.partition(":")
    uvicorn.run(app, host=host or "127.0.0.1", port=int(port or 8080))
