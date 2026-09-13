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
import contextlib
import logging
import secrets
import time
from contextlib import asynccontextmanager
from pathlib import Path

import yaml
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from pydantic_settings import BaseSettings, SettingsConfigDict

from meshcompute_protocol import (
    CapabilityRecord,
    ConnectRequest,
    credit_for_receipt,
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
if not logging.getLogger().handlers:      # uvicorn doesn't configure app loggers
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger.setLevel(logging.INFO)

# A node considered offline if it hasn't registered/heartbeat within this window.
HEARTBEAT_TIMEOUT_S = 90.0
# Idle timeout for an unused relay side — never block forever (INSTRUCTIONS §26.12).
RELAY_IDLE_TIMEOUT_S = 300.0
# How often the catalog dir is re-scanned for pushed channel updates.
MANIFEST_RESCAN_S = 30.0


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

    seen_manifest_files: dict[str, float] = {}   # path -> mtime last loaded

    def scan_manifests() -> list[str]:
        """Load new/changed manifest YAMLs from the catalog dir, sign them, and
        publish them as the channel's current version. A file whose `version`
        is LOWER than what is already published for that alias is refused (a
        rollback must be an explicit, higher-versioned push — never an accident).
        Returns the aliases that changed. Runs at startup and every
        MANIFEST_RESCAN_S — this is how an operator pushes a model update."""
        manifest_dir = Path(settings.manifest_dir)
        if not manifest_dir.is_dir():
            logger.warning("manifest dir %s does not exist", manifest_dir)
            return []
        changed: list[str] = []
        for path in sorted(manifest_dir.glob("*.yaml")):
            try:
                mtime = path.stat().st_mtime
            except OSError:
                continue
            if seen_manifest_files.get(str(path)) == mtime:
                continue
            try:
                manifest = ModelManifest.model_validate(yaml.safe_load(path.read_text()))
                manifest_hash = manifest.manifest_hash()
                current = store.get_manifest(manifest.id)
                if current is not None:
                    cur_version = int(current["manifest"].get("version", 1))
                    if current["manifest_hash"] == manifest_hash:
                        seen_manifest_files[str(path)] = mtime
                        continue
                    if manifest.version < cur_version:
                        logger.warning("refusing %s: version %d < published v%d (rollbacks must "
                                       "bump the version)", path.name, manifest.version, cur_version)
                        seen_manifest_files[str(path)] = mtime
                        continue
                signed = SignedManifest(
                    manifest=manifest, manifest_hash=manifest_hash,
                    public_b64=identity.public_b64,
                    signature_b64=identity.sign_json(manifest.model_dump(mode="json")))
                store.put_manifest(manifest.id, manifest_hash, signed.model_dump(mode="json"))
                seen_manifest_files[str(path)] = mtime
                changed.append(manifest.id)
                logger.info("channel %s -> v%d (%s)", manifest.id, manifest.version, manifest_hash[:12])
            except Exception:
                logger.exception("failed to load manifest %s", path)
        return changed

    app.state.scan_manifests = scan_manifests

    async def _rescan_loop() -> None:
        while True:
            await asyncio.sleep(MANIFEST_RESCAN_S)
            with contextlib.suppress(Exception):
                scan_manifests()

    @asynccontextmanager
    async def _lifespan(_app: FastAPI):
        scan_manifests()
        task = asyncio.create_task(_rescan_loop())
        try:
            yield
        finally:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

    app.router.lifespan_context = _lifespan

    # ---- nodes --------------------------------------------------------------
    # Nonces the control plane has ISSUED to each node (register/heartbeat) and
    # not yet seen in a receipt. A receipt is only credited when its
    # challenge_nonce is one of these, so replay AND fabrication are rejected.
    issued_nonces: dict[str, set[str]] = {}
    MAX_OUTSTANDING_NONCES = 64

    def _issue_nonce(node_id: str) -> str:
        nonce = secrets.token_hex(16)
        bucket = issued_nonces.setdefault(node_id, set())
        if len(bucket) >= MAX_OUTSTANDING_NONCES:
            bucket.pop()
        bucket.add(nonce)
        return nonce

    def _signed_body(model, exclude: str = "signature_b64") -> dict:
        return model.model_dump(mode="json", exclude={exclude})

    @app.post("/api/v1/nodes/register", response_model=RegisterResponse)
    async def register_node(req: RegisterRequest) -> RegisterResponse:
        # node_id is DERIVED from the key, and the body is signed by that key —
        # so no one can bind a foreign key to another node's id, and nobody can
        # register a key they do not hold.
        if _node_id_from_public_b64(req.public_b64) != req.node_id:
            raise HTTPException(400, "public key does not derive the claimed node_id")
        if not req.signature_b64 or not verify_json(req.public_b64, _signed_body(req), req.signature_b64):
            raise HTTPException(400, "bad or missing registration signature")
        store.upsert_node(req.node_id, req.public_b64, req.pool_ids, time.time(), online=True)
        return RegisterResponse(
            ok=True, node_id=req.node_id, challenge_nonce=_issue_nonce(req.node_id),
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
        node = store.get_node(node_id)
        if node is None:
            raise HTTPException(404, f"unknown node {node_id}")
        if req.node_id != node_id or not req.signature_b64 or \
                not verify_json(node["public_b64"], _signed_body(req), req.signature_b64):
            raise HTTPException(400, "bad or missing heartbeat signature")
        live_heartbeats[node_id] = req
        store.touch_node(node_id, time.time(), online=True)
        return HeartbeatResponse(ok=True, challenge_nonce=_issue_nonce(node_id))

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
            quants = manifest.get("quantizations") or []
            size = sum(f.get("size_bytes", 0) for f in quants[0].get("files", [])) if quants else 0
            out.append(ModelView(
                id=m["alias"], manifest_hash=m["manifest_hash"],
                display_name=manifest.get("display_name", ""),
                context_length=manifest.get("context_length_configured", 0),
                signed=True, version=int(manifest.get("version", 1)), size_bytes=size,
            ))
        return out

    @app.get("/api/v1/models/{alias:path}", response_model=SignedManifest)
    async def get_model(alias: str) -> SignedManifest:
        """Aliases contain '/' (public/qwen3.8-27b-fable) — `:path` captures them."""
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
        # node<->node edges come from what nodes actually MEASURED over QUIC
        # (heartbeat.peer_rtt_ms); an unmeasured pair stays pessimistic, so the
        # scheduler never forms a "fast pod" out of two peers that never met.
        for nid in nodes:
            hb = live_heartbeats.get(nid)
            if hb is None:
                continue
            for peer_id, rtt in hb.peer_rtt_ms.items():
                if peer_id in nodes and rtt >= 0:
                    graph.set_link(nid, peer_id, LinkMetrics(rtt_ms=float(rtt), path_type="direct"))

        # current queue depth per node, straight from the latest heartbeat — this is
        # what lets the scheduler's replica routing prefer a less-busy node.
        load = {nid: live_heartbeats[nid].queue_depth for nid in nodes if nid in live_heartbeats}
        try:
            plan = scheduler.plan(
                model=manifest, nodes=nodes, graph=graph, client_node_id=req.client_node_id,
                context_length=req.context_length or manifest.context_length_configured,
                prompt_tokens=req.prompt_tokens, gen_tokens=req.gen_tokens, load=load,
                plan_id="plan_" + secrets.token_hex(8),
                executable_strategies=req.executable_strategies or None,
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
        r = sr.receipt
        node = store.get_node(r.node_id)
        if node is None:
            raise HTTPException(404, f"unknown node {r.node_id}")
        if sr.public_b64 != node["public_b64"]:
            raise HTTPException(400, "public key does not match the key registered for this node")
        if not verify_json(sr.public_b64, r.model_dump(mode="json"), sr.signature_b64):
            raise HTTPException(400, "bad receipt signature")
        # Credit only work the scheduler actually assigned (INSTRUCTIONS §11/§12):
        plan = store.get_session(r.plan_id)
        if plan is None:
            raise HTTPException(400, f"unknown plan_id {r.plan_id}")
        if r.node_id not in plan.get("peer_chain", []):
            raise HTTPException(400, "node was not part of that plan")
        if r.model_manifest_hash != plan.get("model_manifest_hash"):
            raise HTTPException(400, "receipt model does not match the plan")
        # ...with a nonce THIS control plane issued to THIS node, unused so far:
        if r.challenge_nonce not in issued_nonces.get(r.node_id, set()):
            raise HTTPException(400, "challenge_nonce was not issued to this node (replay or forgery)")
        # ...and measured work that fits inside the wall-clock window it claims.
        now = time.time()
        elapsed_ms = (r.finished_at - r.started_at) * 1000.0
        if r.finished_at > now + 60 or elapsed_ms < 0 or \
                r.measured_work.gpu_active_ms > elapsed_ms * 1.05 + 1000:
            raise HTTPException(400, "measured_work is not plausible for the receipt's time window")

        ok = store.put_receipt(r.receipt_id, r.node_id, r.plan_id, r.challenge_nonce,
                               sr.model_dump(mode="json"))
        if not ok:
            raise HTTPException(409, "duplicate receipt_id or replayed challenge_nonce")
        issued_nonces[r.node_id].discard(r.challenge_nonce)
        credited = credit_for_receipt(sr)   # same formula the on-chain ledger verifies
        store.append_ledger(r.node_id, credited, "work_receipt", r.receipt_id, now)
        return {"ok": True, "credited": credited}

    # ---- pools (API.md): public pool is platform-controlled, private pools are owner-created
    @app.get("/api/v1/pools")
    async def list_pools() -> list[dict]:
        return store.list_pools()

    @app.post("/api/v1/pools")
    async def create_pool(body: dict) -> dict:
        pool_id = str(body.get("id", "")).strip()
        if not pool_id or "/" in pool_id or len(pool_id) > 64:
            raise HTTPException(400, "pool id must be 1-64 chars without '/'")
        if not store.create_pool(pool_id, bool(body.get("private", False)), time.time()):
            raise HTTPException(409, f"pool {pool_id} already exists")
        return {"ok": True, "id": pool_id, "private": bool(body.get("private", False))}

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
        if not rendezvous.session_known(session):
            await ws.close(code=1008, reason="unknown relay session")
            return
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
