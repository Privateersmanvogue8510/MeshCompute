"""HTTP API request/response models — the FROZEN contract between services.

The control-plane app, the gateway and the node daemon all import these. Do not
change a field without bumping PROTOCOL_VERSION and noting compatibility behavior.

Endpoints (implemented by apps/control-plane):
  POST /api/v1/nodes/register        -> RegisterResponse
  POST /api/v1/nodes/{id}/heartbeat  -> HeartbeatResponse
  POST /api/v1/nodes/{id}/capabilities (SignedCapability) -> {ok}
  GET  /api/v1/nodes                 -> [NodeView]
  GET  /api/v1/models                -> [ModelView]           (native)
  GET  /api/v1/models/{alias}        -> SignedManifest
  POST /api/v1/schedule (ScheduleRequest) -> ExecutionPlan
  POST /api/v1/work-receipts (SignedReceipt) -> {ok, credited}
  # rendezvous (internet-native NAT traversal, no VPN):
  POST /api/v1/rendezvous/announce (RendezvousAnnounce) -> RendezvousPeers
  POST /api/v1/rendezvous/connect  (ConnectRequest) -> ConnectTicket
  GET  /api/v1/rendezvous/relay/{session}  (WebSocket/stream) -> relayed frames

Gateway (apps/gateway) exposes the OpenAI-compatible surface (API.md):
  GET  /v1/models
  POST /v1/chat/completions   (stream=true -> SSE)
  POST /api/v1/sessions ...   (native sessions)
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from .capability import SignedCapability
from .versioning import PROTOCOL_VERSION


class RegisterRequest(BaseModel):
    protocol_version: int = PROTOCOL_VERSION
    node_id: str
    public_b64: str
    pool_ids: list[str] = Field(default_factory=lambda: ["public"])


class RegisterResponse(BaseModel):
    ok: bool
    node_id: str
    challenge_nonce: str = ""           # anti-replay seed for the next receipt
    heartbeat_interval_s: int = 30
    rendezvous_url: str = ""


class HeartbeatRequest(BaseModel):
    node_id: str
    free_vram_bytes: int = 0
    ram_free_bytes: int = 0
    queue_depth: int = 0
    active_sessions: int = 0
    cached_manifest_hashes: list[str] = Field(default_factory=list)


class HeartbeatResponse(BaseModel):
    ok: bool
    assignments: list[str] = Field(default_factory=list)   # plan_ids assigned to this node
    challenge_nonce: str = ""


class NodeView(BaseModel):
    node_id: str
    online: bool
    pool_ids: list[str]
    capability: SignedCapability | None = None
    last_seen: float = 0.0


class ModelView(BaseModel):
    id: str                             # platform alias
    manifest_hash: str
    display_name: str = ""
    context_length: int = 0
    signed: bool = False


class ScheduleRequest(BaseModel):
    protocol_version: int = PROTOCOL_VERSION
    model_id: str                       # alias or manifest hash
    pool_id: str = "public"
    client_node_id: str = "client"      # where the consumer sits (for topology cost)
    context_length: int = 0             # 0 -> manifest default
    prompt_tokens: int = 512            # estimate for prefill cost
    gen_tokens: int = 256
    priority: int = 0
    user_id: str = "anon"


# --- rendezvous (public tracker; BitTorrent-like peer introduction) ----------
class RendezvousAnnounce(BaseModel):
    protocol_version: int = PROTOCOL_VERSION
    node_id: str
    public_b64: str
    local_addrs: list[str] = Field(default_factory=list)
    reflexive_addr: str | None = None   # STUN-discovered public ip:port
    quic_port: int = 0
    nat_type: str = "unknown"
    # content the node can seed (manifest/chunk availability for the swarm)
    seeding_manifest_hashes: list[str] = Field(default_factory=list)
    signature_b64: str = ""             # sign(canonical_json of the above minus this)


class PeerCandidate(BaseModel):
    node_id: str
    public_b64: str
    local_addrs: list[str] = Field(default_factory=list)
    reflexive_addr: str | None = None
    quic_port: int = 0
    nat_type: str = "unknown"
    relay_addr: str | None = None


class RendezvousPeers(BaseModel):
    ok: bool
    peers: list[PeerCandidate] = Field(default_factory=list)
    your_reflexive_addr: str | None = None   # server-observed source addr (STUN-like)


class ConnectRequest(BaseModel):
    from_node_id: str
    to_node_id: str
    plan_id: str = ""
    signature_b64: str = ""


class ConnectTicket(BaseModel):
    ok: bool
    session_id: str
    peer: PeerCandidate | None = None
    method: str = "holepunch"           # holepunch | direct | relay
    relay_url: str | None = None
    expires_at: float = 0.0
    control_signature_b64: str = ""     # control-plane authorises the session
