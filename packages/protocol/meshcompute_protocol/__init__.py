"""MeshCompute protocol package — versioned wire objects shared across services.

Nothing here talks to the network; these are the schemas + primitives (identity,
content addressing, framing) that the control plane, gateway, node daemon and
runtime all agree on. Every object carries protocol_version (INSTRUCTIONS §26.8).
"""

from __future__ import annotations

from .versioning import PROTOCOL_VERSION, MIN_RUNTIME_VERSION, is_compatible
from .content import blake3_hex, chunk_id, verify_chunk, DEFAULT_CHUNK_BYTES
from .identity import (
    NodeIdentity,
    canonical_json,
    verify,
    verify_json,
    node_id_from_public,
)
from .capability import (
    CapabilityRecord,
    SignedCapability,
    GpuInfo,
    ContributionPolicy,
    BenchmarkResult,
)
from .manifest import (
    ModelManifest,
    SignedManifest,
    UpstreamRef,
    LicenseRef,
    Runtime,
    TokenizerRef,
    ChatTemplateRef,
    ToolParser,
    Quantization,
    ShardFile,
    Parallelism,
    SamplingDefaults,
)
from .session import (
    Strategy,
    ShardAssignment,
    ExecutionPlan,
    InferenceSessionOpen,
    FailurePolicy,
)
from .receipts import WorkReceipt, SignedReceipt, MeasuredWork, Outcome
from .frames import Frame, PacketClass, Dtype, Compression, MAX_PAYLOAD, MAX_NDIM
from .ledger import (
    LedgerEntry, BlockHeader, Block, Chain, ChainError,
    merkle_root, credit_for_receipt, pick_best_chain, verify_block, GENESIS_PREV,
)
from .api_models import (
    RegisterRequest, RegisterResponse, HeartbeatRequest, HeartbeatResponse,
    NodeView, ModelView, ScheduleRequest,
    RendezvousAnnounce, PeerCandidate, RendezvousPeers, ConnectRequest, ConnectTicket,
)

__all__ = [
    "PROTOCOL_VERSION", "MIN_RUNTIME_VERSION", "is_compatible",
    "blake3_hex", "chunk_id", "verify_chunk", "DEFAULT_CHUNK_BYTES",
    "NodeIdentity", "canonical_json", "verify", "verify_json", "node_id_from_public",
    "CapabilityRecord", "SignedCapability", "GpuInfo", "ContributionPolicy", "BenchmarkResult",
    "ModelManifest", "SignedManifest", "UpstreamRef", "LicenseRef", "Runtime",
    "TokenizerRef", "ChatTemplateRef", "ToolParser", "Quantization", "ShardFile",
    "Parallelism", "SamplingDefaults",
    "Strategy", "ShardAssignment", "ExecutionPlan", "InferenceSessionOpen", "FailurePolicy",
    "WorkReceipt", "SignedReceipt", "MeasuredWork", "Outcome",
    "Frame", "PacketClass", "Dtype", "Compression", "MAX_PAYLOAD", "MAX_NDIM",
    "RegisterRequest", "RegisterResponse", "HeartbeatRequest", "HeartbeatResponse",
    "NodeView", "ModelView", "ScheduleRequest",
    "RendezvousAnnounce", "PeerCandidate", "RendezvousPeers", "ConnectRequest", "ConnectTicket",
    "LedgerEntry", "BlockHeader", "Block", "Chain", "ChainError",
    "merkle_root", "credit_for_receipt", "pick_best_chain", "verify_block", "GENESIS_PREV",
]
