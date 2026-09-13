"""Inference session objects (PROTOCOL.md "Session handshake", "Activation frame").

The control plane issues a signed execution plan + peer tickets. Each peer validates
the ticket signature, expiry, model hash and its assigned role before doing work.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field

from .versioning import PROTOCOL_VERSION


class Strategy(str, Enum):
    SINGLE = "single"            # one peer / one pod runs the whole model
    PIPELINE = "pipeline"        # transformer blocks split across peers
    TENSOR = "tensor"            # tensor-parallel within a low-latency pod
    EXPERT = "expert"            # MoE experts distributed
    SPECULATIVE = "speculative"  # draft peer + verifier peer
    REPLICA = "replica"          # independent replicas for concurrency


class ShardAssignment(BaseModel):
    node_id: str
    role: Strategy
    layer_start: int | None = None
    layer_end: int | None = None
    backend: str = ""
    # for rpc/pipeline: the address this shard is reachable on
    addr: str | None = None


class ExecutionPlan(BaseModel):
    """Reproducible enough to explain why a request landed on specific peers
    (INSTRUCTIONS §8, ARCHITECTURE.md scheduling cost model)."""
    protocol_version: int = PROTOCOL_VERSION
    plan_id: str
    model_manifest_hash: str
    model_id: str
    strategy: Strategy
    peer_chain: list[str]                      # ordered node_ids
    shard_assignment: list[ShardAssignment] = Field(default_factory=list)
    context_length: int = 0
    # scheduler decision trace — WHY this plan (INSTRUCTIONS §8 last line)
    predicted_ttft_s: float = 0.0
    predicted_decode_tps: float = 0.0
    decision_trace: list[str] = Field(default_factory=list)
    rejected_alternatives: list[str] = Field(default_factory=list)


class InferenceSessionOpen(BaseModel):
    protocol_version: int = PROTOCOL_VERSION
    plan_id: str
    session_id: str
    model_manifest_hash: str
    strategy: Strategy
    shard_assignment: list[ShardAssignment]
    context_parameters: dict = Field(default_factory=dict)
    peer_chain: list[str]
    expiration: float                          # epoch seconds (stamped by caller)
    signed_ticket: str = ""                    # control-plane signature over the plan


class FailurePolicy(str, Enum):
    RETRY_PEER = "retry_peer"
    BYPASS_PEER = "bypass_peer"
    REBUILD_PATH = "rebuild_path"
    RESTART_GENERATION = "restart_generation"  # Phase-1 acceptable (ARCHITECTURE.md)
    FAIL_CLEAN = "fail_clean"
