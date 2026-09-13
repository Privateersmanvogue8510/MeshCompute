"""Signed work receipts (PROTOCOL.md "Work receipts", INSTRUCTIONS §11-12).

Every completed assignment produces a signed receipt. The accounting service
validates the signature and the challenge nonce before crediting contribution.
Self-reported tokens/sec is never trusted on its own; measured_work is cross-checked
against the plan and (Phase 2) sampled redundant execution.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field

from .versioning import PROTOCOL_VERSION


class Outcome(str, Enum):
    OK = "ok"
    PARTIAL = "partial"
    FAILED = "failed"
    CANCELLED = "cancelled"


class MeasuredWork(BaseModel):
    gpu_active_ms: float = 0.0
    layers_processed: int = 0
    tokens_generated: int = 0
    activation_bytes: int = 0
    model_shard_bytes_served: int = 0
    draft_tokens_accepted: int = 0


class WorkReceipt(BaseModel):
    protocol_version: int = PROTOCOL_VERSION
    receipt_id: str
    session_id: str
    plan_id: str
    node_id: str
    model_manifest_hash: str
    role: str
    measured_work: MeasuredWork = Field(default_factory=MeasuredWork)
    started_at: float = 0.0   # epoch seconds, stamped by caller
    finished_at: float = 0.0
    outcome: Outcome = Outcome.OK
    challenge_nonce: str = ""  # anti-replay: control plane issued, echoed here


class SignedReceipt(BaseModel):
    receipt: WorkReceipt
    public_b64: str
    signature_b64: str
