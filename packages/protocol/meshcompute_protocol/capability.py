"""Node capability advertisement (INSTRUCTIONS §7).

A worker publishes a *signed* capability record with measured and static fields.
Performance claims are never trusted without benchmark verification — this record
carries claimed fields plus a measured `benchmark` block the scheduler weights by
node reputation (Phase 2 anti-cheat).
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from .versioning import PROTOCOL_VERSION


class GpuInfo(BaseModel):
    vendor: Literal["nvidia", "amd", "apple", "intel", "cpu"] = "cpu"
    model: str = "unknown"
    vram_bytes: int = 0
    free_vram_bytes: int = 0
    compute_capability: str = ""
    backend_support: list[str] = Field(default_factory=list)  # e.g. ["cuda","torch","llamacpp"]


class ContributionPolicy(BaseModel):
    idle_only: bool = True
    max_gpu_percent: int = 90
    max_vram_percent: int = 85
    max_cpu_percent: int = 50
    allow_public_pool: bool = True
    # --- CONFIGURATION.md "availability"/"thermal" controls. All optional +
    # backward-compatible (defaults match CONFIGURATION.md's own example
    # config) so existing callers/tests that only set the fields above are
    # unaffected. Enforced by node/daemon/meshcompute_node/contribution.py.
    idle_minutes_before_start: int = 10
    pause_on_user_activity: bool = True
    require_ac_power: bool = False
    max_ram_gb: int | None = None          # None = no explicit absolute cap
    gpu_temperature_limit_c: int = 82
    schedule: str | None = None            # TODO(phase-1.5): real schedule syntax + parser


class BenchmarkResult(BaseModel):
    """Measured, not claimed. Populated by the daemon's self-benchmark and by
    control-plane challenge tasks. Absent fields mean 'not yet measured'."""
    decode_tokens_per_sec: float | None = None
    prefill_tokens_per_sec: float | None = None
    measured_model: str | None = None
    measured_at: str | None = None  # ISO8601, stamped by caller (scripts, not protocol)


class CapabilityRecord(BaseModel):
    protocol_version: int = PROTOCOL_VERSION
    node_id: str
    runtime_version: str = "0.1.0"
    os: str = "linux"
    arch: str = "x86_64"
    gpus: list[GpuInfo] = Field(default_factory=list)
    ram_free_bytes: int = 0
    storage_share_bytes: int = 0
    max_bandwidth_mbps: int = 0
    # backends this node can actually run (advertise capability, not device name — rule 9)
    backends: list[str] = Field(default_factory=list)
    # strategies this node can participate in
    strategies: list[str] = Field(default_factory=list)  # pipeline, tensor, replica, draft...
    contribution_policy: ContributionPolicy = Field(default_factory=ContributionPolicy)
    benchmark: BenchmarkResult = Field(default_factory=BenchmarkResult)
    # --- internet-native reachability (no VPN / no tailnet) -----------------
    # Candidate addresses for NAT traversal, ICE-style. The rendezvous server
    # exchanges these between peers; QUIC hole-punching / relay uses them.
    local_addrs: list[str] = Field(default_factory=list)   # host:port on LAN interfaces
    reflexive_addr: str | None = None                       # public host:port via STUN
    relay_addr: str | None = None                           # fallback relay endpoint
    nat_type: str = "unknown"        # open | full-cone | restricted | symmetric | unknown
    quic_port: int = 0               # UDP port the QUIC data plane listens on

    def total_vram_bytes(self) -> int:
        return sum(g.vram_bytes for g in self.gpus)

    def free_vram_bytes(self) -> int:
        return sum(g.free_vram_bytes for g in self.gpus)


class SignedCapability(BaseModel):
    """Capability record + detached signature + public key for verification."""
    record: CapabilityRecord
    public_b64: str
    signature_b64: str
