"""Signed model manifest (INSTRUCTIONS §2.2, MODEL_CATALOG.md, SECURITY.md).

Public models are described by a signed, content-addressed manifest. No arbitrary
model code — weights are GGUF/safetensors (non-executable). The manifest pins the
exact upstream revision, tokenizer/template/tool-parser, quantization artifact
hashes, supported strategies, and required backend capabilities.

The manifest hash is BLAKE3 over the canonical JSON of the manifest MINUS the
signature envelope, so signing does not change the hash the network addresses.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from .content import blake3_hex
from .identity import canonical_json
from .versioning import PROTOCOL_VERSION


class UpstreamRef(BaseModel):
    provider: str
    repository: str          # HF repo id
    revision: str            # exact commit / content sha — pinned, never a branch


class LicenseRef(BaseModel):
    id: str                  # SPDX id where possible


class TokenizerRef(BaseModel):
    revision: str


class ChatTemplateRef(BaseModel):
    revision: str
    file: str | None = None  # template file within the repo, if separate


class ToolParser(BaseModel):
    id: str                  # exact parser id tested against this model
    notes: str = ""


class ShardFile(BaseModel):
    """One artifact file. artifact_root_hash is the Merkle/BLAKE3 root over its chunks.

    For the POC we record the file's whole-content BLAKE3 as artifact_root_hash and
    a nominal chunk size; the full Merkle chunk graph is a Phase-2 refinement of the
    same field (documented, not faked)."""
    rfilename: str
    size_bytes: int
    artifact_root_hash: str  # BLAKE3 of the file bytes (Phase 1) / Merkle root (Phase 2)
    chunk_bytes: int = 1 << 20


class Quantization(BaseModel):
    id: str                  # e.g. "Q6_K-MAX-MTP"
    files: list[ShardFile] = Field(default_factory=list)
    mtp: bool = False        # multi-token prediction variant
    notes: str = ""


class Parallelism(BaseModel):
    pipeline: bool = True
    tensor: Literal["yes", "no", "conditional"] = "conditional"
    expert: Literal["yes", "no", "conditional"] = "no"
    speculative_draft: bool = False
    replica: bool = True


class Runtime(BaseModel):
    min_meshcompute: str = "0.1.0"
    backends: list[str] = Field(default_factory=list)      # sglang, vllm, llamacpp, lmstudio
    required_backend_capabilities: list[str] = Field(default_factory=list)


class SamplingDefaults(BaseModel):
    temperature: float = 0.7
    top_p: float = 0.8
    top_k: int = 20
    min_p: float = 0.0
    repetition_penalty: float = 1.0
    presence_penalty: float = 0.0


class ModelManifest(BaseModel):
    protocol_version: int = PROTOCOL_VERSION
    id: str                                    # platform alias, e.g. public/qwen3.8-27b-fable
    version: int = 1
    display_name: str = ""
    architecture: str = ""                     # e.g. qwen35
    context_length_max: int = 0                # model's native max
    context_length_configured: int = 0         # what this platform serves (KV-bounded)
    upstream: UpstreamRef
    license: LicenseRef
    runtime: Runtime
    tokenizer: TokenizerRef
    chat_template: ChatTemplateRef
    tool_parser: ToolParser | None = None
    quantizations: list[Quantization] = Field(default_factory=list)
    parallelism: Parallelism = Field(default_factory=Parallelism)
    sampling_defaults: SamplingDefaults = Field(default_factory=SamplingDefaults)
    multimodal: bool = False
    mmproj_files: list[ShardFile] = Field(default_factory=list)
    safety_notes: str = ""
    compat_notes: str = ""

    def manifest_hash(self) -> str:
        """BLAKE3 over canonical JSON of the manifest body (excludes any envelope)."""
        return blake3_hex(canonical_json(self.model_dump(mode="json")))


class SignedManifest(BaseModel):
    manifest: ModelManifest
    manifest_hash: str
    public_b64: str
    signature_b64: str
