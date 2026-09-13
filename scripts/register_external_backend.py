#!/usr/bin/env python3
"""Register an EXISTING OpenAI-compatible endpoint (e.g. LM Studio) as a mesh node.

This is a stand-in for the full node daemon, used to demo the end-to-end stack with
a backend that already exists (the 2x3090 pod running LM Studio). It is NOT the
product's default path: the standalone node daemon installs its own llama.cpp
runtime and downloads the model itself (no LM Studio required). Use this only when a
node already exposes an OpenAI-compatible server you want the mesh to route to.

It: measures the endpoint's real steady-state decode tok/s, creates a signed node
identity, registers + advertises a signed capability with that measurement, and
writes the derived node_id back into deploy/nodes.local.yaml so the gateway can map
node_id -> backend_url.
"""
from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path

import httpx
import yaml

sys.path[:0] = ["packages/protocol", "node/runtime"]
from meshcompute_protocol import (  # noqa: E402
    NodeIdentity, CapabilityRecord, GpuInfo, BenchmarkResult, SignedCapability,
    RegisterRequest, HeartbeatRequest,
)
from meshcompute_runtime.backends import OpenAICompatBackend, ChatRequest  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
LOCAL = REPO / "deploy" / "nodes.local.yaml"


async def measure_decode(backend: OpenAICompatBackend, model: str) -> tuple[float, float]:
    """Steady-state decode tok/s over the streaming window (ignores initial think
    latency) + time-to-first-streamed-chunk. Works for reasoning models that emit a
    long <think> block first."""
    req = ChatRequest(model=model, max_tokens=160, temperature=0.6, top_p=0.95, top_k=20,
                      stream=True,
                      messages=[{"role": "user", "content":
                                 "List the numbers 1 through 40 separated by commas."}])
    t0 = time.monotonic()
    first = None
    last = None
    n = 0
    async for ch in backend.chat_stream(req):
        if ch.text:
            now = time.monotonic()
            if first is None:
                first = now
            last = now
            n += ch.text.count(" ") + 1  # rough token proxy
    if not first or not last or last <= first:
        return 0.0, (first - t0 if first else 0.0)
    return round(n / (last - first), 1), round(first - t0, 2)


async def main() -> None:
    cfg = yaml.safe_load(LOCAL.read_text())
    control_url = cfg["gateway"]["control_url"]
    node_cfg = cfg["nodes"][0]
    backend_url = node_cfg["backend_url"]
    pool = node_cfg.get("pool", "public")

    backend = OpenAICompatBackend(backend_url, backend_name=node_cfg.get("backend", "lmstudio"))
    if not await backend.health():
        print(f"backend {backend_url} not reachable"); sys.exit(1)
    models = await backend.list_models()
    model = models[0]
    print(f"backend up: {backend_url}, model {model}")

    decode_tps, ttft = await measure_decode(backend, model)
    print(f"measured: decode ~{decode_tps} tok/s (steady-state), first-chunk {ttft}s")

    ident = NodeIdentity.load_or_create("deploy/identities/external-pod.key.json")
    node_id = ident.node_id
    gpus = int(node_cfg.get("gpus", 2))
    vram_each = int(node_cfg.get("vram_bytes_each", 24_000_000_000))
    rec = CapabilityRecord(
        node_id=node_id, backends=[node_cfg.get("backend", "lmstudio")],
        strategies=["single", "replica", "tensor"],
        gpus=[GpuInfo(vendor="nvidia", model="RTX 3090", vram_bytes=vram_each,
                      free_vram_bytes=vram_each, backend_support=["cuda", "lmstudio"])
              for _ in range(gpus)],
        ram_free_bytes=64_000_000_000,
        benchmark=BenchmarkResult(decode_tokens_per_sec=decode_tps,
                                  measured_model=model, measured_at="live"))
    # node_id -> backend_url mapping is carried in deploy/nodes.local.yaml (written
    # below), which is how the gateway resolves the worker endpoint.
    signed = SignedCapability(record=rec, public_b64=ident.public_b64,
                              signature_b64=ident.sign_json(rec.model_dump(mode="json")))

    async with httpx.AsyncClient(timeout=15.0) as c:
        r = await c.post(f"{control_url}/api/v1/nodes/register",
                         json=RegisterRequest(node_id=node_id, public_b64=ident.public_b64,
                                              pool_ids=[pool]).model_dump(mode="json"))
        r.raise_for_status()
        print("registered:", r.json())
        r = await c.post(f"{control_url}/api/v1/nodes/{node_id}/capabilities",
                         json=signed.model_dump(mode="json"))
        r.raise_for_status()
        print("capability accepted:", r.json())
        r = await c.post(f"{control_url}/api/v1/nodes/{node_id}/heartbeat",
                         json=HeartbeatRequest(node_id=node_id, free_vram_bytes=gpus * vram_each,
                                              ram_free_bytes=64_000_000_000).model_dump(mode="json"))
        r.raise_for_status()
        print("heartbeat ok")

    # write node_id back so the gateway maps it -> backend_url
    node_cfg["node_id"] = node_id
    LOCAL.write_text(yaml.safe_dump(cfg, sort_keys=False))
    print(f"\nnode_id {node_id} registered and written to {LOCAL.name}")


if __name__ == "__main__":
    asyncio.run(main())
