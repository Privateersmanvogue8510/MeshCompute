"""The standalone MeshCompute node daemon (INSTRUCTIONS §2.1, §7, §9).

One program a user launches. It installs its own inference engine, joins the
network, downloads the model it needs, runs inference itself, contributes its
measured compute share to the network, and exposes a local OpenAI-compatible
endpoint. No LM Studio, no Ollama — llama.cpp runs embedded (runtime_installer
+ backends/llamacpp), the same way an app bundles ffmpeg.

`mesh node start` (apps/cli/meshcompute_cli/main.py) calls run()/async_main()
with backend/backend_url/pool_id/idle_only/max_vram_percent/control_url; this
module accepts those exact names plus the richer names from the daemon spec
(model, pool, port, max_vram, smoke) as aliases, and ignores anything unknown.
"""

from __future__ import annotations

import asyncio
import contextlib
import signal
from pathlib import Path

import httpx

from meshcompute_protocol import (
    CapabilityRecord,
    HeartbeatRequest,
    NodeIdentity,
    RegisterRequest,
    SignedCapability,
    SignedManifest,
)
from meshcompute_runtime.backends.base import BackendAdapter, ChatRequest
from meshcompute_runtime.backends.lmstudio import OpenAICompatBackend
from meshcompute_runtime.backends.llamacpp import LlamaCppBackend
from meshcompute_runtime.runtime_installer import SMOKE_MODEL, detect_accel, ensure_llama_server, ensure_model

from .capability import measure_capability, ram_free_bytes
from .rendezvous_client import RendezvousClient
from .transport_quic import QuicTransport

DEFAULT_CONTROL_URL = "http://127.0.0.1:8080"          # matches apps/cli/meshcompute_cli/config.py
DEFAULT_IDENTITY_PATH = Path.home() / ".mesh" / "identity.json"
HEARTBEAT_INTERVAL_S = 30.0


# --------------------------------------------------------------------------- model resolution
async def _resolve_model(model: str | None, control_url: str):
    """A `model` alias (e.g. "public/qwen3.8-27b-fable") resolves against the
    control-plane catalog; missing/unreachable/None falls back to the
    CPU-box smoke default (this is only the smoke default — the real target
    model is for GPU nodes, per the daemon spec)."""
    if model:
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                resp = await client.get(f"{control_url}/api/v1/models/{model}")
            if resp.status_code == 200:
                signed = SignedManifest.model_validate(resp.json())
                print(f"[mesh] resolved model '{model}' from control-plane catalog "
                      f"({signed.manifest_hash[:12]})")
                return signed.manifest
        except httpx.HTTPError:
            pass
        print(f"[mesh] could not resolve model '{model}' from {control_url}; "
              f"falling back to the CPU smoke model")
    return SMOKE_MODEL


# --------------------------------------------------------------------------- control-plane calls
async def _try_register(control_url: str, identity: NodeIdentity, pool_id: str) -> bool:
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(
                f"{control_url}/api/v1/nodes/register",
                json=RegisterRequest(node_id=identity.node_id, public_b64=identity.public_b64,
                                      pool_ids=[pool_id]).model_dump(mode="json"))
            resp.raise_for_status()
        return True
    except httpx.HTTPError:
        return False


async def _post_capability(control_url: str, identity: NodeIdentity, backend: BackendAdapter,
                            config: dict) -> CapabilityRecord:
    record = await measure_capability(identity, backend, config)
    signature = identity.sign_json(record.model_dump(mode="json"))
    signed = SignedCapability(record=record, public_b64=identity.public_b64, signature_b64=signature)
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(f"{control_url}/api/v1/nodes/{identity.node_id}/capabilities",
                                  json=signed.model_dump(mode="json"))
        resp.raise_for_status()
    return record


async def _announce_rendezvous(control_url: str, identity: NodeIdentity, quic_port: int) -> None:
    try:
        await RendezvousClient(control_url).announce(
            identity, local_addrs=[], reflexive_addr=None, quic_port=quic_port)
    except httpx.HTTPError:
        pass  # best-effort per the daemon spec: skip cleanly if unreachable


async def _heartbeat_loop(control_url: str, identity: NodeIdentity, interval_s: float,
                           stop_event: asyncio.Event) -> None:
    async with httpx.AsyncClient(timeout=10.0) as client:
        while not stop_event.is_set():
            with contextlib.suppress(httpx.HTTPError):
                await client.post(
                    f"{control_url}/api/v1/nodes/{identity.node_id}/heartbeat",
                    json=HeartbeatRequest(node_id=identity.node_id,
                                           ram_free_bytes=ram_free_bytes()).model_dump(mode="json"))
            with contextlib.suppress(asyncio.TimeoutError):
                await asyncio.wait_for(stop_event.wait(), timeout=interval_s)


# --------------------------------------------------------------------------- main flow
async def async_main(*, backend: str = "llamacpp", backend_url: str | None = None,
                      model: str | None = None, pool_id: str = "public", pool: str | None = None,
                      control_url: str | None = None, port: int | None = None,
                      idle_only: bool = True, max_vram_percent: int = 85,
                      max_vram: int | None = None, smoke: bool = False,
                      identity_path: str | None = None, ctx_size: int = 2048,
                      heartbeat_interval_s: float = HEARTBEAT_INTERVAL_S, **_ignored) -> None:
    pool_id = pool or pool_id
    max_vram_percent = max_vram if max_vram is not None else max_vram_percent
    control_url = (control_url or DEFAULT_CONTROL_URL).rstrip("/")

    identity = NodeIdentity.load_or_create(identity_path or DEFAULT_IDENTITY_PATH)
    print(f"[mesh] node identity: {identity.node_id}")

    backend_adapter: BackendAdapter
    llama_backend: LlamaCppBackend | None = None

    if backend == "llamacpp":
        accel = detect_accel()
        print(f"[mesh] detected accel: {accel}")
        server_bin = await ensure_llama_server(accel=accel)
        print(f"[mesh] engine ready: {server_bin}")

        manifest_or_spec = await _resolve_model(model, control_url)
        model_path = await ensure_model(manifest_or_spec)
        print(f"[mesh] model ready: {model_path}")

        # TODO(phase-1.5): also offload on accel=="metal" once tested on an Apple box.
        n_gpu_layers = 999 if accel == "cuda" else 0
        llama_backend = LlamaCppBackend(server_bin, model_path, host="127.0.0.1", port=port,
                                         ctx_size=ctx_size, n_gpu_layers=n_gpu_layers)
        await llama_backend.start()
        backend_adapter = llama_backend
        endpoint = f"{llama_backend.base_url}/v1"
    else:
        if not backend_url:
            raise SystemExit(f"--backend {backend} requires --backend-url")
        backend_adapter = OpenAICompatBackend(backend_url, backend_name=backend)
        endpoint = f"{backend_url.rstrip('/')}/v1"

    print(f"[mesh] Local endpoint ready: {endpoint}  (point your OpenAI client here)")

    if smoke:
        models = await backend_adapter.list_models()
        model_id = models[0] if models else "local"
        req = ChatRequest(model=model_id,
                           messages=[{"role": "user", "content": "Say hello in exactly three words."}],
                           max_tokens=32, temperature=0.2, stream=True)
        text = "".join([c.text async for c in backend_adapter.chat_stream(req) if c.text])
        print(f"[mesh] smoke self-test completion: {text!r}")

    transport = QuicTransport(identity, bind_port=0)
    await transport.start()

    stop_event = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        with contextlib.suppress(NotImplementedError, RuntimeError):
            loop.add_signal_handler(sig, stop_event.set)

    heartbeat_task: asyncio.Task | None = None
    registered = False
    try:
        registered = await _try_register(control_url, identity, pool_id)
        if registered:
            config = {"contribution_policy": {"idle_only": idle_only,
                                               "max_vram_percent": max_vram_percent}}
            record = await _post_capability(control_url, identity, backend_adapter, config)
            print(f"[mesh] posted signed capability: decode "
                  f"{record.benchmark.decode_tokens_per_sec} tok/s, backends={record.backends}")
            await _announce_rendezvous(control_url, identity, transport.local_quic_port)
            heartbeat_task = asyncio.create_task(
                _heartbeat_loop(control_url, identity, heartbeat_interval_s, stop_event))
            print(f"[mesh] registered with control plane at {control_url}; "
                  f"contributing to pool '{pool_id}'")
        else:
            print(f"[mesh] control plane at {control_url} unreachable; running in SOLO mode "
                  f"(local endpoint only — not contributing to the network)")

        if smoke:
            return
        await stop_event.wait()
    finally:
        print("[mesh] shutting down...")
        if heartbeat_task is not None:
            heartbeat_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await heartbeat_task
        # TODO(phase-1.5): the frozen HTTP contract (api_models.py) has no
        # deregister endpoint yet; stopping the heartbeat is the best-effort
        # signal today — the control plane's own HEARTBEAT_TIMEOUT_S (90s)
        # marks us offline.
        await transport.stop()
        if llama_backend is not None:
            llama_backend.stop()
        print("[mesh] stopped.")


def run(**opts) -> None:
    asyncio.run(async_main(**opts))


if __name__ == "__main__":
    run(smoke=True)
