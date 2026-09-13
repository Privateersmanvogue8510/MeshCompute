# MeshCompute Phase-1 Benchmarks

Measured on real hardware, 2026-09-14. The rule (INSTRUCTIONS §26.7): no
performance claim without a benchmark, and always report TTFT + topology, not just
tokens/sec.

## Topology under test

| Node | Where | Hardware | Role |
|------|-------|----------|------|
| GPU pod | Las Vegas | 2x RTX 3090 (48 GB) | runs the 27B model |
| client / control-plane / gateway | Hong Kong | 16-core CPU, 30 GB RAM, no discrete GPU | consumer + mesh services |

Link consumer → GPU pod: **255 ms** RTT mean (mdev 14 ms), cross-Pacific WAN.
This is the physical latency; no VPN or tunnel changes it (see `docs/PHYSICS.md`).

## Model

`public/qwen3.8-27b-fable` — qwen35, 27B, MTP GGUF, 262k native context (served at
32k for the POC). Manifest hash `f88a5778…edbb10` (signed).

## End-to-end over the WAN (full stack: client → gateway → scheduler → pod)

The request routes through the gateway, the control-plane's topology scheduler picks
the pod (strategy `single`), and the gateway streams tokens back from the pod over
the 255 ms link.

| Metric | Value | How measured |
|--------|-------|--------------|
| Time to first streamed chunk | ~4.2 s | thinking-mode model emits a `<think>` block first |
| Steady-state decode | **36.5 tok/s** | tokens over the streaming window, `scripts/register_external_backend.py` |
| Scheduler strategy chosen | `single` (the pod) | `X-Mesh-Strategy` response header |
| Execution path | the scheduled node id | `X-Mesh-Path` response header |

The steady-state decode is pod-bound, not link-bound: streaming pays the 255 ms once
at first token, then tokens pipeline back. This is why serving the whole model on the
pod beats splitting it across the WAN.

Note on TTFT: ~4.2 s is dominated by the model's default thinking mode (a long
`<think>` reasoning block before the first content token). Instruct/non-thinking mode
(`reasoning_effort` low, or the instruct template) cuts this sharply for
latency-sensitive/agent use; the manifest records both sampling presets.

## Scheduler decision (predicted), same topology

The scheduler scored three candidate plans for a representative 512-token prompt +
256-token reply and logged why (`ExecutionPlan.decision_trace`):

| Candidate | Predicted decode | Score (lower better) | Verdict |
|-----------|------------------|----------------------|---------|
| single — GPU pod, stream over WAN | 90 tok/s | **3.5 s** | **selected** |
| single — HK CPU box | 6 tok/s | 47 s | rejected |
| pipeline — pod + CPU split across 255 ms WAN | 2 tok/s | 137 s | rejected |

The WAN layer-split is ~39x worse than running on the pod. The scheduler picks the
pod and records the rejected alternatives.

## "More nodes = faster" (load-aware replica routing)

With three interchangeable replicas of the model and the nearest one busy (queue
depth 3), the scheduler routed to the idle nearby replica instead of the busy fast
one, and reported pool capacity in the trace. Adding replicas lowers queue wait and
raises concurrent capacity — verified in `tests/scheduler`.

## Peer transport + model swarm (loopback, this box)

| Metric | Value |
|--------|-------|
| QUIC frame roundtrip | payload + shape preserved |
| App-level RTT (PING/PONG) | 0.55 ms (loopback) |
| Swarm: 5.25 MB file seed→leech | BLAKE3 root matched; every chunk verified |
| Corrupt chunk in transit | rejected (`ChunkVerifyError`), refetched |
| Resume from partial (3/7 chunks, 1 rotted on disk) | 2 intact kept, 5 re-fetched; final digest matched |

## Two live nodes on one box: peer-to-peer model fetch + channel update (2026-09-14)

Control plane + two `mesh node start` daemons (`MESH_HOME` separates them),
channel `public/smollm2-360m` (258 MiB GGUF, 259 × 1 MiB chunks).

| Step | Observed |
|---|---|
| Node 1 joins | no seeder yet → Hugging Face origin; registers; announces seeding |
| Node 2 joins | `fetching ... from peer nd_633c06ee5 @ <lan>:40494 (rtt 1.54 ms)` → `259/259 chunks` → `obtained from peers (verified)` — Hugging Face never contacted |
| Peer transfer rate | ~10 MiB/s, single QUIC stream, pure-Python `aioquic` (CPU-bound, not disk-bound). Phase-1.5: parallel streams / larger chunks |
| Both nodes advertised | `online`, `storage_share_bytes` = min(--max-storage, free disk) |
| Channel push (v1→v2, v2→v3, v3→v4) | control plane re-scanned the manifest dir and published each within 30 s; both nodes downloaded, swapped engines on the same port, deleted the old file (`now serving vN; deleted 1 old file(s)`) — but in all three waves **both** nodes fetched from Hugging Face: they polled inside the same window and neither held the new file yet. That is the bug the origin-leader election fixes |
| Channel push v4→v5 (with origin-leader election) | node 2 polled first: `no peer is seeding ... fetching it from origin for the network`. node 1: `peer nd_e7e236db2 is already fetching ... waiting to get it from that peer instead` → `259/259 chunks` → `obtained from peers (verified)` → `now serving v5; deleted 1 old file(s)`. One origin download for the wave; the rest peer-to-peer |
| Gateway → scheduler → live node | `POST /v1/chat/completions` through the gateway: reply `'Pong.'`, `x-mesh-strategy: single`, `x-mesh-path: <node 1>`; unknown model → 404; unsigned node registration → 400 |

## Not yet benchmarked (needs hardware/access)

- **Standalone node on GPU.** The self-installing llama.cpp engine is proven on CPU
  with a small model on this box; the CUDA path and the 27B on GPU need a node with
  shell access to the GPU box (currently the pod is reached as an external endpoint).
- **True 2-machine GPU layer split** (llama.cpp RPC) on a low-latency link — needs a
  second GPU peer within ~5 ms (e.g. two nodes in one cloud region). Documented in
  `docs/DEPLOYMENT.md`; the scheduler already admits it via `is_lan_class()`.
- **Real NAT hole-punch / relay** across two independently-NATed internet hosts.

## Reproduce

```bash
# scheduler predictions + load-balancing:
python -m pytest tests/scheduler -q
# live WAN decode against an external OpenAI endpoint configured in deploy/nodes.local.yaml:
python scripts/register_external_backend.py
# full stack: start control-plane + gateway, then: mesh bench --model public/qwen3.8-27b-fable
```
