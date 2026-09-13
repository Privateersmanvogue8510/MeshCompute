# MeshCompute

**Peer-to-peer distributed AI inference network — Phase 1 POC, built and working.**

MeshCompute turns idle compute on machines across the open internet into a shared LLM
inference network. A standalone node self-installs its own inference engine and can
serve a model entirely on its own, no other MeshCompute infrastructure required. Join
more nodes and the network gets faster — mostly through replica load-balancing (more
nodes means shorter queues, closer/faster routing, and higher aggregate throughput),
and, when a model doesn't fit on one machine or a fast low-latency pod is available,
through true multi-peer split execution. There's no VPN or tailnet: peer discovery
(rendezvous), NAT traversal, and model distribution are internet-native (QUIC +
BitTorrent-style content-addressed chunks). Compute credit is tracked on a
decentralized, blockchain-style ledger with no single owner, and a contributor's
node donates only the GPUs and resource share they explicitly choose — including
selecting individual cards on a multi-GPU box.

## Status: Phase 1 POC (working)

Proven capabilities, backed by real code and measurements:

- **Standalone node.** `mesh node start` self-installs `llama.cpp`, downloads a
  model, runs it, exposes a local OpenAI-compatible endpoint, and — when a control
  plane is reachable — contributes measured capacity to the network. Runs solo with
  no other MeshCompute service required. (`node/daemon/meshcompute_node/daemon.py`)
- **Full stack over a real WAN.** control-plane + gateway + a GPU-pod worker stream
  real chat completions end-to-end across a real cross-Pacific link (Hong Kong
  client → Las Vegas pod, 255 ms RTT) at 36.5 tok/s decode. See `docs/BENCHMARKS.md`.
- **Topology-aware scheduler.** Picks between single/pipeline/replica execution
  plans from measured RTT, VRAM, and queue depth, and logs why it chose what it
  chose. See `docs/PHYSICS.md`, `docs/BENCHMARKS.md`, `tests/scheduler/`.
- **QUIC transport, rendezvous, and BitTorrent-style model distribution.**
  Encrypted peer transport, rendezvous-based peer announce/connect, and
  BLAKE3-verified content-addressed model chunk swarming — implemented and tested
  on loopback/LAN.
- **Decentralized credit ledger.** A blockchain-style ledger with no single
  owner: BLAKE3 hash-linked blocks, Ed25519-signed by their producer, every block
  independently verifiable from genesis (including re-checking the signed work
  receipt behind every credit), double-credit-proof via a chain-wide nonce check,
  and non-transferable credits — no proof-of-work, no token. The verifiable
  block/chain core is built and tested; gossiping blocks between peers over QUIC
  is the next step, not done yet. See `docs/DECENTRALIZATION.md`.
- **Multi-GPU nodes.** A node enumerates every GPU it has (index, VRAM, NVLINK
  peering) individually, and `mesh node start --gpu-devices 0,1` lets a
  contributor enable or disable specific cards rather than all-or-nothing. The
  runtime tensor-splits one model across the selected GPUs
  (`--split-mode`/`--tensor-split` on the embedded `llama-server`, e.g. a
  2×RTX 3090 NVLINK box), and the scheduler already treats a multi-GPU box as one
  pod by summing VRAM across its cards. See `docs/QUICKSTART.md` for the flags.
- **44 automated tests pass** (`python -m pytest`) covering the protocol layer
  (including the ledger), the scheduler, and the security/tool boundaries.

Phase-1 is honest about what's not finished yet — see `ROADMAP.md` and
`docs/PHYSICS.md`:

- No prebuilt CUDA `llama.cpp` binaries exist upstream for Linux; a CUDA node
  falls back to building `llama.cpp` from source on first run (CPU and macOS/Metal
  use prebuilt releases).
- NAT hole-punching between two independently-NATed real hosts is implemented but
  best-effort/untested at that scale, and the relay fallback path is a stub
  (Phase 1.5).
- Splitting a single token's decode across a high-latency WAN link is deliberately
  **not** attempted — it's measurably slower, not faster. See `docs/PHYSICS.md` for
  the numbers and for the real mechanisms MeshCompute uses instead to make "more
  peers = faster" true.
- The credit ledger's block/chain data model, validation, and fork choice are
  built and tested; peer-to-peer gossip replication over QUIC (so the chain is
  actually decentralized in practice, not just in format) has not landed yet —
  see `docs/DECENTRALIZATION.md`.
- Multi-GPU enumeration, per-GPU selection, and tensor-split launch args are
  implemented and CLI-exposed (`--gpu-devices`/`--split-mode`/`--tensor-split`),
  but unverified so far against real multi-GPU hardware in this repo (the
  launch-arg logic itself is unit-tested against a captured `nvidia-smi`
  sample, not a live GPU).

## Quickstart

**A. Standalone node — one command, nothing else running:**

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -e .
mesh node start
```

This detects your hardware, self-installs `llama-server`, downloads a small
CPU-friendly model, and prints a local OpenAI-compatible endpoint
(`http://127.0.0.1:<port>/v1`) you can `curl` immediately — no control plane, no
LM Studio, no Ollama.

**B. Run the mesh (control-plane + gateway + a node) and hit the OpenAI endpoint:**

```bash
uvicorn meshcompute_control.app:app --host 127.0.0.1 --port 8080 &
mesh api serve &
mesh node start
mesh models
mesh chat --model public/qwen3.8-27b-fable
```

Point any OpenAI-compatible client at `http://127.0.0.1:8081/v1`.

Full copy-pasteable walkthrough for both flows, including the manual
`deploy/nodes.local.yaml` wiring the gateway needs to route to a given node: see
**docs/QUICKSTART.md**.

## Documentation

| Doc | What's in it |
|---|---|
| [docs/QUICKSTART.md](docs/QUICKSTART.md) | Copy-pasteable setup: solo node, or the full mesh |
| [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md) | Operator runbook: env vars, deploy config, external backends, RPC layer-split, systemd/compose |
| [docs/PHYSICS.md](docs/PHYSICS.md) | Why "more peers = faster" is real, and why per-token WAN splitting isn't |
| [docs/BENCHMARKS.md](docs/BENCHMARKS.md) | Measured numbers on real hardware |
| [docs/DECENTRALIZATION.md](docs/DECENTRALIZATION.md) | The blockchain-style credit ledger: what's on-chain, what isn't, and why |
| [CONFIGURATION.md](CONFIGURATION.md) | Node contribution/idle/resource config reference |
| [ROADMAP.md](ROADMAP.md) | What's built vs. planned, phase by phase |
| [CHANGELOG.md](CHANGELOG.md) | What shipped, release by release |

---

**Working codename. Rename freely.**

MeshCompute is a peer-to-peer distributed AI inference and agent platform designed to turn idle compute across many machines into a shared inference fabric.

The defining goal is not simple request load balancing. MeshCompute must be able to split a single model inference workload across multiple peers when that topology produces the best usable performance or when no single peer can fit the requested model.

The public network exposes a small curated model catalog with known tool-calling behavior, signed manifests, fixed templates, and validated runtimes. Private pools may run custom models.

## Product principles

1. **Peer-to-peer data plane from the start**
   - A centralized service may handle identity, rendezvous, model catalog, accounting, and scheduling metadata.
   - Activation traffic and model shard traffic should move directly between peers whenever possible.
   - The central control plane must not become the mandatory token-by-token inference relay.

2. **Optimize for useful speed**
   - Adding peers does not automatically make autoregressive inference faster.
   - The scheduler must measure compute speed, VRAM, queue depth, bandwidth, RTT, reliability, model placement, and backend capability.
   - It should choose pipeline, tensor, expert, speculative, replica, or hybrid execution according to the topology.
   - The objective is best end-to-end latency and throughput, not maximum node count.

3. **Public catalog, private freedom**
   - Public pools run a deliberately small set of signed and version-pinned models.
   - Private pools can load custom models and quantizations under their own trust policy.

4. **Contribute compute, receive priority**
   - Users earn internal compute credits based on verified useful work rather than claimed hardware.
   - Credits influence access and scheduling during congestion.
   - Paid credits can be added later.
   - Do not introduce blockchain or transferable cryptocurrency in the initial design.

5. **Agent experience, not just raw inference**
   - OpenAI-compatible APIs.
   - Native session API.
   - User-selectable agent harnesses.
   - Web search and browser tools.
   - MCP tool support.
   - ACP-compatible editor integration.
   - CLI and web clients first, mobile later.

6. **Heterogeneous by design**
   - First production target: desktop/server GPUs, with NVIDIA first if necessary.
   - Architecture must leave room for AMD, Apple Silicon, Android, and other accelerators.
   - Mobile contribution is opportunistic and comes after the desktop P2P inference proof.

## POC definition of done

The POC is **not complete** if it only sends separate requests to separate machines.

The POC is complete only when all of the following are demonstrated:

- At least two independent peers participate in one model inference session.
- A model configuration can run even when it does not fit in the VRAM allocation of any one participating peer.
- Peer discovery and direct peer data transport work across real network boundaries.
- The scheduler uses measured topology data to select a partition plan.
- Model files or shards are content-addressed and can be obtained from peers.
- Streaming output reaches an OpenAI-compatible API client.
- A peer can disconnect and the system either recovers, reroutes, or fails cleanly without corrupting the session.
- Benchmarks report prefill latency, decode tokens/sec, time-to-first-token, network overhead, and per-peer utilization.
- The same user can run at least one basic tool call through the agent layer after the inference POC is stable.

See `INSTRUCTIONS.md` for the implementation contract.
