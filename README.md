# MeshCompute

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
