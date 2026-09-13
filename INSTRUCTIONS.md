# MeshCompute - Master Implementation Instructions

Date baseline: 2026-09-14

This file is the primary instruction document for coding agents working on MeshCompute. Read this file before making architectural changes.

## 1. Mission

Build a usable peer-to-peer AI compute network where people contribute idle GPU, CPU, RAM, storage, and network capacity, receive compute credit for verified work, and consume inference from the shared network.

The platform must support true distributed inference for a single model across multiple machines. Request-level load balancing alone is not the product.

The product should eventually feel like a combination of:

- a local-model runtime,
- a community compute swarm,
- an OpenAI-compatible inference endpoint,
- a Claude Code / OpenCode-style agent environment,
- an MCP tool host,
- a browser-capable research agent,
- and a portable account/session layer usable from desktop, web, Android, and iOS.

## 2. Hard requirements

These are non-negotiable unless the project owner explicitly changes them.

### 2.1 P2P inference is foundational

From the first POC, the architecture must allow one generation to traverse multiple peers.

A control plane is allowed for:
- authentication,
- peer registration,
- rendezvous,
- model catalog,
- scheduling metadata,
- credit accounting,
- reputation,
- telemetry,
- policy.

The control plane must not be the high-bandwidth activation relay by default.

After rendezvous, peers should create direct encrypted links using QUIC. Use NAT traversal and relay fallback where direct connectivity is unavailable.

### 2.2 Public network model catalog is curated

The public network must not accept arbitrary model code.

Each public model entry is a signed manifest containing at minimum:
- model identifier,
- upstream source and license metadata,
- exact revision or content hash,
- tokenizer revision,
- chat template,
- tool parser,
- supported quantizations,
- tensor format,
- minimum runtime version,
- supported inference strategies,
- required kernel/backend capabilities,
- context limits configured by this platform,
- shard map,
- integrity hashes,
- safety and compatibility notes.

Private pools may use custom models.

### 2.3 Harness is user-selectable

A user account has a default `harness_id`.

Every session may override it.

Harnesses must implement a common adapter contract. The inference network does not care which harness is active. The harness talks to the agent gateway, which talks to inference and tools.

Initial adapter candidates:
- native MeshCompute agent,
- OpenCode,
- OpenHands,
- Aider,
- Kilo Code or compatible Cline-family adapter,
- generic ACP agent adapter.

Do not tightly couple user/session storage to any one harness.

### 2.4 Tools stay out of untrusted GPU workers

Credentials, shell access, Git credentials, browser cookies, API keys, local files, and MCP secrets must never be shipped to arbitrary public GPU workers.

The agent gateway executes tools in the user's trusted environment or in a controlled platform sandbox.

Inference workers receive only the model data necessary for the inference protocol.

### 2.5 API-first

Provide:
- OpenAI-compatible model listing,
- OpenAI-compatible chat completions,
- OpenAI-compatible streaming,
- OpenAI-compatible responses-style endpoint where practical,
- native MeshCompute session endpoints,
- WebSocket or SSE event streaming for the first-party clients,
- MCP client/server integration,
- ACP bridge for editor-facing agents.

Clients are replaceable views over these APIs.

## 3. Reality constraint: WAN inference physics

Do not assume that more internet-connected GPUs always increase decode speed.

Autoregressive decoding is sensitive to synchronization and round-trip latency. A naive layer-by-layer ring across high-latency WAN peers can be much slower than one capable GPU.

Therefore, MeshCompute must be topology-aware.

The scheduler's objective is:

> maximize expected useful throughput and minimize user-visible latency, subject to model-fit, trust, credit, reliability, and policy constraints.

It may choose multiple peers because:
- no single node fits the model,
- a tightly connected pod provides faster tensor/pipeline parallel execution,
- MoE expert placement benefits from distribution,
- speculative decoding can use weaker peers,
- prefill and decode can be disaggregated profitably,
- multiple replicas are needed for concurrency.

It may choose fewer peers when additional WAN hops would make the request slower.

This still satisfies the product requirement because the network is built to split work from day one. It does not artificially force a slower split when a better plan exists.

## 4. Execution strategies

Implement strategy interfaces rather than one hardcoded execution path.

### 4.1 Pipeline parallel

Split transformer blocks across peers.

Best first true-P2P strategy because it is conceptually simple and proves layer ownership, activation transport, KV handling, peer failure behavior, and scheduling.

Requirements:
- weighted layer assignment,
- activation serialization,
- session affinity,
- KV-cache ownership,
- micro-batching where useful,
- direct peer handoff,
- timeout and retry rules,
- topology-aware placement.

### 4.2 Tensor parallel

Use when peers have sufficiently low-latency, high-bandwidth connectivity.

For a trusted local pod, use an optimized backend such as NCCL where supported.

Do not stretch NCCL blindly over arbitrary consumer WAN links.

### 4.3 Expert parallel

For compatible Mixture-of-Experts models, allow experts to be distributed and routed dynamically.

This is a later optimization, but model manifests and scheduler interfaces must leave room for it.

### 4.4 Speculative execution

Support a role where smaller or weaker peers run draft models and stronger peers verify.

This is especially useful for heterogeneous or mobile contributors.

### 4.5 Replica parallelism

Run complete or partial replicas to serve concurrent sessions.

Replica parallelism is useful but must never be confused with the project's true distributed inference requirement.

### 4.6 Hybrid plans

The eventual scheduler can combine:
- tensor parallel inside a low-latency pod,
- pipeline parallel between pods,
- speculative draft peers,
- replicas for concurrency.

## 5. Recommended repository layout

```text
/
├── apps/
│   ├── control-plane/        # REST/API, auth, peer registry, credits, catalog
│   ├── agent-gateway/        # sessions, harnesses, tools, MCP, web, approvals
│   ├── web/                  # first-party web UI
│   └── cli/                  # user and operator CLI
├── node/
│   ├── daemon/               # P2P node daemon, preferably Rust
│   ├── runtime/              # Python inference runtime and backend adapters
│   ├── backends/
│   │   ├── torch/
│   │   ├── vllm/
│   │   ├── sglang/
│   │   └── future/
│   └── platform/
│       ├── nvidia/
│       ├── amd/
│       ├── apple/
│       └── mobile/
├── packages/
│   ├── protocol/
│   ├── sdk-python/
│   ├── sdk-typescript/
│   └── shared/
├── proto/
│   ├── control.proto
│   ├── inference.proto
│   └── peer.proto
├── models/
│   ├── catalog/
│   └── manifests/
├── infra/
│   ├── docker/
│   ├── compose/
│   └── migrations/
├── tests/
│   ├── integration/
│   ├── chaos/
│   ├── benchmark/
│   └── protocol/
└── docs/
```

## 6. Language and stack guidance

Prefer clean interfaces over stack purity.

Suggested starting stack:

### Node daemon
- Rust
- Tokio async runtime
- QUIC transport
- libp2p-style peer identity, relay, discovery, and hole punching patterns
- Ed25519 node identities
- BLAKE3 content hashes

### Inference runtime
- Python 3.12+
- PyTorch
- backend adapters for SGLang and vLLM where their execution model fits
- CUDA first for the initial performance target
- safetensors or equivalent non-executable weight format
- no arbitrary `trust_remote_code` on public pools

### Control plane
- FastAPI or another lightweight typed service
- PostgreSQL
- Redis only where it materially improves ephemeral scheduling/queues
- migrations from the first schema version
- OpenTelemetry

### Web client
- TypeScript
- React
- minimal dependency surface
- streaming responses
- node contribution dashboard
- model/harness selector
- credit/usage view

Do not block the POC on polishing the web UI.

## 7. Node capability advertisement

A worker must publish a signed capability record containing measured and static fields.

Example conceptual payload:

```json
{
  "node_id": "ed25519-public-key-derived-id",
  "runtime_version": "0.1.0",
  "os": "linux",
  "arch": "x86_64",
  "gpus": [
    {
      "vendor": "nvidia",
      "model": "example",
      "vram_bytes": 0,
      "free_vram_bytes": 0,
      "compute_capability": "example",
      "backend_support": ["cuda", "torch"]
    }
  ],
  "ram_free_bytes": 0,
  "storage_share_bytes": 0,
  "max_bandwidth_mbps": 0,
  "contribution_policy": {
    "idle_only": true,
    "max_gpu_percent": 90,
    "max_vram_percent": 85,
    "max_cpu_percent": 50,
    "allow_public_pool": true
  }
}
```

Never trust performance claims without benchmark verification.

## 8. Topology graph and scheduler

Maintain a graph:

- vertices = workers or local multi-GPU pods,
- edges = measured RTT, throughput, packet loss, NAT path type, relay requirement, and historical stability.

Update edge metrics continuously at a low overhead.

For each request:

1. Resolve public/private pool.
2. Resolve model manifest.
3. Resolve user priority and available credit.
4. Find workers with compatible shards/backends.
5. Estimate cold-load and shard-download cost.
6. Generate candidate execution plans.
7. Predict TTFT and decode throughput.
8. Apply reliability and trust penalties.
9. Select plan.
10. Reserve resources.
11. Establish direct session path.
12. Stream inference.
13. Record verified work and outcome.

The scheduler must log why a plan was selected so benchmarking can improve it later.

## 9. P2P networking

Required features:
- stable peer identity,
- encrypted QUIC,
- LAN discovery,
- DHT or rendezvous discovery,
- NAT traversal,
- hole punching where supported,
- relay fallback,
- peer-to-peer bandwidth measurement,
- stream multiplexing,
- cancellation,
- backpressure,
- reconnect semantics,
- session path versioning.

Separate transport concerns from inference semantics.

Define a transport abstraction so a local pod can later use:
- QUIC,
- NCCL,
- RDMA,
- shared memory,
- Thunderbolt or other high-speed fabric.

## 10. BitTorrent-like model distribution

Model distribution and inference are separate protocols.

Use content-addressed chunks.

Suggested model artifact design:
- signed manifest,
- Merkle-rooted chunk graph,
- BLAKE3 chunk hashes,
- configurable chunk size,
- resume support,
- peer rarity awareness,
- local cache,
- LRU eviction,
- prefetch for likely assignments,
- optional HTTP origin fallback.

A peer should download only the shards it needs plus configured redundancy.

Never execute code from a public model artifact.

## 11. Credit and priority system

Use an internal ledger, not a blockchain.

Terminology:
- `earned_compute`: verified contribution
- `spent_compute`: verified consumption
- `available_credit`: balance
- `contribution_score`: recent useful contribution and reliability
- `priority_score`: scheduling value during congestion

A conceptual earning formula:

```text
earned =
    verified_work_units
  * demand_multiplier
  * reliability_multiplier
  * scarcity_multiplier
```

A work unit should be based on measured useful work, not model name or claimed hardware.

Candidate measurement inputs:
- GPU-active milliseconds,
- completed layer/expert work,
- normalized FLOP estimate,
- bytes moved for accepted model distribution,
- draft tokens accepted during speculative decoding,
- storage availability where the network actually needs replicas.

Do not over-reward useless network traffic or idle uptime.

### Congestion behavior

During low load:
- admit most valid requests,
- use idle capacity generously.

During high load:
- paid or earned balance matters,
- recent contributors receive priority,
- users with lower contribution are throttled or queued rather than hard-blocked where possible.

## 12. Anti-cheat and node reputation

Assume some public workers are faulty or dishonest.

Implement:
- signed work receipts,
- benchmark challenges,
- randomized verification tasks,
- sampled redundant execution for high-risk nodes,
- result consistency checks,
- reputation decay,
- failure penalties,
- impossible-performance detection,
- node software version attestation where practical.

Do not rely on self-reported tokens/sec.

## 13. Agent gateway

The agent gateway is the trusted orchestration layer.

Responsibilities:
- user/session state,
- harness selection,
- model selection,
- tool registry,
- MCP connectivity,
- web search,
- browser/fetch,
- shell sandbox,
- file tools,
- Git tools,
- approval gates,
- context construction,
- memory hooks,
- citations,
- inference calls,
- usage accounting.

A harness adapter receives a stable interface:

```text
HarnessAdapter
  id()
  capabilities()
  create_session(config)
  resume_session(session_id)
  handle_user_message(message, context)
  handle_tool_result(result)
  export_state()
  import_state(optional)
  cancel()
```

Cross-harness session switching is a state migration event, not an assumption that every internal state object is portable.

## 14. Web search and browser tools

Web capability is required for the full product.

Create provider interfaces:

```text
SearchProvider
  search(query, options) -> SearchResult[]

FetchProvider
  fetch(url, options) -> Document

BrowserProvider
  open(...)
  click(...)
  type(...)
  screenshot(...)
  extract(...)
```

Support a low-cost/self-host path such as SearXNG where appropriate, plus optional commercial providers.

Keep search separate from fetch.

Use a sandboxed browser runtime for interactive pages.

Treat all web content as untrusted input. Prompt injection defense is required at the tool boundary.

The model must not receive browser cookies or platform credentials unless a specific trusted tool intentionally exposes scoped data.

## 15. MCP and ACP

Target MCP specification `2026-07-28` or newer after compatibility testing.

Important 2026 behavior:
- MCP core is stateless at the protocol layer.
- Tool/session state remains an application concern.
- Build authorization and routing around the current specification rather than obsolete session assumptions.

Use ACP as an editor/agent interoperability bridge so the platform is not tied to one IDE.

## 16. Public model target

Start with a single model that can validate:
- multi-node inference,
- coding,
- tool calls,
- long context at a controlled configured size.

A strong first benchmark target is the Qwen3-Coder 30B class, specifically a version-pinned Qwen3-Coder-30B-A3B-Instruct manifest if its license and runtime behavior remain suitable at implementation time.

Do not hardcode the system around Qwen.

The model catalog must support later replacement without changing the network protocol.

Qwen's official documentation notes that Qwen3-Coder function calling depends on the appropriate tool parser in current SGLang/vLLM versions. Pin and test that exact parser behavior in the model manifest.

## 17. Public vs private pools

### Public pool

- signed approved models only,
- platform runtime only,
- no arbitrary remote code,
- public worker reputation,
- compute credit accounting,
- clearly disclosed public-swarm privacy model.

### Private pool

- invite or organization controlled,
- custom model manifests,
- optional custom quantizations,
- custom trust policy,
- custom tool gateway,
- ability to restrict peers by identity,
- optional private accounting policy.

## 18. Privacy

TLS/QUIC encryption protects data in transit but does not make an untrusted inference host blind to the data it processes.

Do not market the public swarm as end-to-end confidential inference.

Phase-1 policy:
- clearly label public-swarm privacy,
- recommend private pools for sensitive code/data,
- keep user secrets out of workers,
- minimize raw prompt exposure where protocol design permits,
- avoid logging prompt bodies by default.

Future research:
- TEEs,
- confidential GPU features,
- privacy-preserving partitioning,
- encrypted or secret-shared inference where practical.

## 19. Session and memory foundation

Full cross-device memory is later, but prepare now.

Use event-sourced session storage.

Every session event includes:
- event ID,
- user ID,
- workspace ID,
- session ID,
- parent event ID,
- timestamp,
- client/device ID,
- harness ID,
- model ID,
- pool ID,
- content type,
- payload reference,
- tool-call linkage,
- schema version.

Do not make memory depend on one device's local filesystem.

Phase 2 can add:
- synchronized conversation history,
- project memory,
- embeddings/retrieval,
- user-controlled retention,
- client-side encryption options,
- conflict resolution.

## 20. Mobile roadmap constraints

### Android

Design the protocol so Android can later contribute useful workloads.

Do not build new work around NNAPI. Android deprecated NNAPI in Android 15 and currently directs developers toward TensorFlow Lite runtimes and GPU acceleration paths.

Potential Android contribution roles:
- small draft model,
- embedding generation,
- reranking,
- verification,
- model chunk seeding,
- lightweight transformer shards on capable devices,
- foreground opt-in GPU work.

Prefer charging + Wi-Fi + thermal/battery gates.

A future native backend may use Vulkan, TFLite GPU paths, or another maintained mobile inference runtime.

### iOS

Treat iOS as opportunistic.

Apple supports background processing and, on supported configurations, background GPU or inference entitlements for continued processing tasks. These jobs remain OS-controlled and should not be treated as guaranteed always-on workers.

Initial iOS priority:
- client access,
- session sync,
- notifications,
- optional foreground contribution.

## 21. User clients

### CLI first

Example goals:

```bash
mesh login
mesh chat --model public/qwen3-coder --harness opencode
mesh node start --idle-only --max-vram 80%
mesh node status
mesh pool create private my-studio
mesh api serve --port 8080
```

### Web

Required eventual views:
- chat,
- harness selector,
- model selector,
- public/private pool selector,
- connected nodes,
- per-node utilization,
- contributions,
- credits,
- current execution path,
- network health,
- settings.

### Mobile

Phase 2/3:
- chat and agent sessions,
- approvals,
- job status,
- notifications,
- session handoff,
- optional contribution mode.

## 22. Observability

Instrument from the first POC.

Per inference collect:
- plan ID,
- node path,
- model/shards,
- TTFT,
- prefill tok/s,
- decode tok/s,
- total latency,
- activation bytes,
- model-shard bytes,
- peer RTT,
- peer throughput,
- retransmits,
- queue delay,
- cold-load delay,
- cache hits,
- failures,
- reroutes,
- credits earned/spent.

Use trace IDs across agent gateway, scheduler, and peer sessions.

## 23. Tests that must exist

### Protocol
- serialization compatibility,
- cancellation,
- reconnect,
- chunk verification,
- malformed message rejection.

### Multi-node inference
- 2-node split,
- 3-node split,
- heterogeneous VRAM,
- heterogeneous compute,
- peer dropout,
- high latency,
- low bandwidth,
- relay path,
- direct path.

### Scheduler
- chooses a faster topology,
- rejects an infeasible topology,
- accounts for model load time,
- avoids unreliable nodes,
- honors pool restrictions,
- honors user priority.

### Security
- untrusted manifest,
- hash mismatch,
- tool credential isolation,
- malicious worker messages,
- replayed work receipt,
- privilege boundary tests.

### Agent
- tool call,
- web search,
- harness selection,
- session resume,
- tool approval,
- streaming.

## 24. Benchmark matrix

Use reproducible benchmark profiles.

At minimum test:
- same-LAN low-latency peers,
- 5 ms RTT,
- 20 ms RTT,
- 50 ms RTT,
- 100 ms RTT,
- 1 Gbps,
- 250 Mbps,
- 100 Mbps,
- packet loss simulation,
- cold vs warm model shards.

Use Linux traffic shaping in integration tests when appropriate.

Do not report only tokens/sec. Always include TTFT and the topology.

## 25. Milestone order

### Milestone 0 - skeleton
- monorepo,
- schemas,
- CI,
- node identity,
- control-plane registration,
- telemetry.

### Milestone 1 - P2P fabric
- peer discovery,
- QUIC,
- NAT traversal,
- relay fallback,
- bandwidth/RTT measurement,
- encrypted direct streams.

### Milestone 2 - model swarm
- signed manifests,
- content-addressed chunks,
- peer distribution,
- cache management.

### Milestone 3 - true distributed inference POC
- one transformer family,
- pipeline partitioning,
- direct activation handoff,
- distributed KV ownership,
- streaming output,
- 2+ peers.

### Milestone 4 - topology scheduler
- measured graph,
- candidate plan scoring,
- failure recovery,
- performance prediction.

### Milestone 5 - public API
- auth,
- OpenAI-compatible endpoints,
- quota/credit hooks,
- streaming.

### Milestone 6 - agent gateway
- native harness,
- harness adapter API,
- MCP,
- web search/fetch,
- sandboxed tools.

### Milestone 7 - contribution economics
- work receipts,
- credit ledger,
- priority,
- reputation,
- anti-cheat.

### Milestone 8 - private pools
- invite control,
- custom model manifests,
- private trust policy.

### Milestone 9 - optimized parallelism
- tensor parallel on suitable links,
- speculative roles,
- MoE/expert strategy where supported,
- disaggregated execution experiments.

### Milestone 10 - memory and multi-device sessions
- event sync,
- durable project memory,
- retention controls.

### Milestone 11 - mobile
- Android/iOS clients,
- Android contribution experiments,
- iOS opportunistic contribution research.

## 26. Coding-agent rules

When an AI coding agent works on this repository:

1. Read `INSTRUCTIONS.md`, `ARCHITECTURE.md`, `PROTOCOL.md`, `SECURITY.md`, and `ROADMAP.md` before architectural work.
2. Do not replace true distributed inference with request routing.
3. Do not introduce a centralized activation relay as the default data path.
4. Do not execute arbitrary model code from public peers.
5. Do not place tool credentials on public inference workers.
6. Do not add blockchain unless explicitly requested.
7. Do not make performance claims without benchmarks.
8. Every new protocol field must be versioned.
9. Every new backend must advertise capabilities rather than rely on device-name assumptions.
10. Every feature touching credits must be auditable.
11. Every feature touching external content must consider prompt injection.
12. Every network path must have cancellation and timeout behavior.
13. Prefer resumable state and idempotent control-plane operations.
14. Keep model/harness/tool providers pluggable.
15. Add or update tests with every behavior change.
16. Keep a changelog entry for architecture-affecting changes.
17. Never silently weaken a security boundary to make a demo pass.

## 27. Immediate next action for the coding agent

Do not attempt to implement the entire product in one pass.

Create Milestones 0 through 3 as the first working branch sequence.

The first compelling demo should be:

1. Start two worker daemons on two machines.
2. Discover/connect them directly.
3. Download or assign different portions of the same approved model.
4. Route one prompt through both workers.
5. Stream tokens from the combined model.
6. Display the active peer path and performance metrics.
7. Disconnect one worker and show controlled failure/recovery behavior.
8. Repeat with artificial network latency and compare scheduling decisions.

Only after this works should the repository spend significant effort on UI polish, credits, mobile clients, or memory.
