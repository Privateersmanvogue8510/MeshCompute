# MeshCompute Roadmap

## Phase 0 - Research spike

Goal: validate the physics before building a large platform.

Deliverables:
- benchmark existing distributed inference approaches,
- measure pipeline partitioning across LAN and WAN,
- measure activation bandwidth for target model families,
- test NAT traversal and QUIC,
- establish baseline single-node numbers.

Exit criterion:
- a written benchmark showing where multi-peer execution helps and where it hurts.

## Phase 1 - Core P2P distributed inference POC

Goal: prove the product's defining capability.

Deliverables:
- node identity,
- discovery/rendezvous,
- direct QUIC,
- relay fallback,
- capability advertisement,
- signed model manifest,
- content-addressed shard cache,
- peer model transfer,
- pipeline model partition,
- distributed KV ownership,
- 2+ peer generation,
- streaming API,
- metrics.

Exit criterion:
- one model inference is demonstrably split across multiple independent machines.

## Phase 1.5 - Topology-aware scheduler

Goal: stop treating all peers as equal.

Deliverables:
- topology graph,
- peer benchmarks,
- bandwidth/RTT measurement,
- execution-plan scoring,
- warm-shard awareness,
- dropout behavior,
- scheduler decision traces.

Exit criterion:
- scheduler picks different plans under different simulated network conditions for defensible performance reasons.

## Phase 2 - Public platform

Goal: make the swarm usable by other people.

Deliverables:
- account/auth,
- curated public catalog,
- public pools,
- node installer,
- contribution controls,
- basic dashboard,
- OpenAI-compatible API,
- credit ledger,
- node reputation,
- work receipts.

Exit criterion:
- an external user can contribute a node and consume inference with earned credit.

## Phase 2.5 - Agent platform

Goal: make the platform useful as an everyday AI system.

Deliverables:
- native agent harness,
- harness adapter API,
- per-user and per-session harness selection,
- MCP,
- ACP,
- web search,
- fetch,
- sandboxed browser,
- files/shell/Git tools,
- approvals,
- citations.

Exit criterion:
- user can perform a coding/research task through a selected harness using distributed inference and tools.

## Phase 3 - Private pools and advanced parallelism

Deliverables:
- custom private model manifests,
- invited peers,
- private policy,
- tensor parallel for suitable topology,
- speculative decoding roles,
- MoE expert distribution experiments,
- disaggregated prefill/decode experiments,
- smarter replication.

Exit criterion:
- platform can choose among multiple parallelism strategies based on measured topology.

## Phase 4 - Cross-device memory and session continuity

Deliverables:
- event-sourced sync,
- web/desktop session handoff,
- workspace memory,
- retrieval,
- retention controls,
- export/delete,
- optional encryption features.

Exit criterion:
- a session can move between two clients without losing project context.

## Phase 5 - Mobile clients

Deliverables:
- Android app,
- iOS app,
- chat,
- approvals,
- notifications,
- session handoff,
- node/credit monitoring.

Contribution experiments:
- Android draft model,
- Android embeddings/reranking,
- Android model-chunk seeding,
- iOS foreground or eligible continued-processing workloads.

Exit criterion:
- mobile is a first-class client, with contribution enabled only where performance and OS policy make sense.

## Phase 6 - Federation and hardening

Possible later directions:
- multiple control-plane operators,
- organization federation,
- signed cross-network accounting,
- advanced privacy,
- trusted execution,
- more decentralized rendezvous,
- economic marketplace features.

Do not start here. Prove the inference fabric first.
