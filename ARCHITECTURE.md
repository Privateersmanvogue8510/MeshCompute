# MeshCompute Architecture

## High-level system

```mermaid
flowchart LR
    U[User Client<br/>CLI / Web / IDE / Mobile] --> AG[Agent Gateway]
    AG --> H[Harness Adapter]
    AG --> T[Trusted Tool Fabric<br/>MCP / Web / Browser / Shell / Files / Git]
    AG --> API[Inference API / Router]

    API --> CP[Control Plane<br/>Auth / Catalog / Scheduler / Credits / Reputation]
    CP -. rendezvous + plan .-> N1[Peer A]
    CP -. rendezvous + plan .-> N2[Peer B]
    CP -. rendezvous + plan .-> N3[Peer C]

    N1 <== direct encrypted data plane ==> N2
    N2 <== direct encrypted data plane ==> N3
    N3 <== direct encrypted data plane ==> N1

    N1 <--> SWARM[Content-addressed Model Swarm]
    N2 <--> SWARM
    N3 <--> SWARM
```

The dashed control-plane links carry metadata and planning.

The peer links carry the inference data path.

## Trust zones

```mermaid
flowchart TB
    subgraph TrustedUserZone[Trusted user or platform agent zone]
        A[Agent Gateway]
        B[Tool Credentials]
        C[Browser / Shell / Files / Git]
        D[Memory / Session Store]
    end

    subgraph PublicControl[Public control plane]
        E[Identity]
        F[Scheduler]
        G[Credits]
        H[Model Catalog]
    end

    subgraph UntrustedWorkers[Potentially untrusted inference workers]
        W1[Worker 1]
        W2[Worker 2]
        W3[Worker 3]
    end

    A --> E
    A --> F
    F --> W1
    F --> W2
    F --> W3
    W1 <--> W2
    W2 <--> W3
    B --> C
    C --> A
    D --> A
```

Secrets remain in the trusted agent zone.

## Request lifecycle

```mermaid
sequenceDiagram
    participant Client
    participant Agent as Agent Gateway
    participant Scheduler
    participant A as Peer A
    participant B as Peer B
    participant C as Peer C

    Client->>Agent: user message
    Agent->>Scheduler: model, pool, priority, requirements
    Scheduler->>Scheduler: score candidate topologies
    Scheduler-->>Agent: execution plan + peer tickets
    Agent->>A: open session
    A->>B: direct QUIC inference stream
    B->>C: direct QUIC inference stream
    C-->>A: sampled token/logit path as strategy requires
    A-->>Agent: streamed generation
    Agent-->>Client: tokens + tool events
    Agent->>Scheduler: usage receipt
    A->>Scheduler: signed work receipt
    B->>Scheduler: signed work receipt
    C->>Scheduler: signed work receipt
```

## Core services

### Control plane

Stateless API where possible, durable state in PostgreSQL.

Owns:
- accounts,
- node identities,
- model manifests,
- pool membership,
- scheduling metadata,
- credits,
- reputation,
- signed execution plans,
- signed work receipts,
- coarse telemetry.

It should be horizontally scalable and must not carry the bulk activation stream.

### Scheduler

Consumes:
- node capabilities,
- peer topology,
- shard availability,
- queue state,
- model strategy support,
- user credit/priority,
- pool policy.

Outputs a versioned execution plan.

The execution plan must be reproducible enough to explain why a request was placed on specific peers.

### Node daemon

Responsibilities:
- peer identity,
- discovery,
- NAT traversal,
- QUIC streams,
- relay fallback,
- telemetry,
- resource policy,
- model cache,
- shard distribution,
- runtime process management,
- signed work receipts.

The node daemon should not contain agent credentials.

### Inference runtime

Separate process boundary from the node daemon.

Responsibilities:
- load model shard,
- run assigned blocks/experts/tensors,
- own relevant KV cache,
- serialize/deserialize activation packets,
- enforce memory limits,
- expose measured performance,
- clean up on cancellation.

### Agent gateway

Responsibilities:
- session orchestration,
- harness adapters,
- prompt/context,
- tools,
- approvals,
- web,
- MCP,
- ACP,
- memory,
- citations,
- calls to inference API.

## Scheduling cost model

A candidate execution plan should estimate:

```text
predicted_total_latency =
    queue_delay
  + shard_load_delay
  + prefill_compute
  + prefill_network
  + decode_compute
  + decode_network
  + reliability_penalty
```

Use real measurements and calibrate prediction error over time.

A path with more GPUs can lose if every token crosses a 70 ms WAN hop.

The system should learn this rather than encode simplistic rules.

## Failure model

Peers can:
- disappear,
- throttle,
- lie,
- return invalid data,
- run out of VRAM,
- suffer thermal throttling,
- change network,
- lose relay connectivity.

Execution plans must specify failure behavior:
- retry same peer,
- bypass peer,
- rebuild path,
- restart generation from checkpoint,
- fail session cleanly.

Phase 1 may restart generation if mid-token-path recovery is too complex. Do not fake seamless recovery.

## Model placement

Each peer advertises cached content hashes.

The scheduler prefers:
1. warm compatible shards,
2. high-reliability peers,
3. topology-compatible peers,
4. peers likely to remain available.

Cold-loading a model can dominate interactive latency, so shard placement is part of scheduling.

## Data-plane packet classes

Keep protocol classes distinct:
- control signal,
- activation tensor,
- KV-cache operation,
- token/logit result,
- model chunk,
- telemetry,
- cancellation,
- work receipt.

Large tensor payloads should not be serialized through JSON.

Use versioned binary framing.
