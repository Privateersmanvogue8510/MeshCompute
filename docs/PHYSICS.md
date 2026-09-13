# Scaling & Inference Physics — how "more people = faster AI" is real

The product idea: **the more people on the network, the faster the AI runs.** This
document explains exactly how that is achieved, backed by the real numbers measured
on this network, and treats the one genuinely hard case as a research frontier
rather than a wall.

There are several independent ways adding nodes makes the network faster. The
scheduler uses all of them and picks per request. They are listed here strongest-
first for the "more people = faster" goal.

## 1. Replica scale-out + smart load balancing (the primary mechanism)

Every node that can run the model is an interchangeable **replica**. When a request
arrives, the scheduler routes it to the *best replica for that request*: fastest
GPU, closest (lowest RTT), and least loaded. This means:

- **More replicas → your request waits less.** With one node, requests queue behind
  each other. With ten, there is almost always a free node, so your request starts
  immediately. Queue wait is modeled directly in the scheduler score.
- **More replicas → your request routes to a closer/faster node.** A user in Asia
  hits an Asian replica; a user in the US hits a US replica. Adding a node near you
  makes *your* requests faster even though no single request is "split."
- **More replicas → higher total throughput.** N replicas serve ~N concurrent
  sessions. The network's aggregate tokens/sec scales linearly with node count.

This is load balancing + horizontal scale-out, and it is the honest core of "more
people = faster." The scheduler's decision trace reports the replica pool size and
the routing choice on every request. Verified: with three replicas and the nearest
one busy, the scheduler routes to the idle nearby replica, not the busy fast one.

## 2. Speculative decoding — extra peers speed a single stream

A weaker/nearer peer runs a small **draft model** and proposes several tokens; a
stronger peer verifies them in one pass. Accepted drafts skip full compute, so a
single request's tokens/sec rises when you add draft-capable peers. Tolerant of
latency because drafts are cheap and batched. (Backend support: Phase-2; the
manifest and scheduler roles already reserve `speculative`.)

## 3. Prefill parallelism — extra peers speed time-to-first-token

Prompt prefill is compute-bound and parallelizes across the sequence. A long prompt
can be sharded across several nearby nodes so TTFT drops as you add nodes. Decode
stays local (it is latency-bound). This is prefill/decode disaggregation.

## 4. Tensor / pipeline parallel inside a low-latency pod — extra GPUs speed one request

Two or more machines on a fast link (same LAN, same rack, same cloud region;
sub-millisecond, high bandwidth) behave like one big machine. Splitting a single
model across *that* is a genuine per-request speedup and lets the network run models
too big for any one node. The scheduler admits this only when the link is LAN-class
(`is_lan_class()`: ≤5 ms, ≥500 Mbps, direct). Peers discovered close together by
rendezvous + NAT hole-punching can form such a pod on demand — no VPN, no tailnet.

## 5. Model distribution (the BitTorrent part) — more seeders = faster onboarding

Content-addressed GGUF chunks pulled in parallel from many seeders get 20+ GB of
weights onto a new worker fast, each chunk verified by BLAKE3. More seeders → a new
node joins and starts serving sooner → capacity grows faster. See
`node/runtime/meshcompute_runtime/swarm.py`.

---

## The one hard case: splitting a single token's decode across a high-latency WAN

This is the case that does NOT get faster with more nodes, and it is worth being
precise about why, because it is the trap that sinks naive designs.

### The measured reality

| Link | RTT | Notes |
|---|---|---|
| consumer → GPU pod (cross-Pacific WAN) | **~255 ms** | ping mean 255 ms, mdev 14 ms |
| between the two GPUs inside the pod | **< 0.1 ms** | PCIe / same host |
| consumer → local CPU box (same LAN) | **~1 ms** | but no usable GPU |

The 2x3090 pod decodes this 27B model at ~90 tok/s.

### Why per-token WAN splitting is latency-bound

Autoregressive decode is sequential: token N+1 needs token N. If layers are split
across two machines, **every decode step** crosses the link between them — one round
trip per token, minimum. At 255 ms/RTT that caps decode near 4 tok/s regardless of
GPU speed. The scheduler scored the candidates for this exact topology:

```
selected single (score 3.19s: ttft 0.63s, decode 90 tok/s)   <- run on the pod, stream back
rejected single chain=[hk_cpu]:            score 46.9s  (decode 6 tok/s)
rejected pipeline chain=[vegas, hk_cpu]:   score 136.6s (decode 2 tok/s)   <- WAN layer split
```

Running the whole model on the pod and streaming tokens over the WAN is ~39x faster
than the WAN layer split, because streaming pays the 255 ms **once** (at first token)
instead of **once per token**. Same hardware, same link — the difference is entirely
where the latency lands in the loop.

### This is a research frontier, not an impossibility

"You can't make a single token's decode cross 255 ms faster" is a statement about
one naive strategy, not a dead end. Real directions to make even the WAN-split
regime scale — tracked as research, not dismissed:

- **Latency hiding via concurrency.** Keep the cross-WAN pipeline full with many
  concurrent requests (micro-batching / continuous batching). Per-request latency
  stays WAN-bound, but aggregate throughput scales with nodes even across the WAN —
  which is exactly "more people = faster" at the system level.
- **Communication/computation overlap.** Prefetch and overlap activation transfer
  with compute so the RTT is partly hidden behind useful work.
- **Sequence/context parallelism** and **speculative cross-WAN drafting**, which
  batch or amortize the round trips.
- **Better placement.** Rendezvous should prefer forming pods among peers that are
  actually close, using the WAN only for replica routing and distribution.

The scheduler is built to *measure and choose*, and its cost constants are meant to
be calibrated against real benchmark error over time (ARCHITECTURE.md), so as these
techniques land they change the plan the scheduler picks — it does not need new
hardcoded rules.

## Reconciling with AGENTS.md

The repo's AGENTS.md says "do not reduce it to ordinary load balancing across
independent model servers." That guards against a system that can *only* fan out
requests and can never split a model it doesn't fit. MeshCompute keeps the true
multi-peer split capability (needed for oversized models and fast pods) **and** does
first-class replica load-balancing/scale-out (the "more people = faster" experience).
Both are real; the scheduler picks per request. The project owner's live direction —
prioritize the scale-out experience — is what sets the default.
