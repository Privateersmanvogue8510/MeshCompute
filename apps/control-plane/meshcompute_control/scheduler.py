"""Topology-aware scheduler — the core thesis of MeshCompute (INSTRUCTIONS §3, §8).

    maximize expected useful throughput, minimize user-visible latency, subject to
    model-fit, trust, credit, reliability and policy constraints.

The scheduler enumerates candidate execution plans, predicts TTFT + decode
throughput for each using a physical cost model over the measured topology, applies
reliability/trust penalties, and picks the best. Critically it will choose FEWER
peers when extra WAN hops would make the request slower — it must *learn/measure*
this rather than blindly maximise node count.

Why this matters for our first target: the powerhouse (2x3090) sits behind a ~255ms
WAN link from the consumer that wants to use it. A layer-by-layer pipeline split
across that link costs multiple RTTs per token. The scheduler must prove it picks
the single low-latency pod and serves it over the WAN, and explain why.

The cost model is intentionally simple and CALIBRATED from measurement, not
hardcoded rules (ARCHITECTURE.md: "The system should learn this rather than encode
simplistic rules"). Calibration constants live in one place and are overridable.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from meshcompute_protocol import (
    CapabilityRecord,
    ExecutionPlan,
    ModelManifest,
    ShardAssignment,
    Strategy,
)

from .topology import LinkMetrics, TopologyGraph


# ---- calibration constants (measured/tunable, not magic) --------------------
@dataclass
class CostModel:
    # per-token pipeline sync cost: crossing a peer boundary costs ~1 RTT of the
    # slowest edge in the chain, per decode step (activations hand off + return).
    rtt_multiplier_per_hop: float = 1.0
    # activation payload per token per hop (bytes). qwen35 hidden ~5k * 2 bytes (bf16)
    # ~= 10 KB order of magnitude; used to add bandwidth term.
    activation_bytes_per_token: int = 12_000
    # fraction of a node's benchmarked decode tps retained when its layers are a
    # slice of the model (pipeline underutilises each node between handoffs).
    pipeline_efficiency: float = 0.65
    # reliability penalty weight (seconds added per expected-restart probability).
    reliability_penalty_s: float = 8.0
    # default per-node decode tps if no benchmark present (pessimistic).
    default_decode_tps: float = 20.0
    default_prefill_tps: float = 400.0
    # --- "more (close) nodes = more single-request speed" ---
    # tensor-parallel across a low-latency pod scales a single request's decode with
    # node count, minus interconnect overhead. NVLINK/PCIe/LAN only (is_lan_class).
    tensor_parallel_efficiency: float = 0.85   # per added node; 2 nodes ~= 1.7x
    # speculative decoding: a draft peer proposes tokens a strong peer verifies in
    # one pass -> single-stream speedup, tolerant of moderate latency.
    speculative_speedup: float = 1.6
    speculative_max_rtt_ms: float = 50.0
    # prefill can be sharded across close nodes -> lower TTFT with more nodes.
    prefill_parallel_efficiency: float = 0.8


@dataclass
class Candidate:
    strategy: Strategy
    peer_chain: list[str]
    predicted_ttft_s: float
    predicted_decode_tps: float
    score: float
    trace: list[str] = field(default_factory=list)


class Scheduler:
    def __init__(self, cost: CostModel | None = None) -> None:
        self.cost = cost or CostModel()

    # --- model fit -----------------------------------------------------------
    def _quant_bytes(self, manifest: ModelManifest) -> int:
        """Bytes of the catalog's FIRST-listed quantization — the one nodes
        actually install (runtime_installer.pick_quant), so model-fit is judged
        against the files a node really loads, not the smallest option."""
        if not manifest.quantizations:
            return 0
        return sum(f.size_bytes for f in manifest.quantizations[0].files)

    def _kv_allowance(self, model_bytes: int, ctx: int) -> int:
        """KV cache budget beside the weights. KV ~ ctx * layers * heads is
        model-specific; use a coarse 20% of weights per 32k ctx as the POC
        allowance. Single-node fit and split fit must use the SAME allowance."""
        return int(model_bytes * 0.20 * max(ctx, 1) / 32768)

    def _usable_bytes(self, node: CapabilityRecord) -> int:
        """Memory a node can actually load weights+KV into. CPU-only nodes
        advertise vram 0 but can run from RAM (slow), so fall back to RAM."""
        vram = node.free_vram_bytes() or node.total_vram_bytes()
        return vram if vram else node.ram_free_bytes

    def _fits_single(self, node: CapabilityRecord, model_bytes: int, ctx: int) -> bool:
        return self._usable_bytes(node) >= model_bytes + self._kv_allowance(model_bytes, ctx)

    def _can(self, node: CapabilityRecord, strategy: str) -> bool:
        """Does the node advertise that it can take part in this strategy?
        An empty list is an older/plain node: whole-model work only."""
        return strategy in (node.strategies or ["single", "replica"])

    # --- cost prediction -----------------------------------------------------
    def _node_decode_tps(self, node: CapabilityRecord) -> float:
        return node.benchmark.decode_tokens_per_sec or self.cost.default_decode_tps

    def _node_prefill_tps(self, node: CapabilityRecord) -> float:
        return node.benchmark.prefill_tokens_per_sec or self.cost.default_prefill_tps

    def _worst_link(self, graph: TopologyGraph, chain: list[str]) -> LinkMetrics:
        worst = LinkMetrics(rtt_ms=0.05, throughput_mbps=1e9, path_type="loopback")
        for i in range(len(chain) - 1):
            lk = graph.link(chain[i], chain[i + 1])
            if lk.rtt_ms > worst.rtt_ms:
                worst = lk
        return worst

    def _predict_single(self, node: CapabilityRecord, client_link: LinkMetrics,
                        prompt_tokens: int, gen_tokens: int) -> tuple[float, float, list[str]]:
        """One pod runs the whole model. Client<->pod link only adds to TTFT (streaming
        hides per-token WAN latency: tokens pipeline back, decode tps is pod-bound)."""
        prefill = prompt_tokens / self._node_prefill_tps(node)
        # TTFT = prefill + one client RTT (request in, first token out).
        ttft = prefill + (client_link.rtt_ms / 1000.0)
        decode_tps = self._node_decode_tps(node)
        trace = [
            f"single: pod {node.node_id} runs whole model; prefill≈{prefill:.2f}s, "
            f"client RTT {client_link.rtt_ms:.0f}ms adds to TTFT only; "
            f"decode {decode_tps:.0f} tok/s is pod-bound (streaming hides WAN)."]
        return ttft, decode_tps, trace

    def _predict_pipeline(self, nodes: list[CapabilityRecord], graph: TopologyGraph,
                         client_link: LinkMetrics, prompt_tokens: int,
                         gen_tokens: int) -> tuple[float, float, list[str]]:
        """Layers split across the chain. EACH decode step crosses every peer boundary,
        so per-token latency includes the inter-peer RTTs. This is where WAN kills it."""
        chain = [n.node_id for n in nodes]
        worst = self._worst_link(graph, chain)
        hops = len(chain) - 1
        # per-token added latency from crossing hops (round trip of activations).
        rtt_s = (worst.rtt_ms / 1000.0) * self.cost.rtt_multiplier_per_hop * hops
        # bandwidth term per token across the worst link.
        bw_s = (self.cost.activation_bytes_per_token * 8 / (worst.throughput_mbps * 1e6)) * hops
        per_token_net = rtt_s + bw_s
        # compute side: slowest node's slice, discounted for pipeline underutilisation.
        slice_tps = min(self._node_decode_tps(n) for n in nodes) * self.cost.pipeline_efficiency
        compute_per_token = 1.0 / slice_tps
        eff_per_token = compute_per_token + per_token_net
        decode_tps = 1.0 / eff_per_token if eff_per_token > 0 else 0.0
        prefill = prompt_tokens / min(self._node_prefill_tps(n) for n in nodes)
        ttft = prefill + per_token_net + (client_link.rtt_ms / 1000.0)
        trace = [
            f"pipeline: {hops} hop(s), worst link {worst.rtt_ms:.0f}ms/{worst.throughput_mbps:.0f}Mbps "
            f"({worst.path_type}); per-token network {per_token_net*1000:.0f}ms "
            f"(rtt {rtt_s*1000:.0f}ms + bw {bw_s*1000:.1f}ms); "
            f"slice compute {compute_per_token*1000:.0f}ms/tok; "
            f"=> decode {decode_tps:.1f} tok/s."]
        return ttft, decode_tps, trace

    def _mutually_lan_class(self, graph: TopologyGraph, chain: list[str]) -> bool:
        """True only if EVERY pair in the chain is on a low-latency, high-bandwidth,
        direct link — the precondition for a fast 'pod' that speeds a single request."""
        for i in range(len(chain)):
            for j in range(i + 1, len(chain)):
                if not graph.link(chain[i], chain[j]).is_lan_class():
                    return False
        return True

    def _predict_tensor(self, nodes: list[CapabilityRecord], graph: TopologyGraph,
                        client_link: LinkMetrics, prompt_tokens: int
                        ) -> tuple[float, float, list[str]]:
        """Tensor-parallel across a low-latency pod: a SINGLE request's decode scales
        with node count (each node computes a slice of every layer in lockstep over a
        fast link). This is how more (close) nodes make one request FASTER."""
        n = len(nodes)
        base = min(self._node_decode_tps(x) for x in nodes)
        # near-linear scaling on a fast interconnect, discounted by efficiency.
        decode_tps = base * (1 + (n - 1) * self.cost.tensor_parallel_efficiency)
        prefill = prompt_tokens / (sum(self._node_prefill_tps(x) for x in nodes)
                                   * self.cost.prefill_parallel_efficiency)
        ttft = prefill + (client_link.rtt_ms / 1000.0)
        trace = [f"tensor-pod: {n} nodes on a LAN-class link act as one unit; "
                 f"single-request decode {base:.0f}->{decode_tps:.0f} tok/s "
                 f"(x{decode_tps/base:.2f}); prefill sharded. More close nodes = faster."]
        return ttft, decode_tps, trace

    def _predict_speculative(self, strong: CapabilityRecord, draft: CapabilityRecord,
                            graph: TopologyGraph, client_link: LinkMetrics,
                            prompt_tokens: int) -> tuple[float, float, list[str]]:
        """Draft peer proposes tokens; strong peer verifies in one pass. Single-stream
        decode speeds up even over a moderate-latency link to the draft peer."""
        base = self._node_decode_tps(strong)
        decode_tps = base * self.cost.speculative_speedup
        prefill = prompt_tokens / self._node_prefill_tps(strong)
        ttft = prefill + (client_link.rtt_ms / 1000.0)
        trace = [f"speculative: draft {draft.node_id} + verifier {strong.node_id}; "
                 f"single-stream decode {base:.0f}->{decode_tps:.0f} tok/s "
                 f"(x{self.cost.speculative_speedup}). Extra peer accelerates one stream."]
        return ttft, decode_tps, trace

    def _score(self, ttft: float, decode_tps: float, nodes: list[CapabilityRecord],
               graph: TopologyGraph, chain: list[str], queue_wait_s: float = 0.0) -> float:
        """Lower is better. Total user-visible time = queue wait + ttft + generation +
        reliability penalty. queue_wait_s is how long this request would sit behind
        others already on the chosen node — this is what makes adding nodes help:
        more replicas -> a free/closer node exists -> less queue wait for you."""
        REPRESENTATIVE_GEN = 256
        total = queue_wait_s + ttft + (REPRESENTATIVE_GEN / decode_tps if decode_tps > 0 else 1e9)
        worst = self._worst_link(graph, chain)
        p_fail = 1.0 - worst.stability
        total += p_fail * self.cost.reliability_penalty_s
        return total

    # --- main entry ----------------------------------------------------------
    def plan(self, *, model: ModelManifest, nodes: dict[str, CapabilityRecord],
             graph: TopologyGraph, client_node_id: str, context_length: int,
             prompt_tokens: int = 512, gen_tokens: int = 256,
             load: dict[str, int] | None = None,
             executable_strategies: list[str] | None = None,
             plan_id: str = "plan_0") -> ExecutionPlan:
        """Enumerate candidates, predict, score, pick best, and record why.

        `executable_strategies` is what the CALLER can actually run (the Phase-1
        gateway drives one OpenAI-compatible backend, so it sends ["single"]).
        Empty/None = no restriction; otherwise strategies outside it are never
        proposed — a plan the caller cannot execute is worse than no plan.

        `load` maps node_id -> current queued/active sessions on that node. It is how
        "more people on the network => faster" becomes real: every node that can run
        the model is a REPLICA, and the scheduler routes each request to the best
        replica for THAT request (fastest + closest + least loaded). Adding replicas
        raises total concurrent capacity AND lowers the chance your request waits.

        Raises ValueError if no feasible plan exists (infeasible topology).
        """
        load = load or {}
        model_bytes = self._quant_bytes(model)
        manifest_hash = model.manifest_hash()
        candidates: list[Candidate] = []
        rejected: list[str] = []
        allowed = set(executable_strategies or ())

        def executable(s: Strategy) -> bool:
            if not allowed or s.value in allowed:
                return True
            line = f"{s.value}: not executable by caller"
            if line not in rejected:
                rejected.append(line)
            return False

        compatible = [n for n in nodes.values()
                      if any(b in model.runtime.backends for b in n.backends)]
        if not compatible:
            raise ValueError(
                f"no node advertises a backend in {model.runtime.backends}; "
                f"nodes have {[n.backends for n in nodes.values()]}")

        # REPLICA POOL: every node that fits the whole model is an interchangeable
        # replica. Score each for THIS request (latency + its own queue wait) and the
        # best one wins. This is load balancing + scale-out: it is the primary way
        # more nodes make the network faster for everyone.
        replicas = [n for n in compatible if self._fits_single(n, model_bytes, context_length)]
        for n in compatible:
            if n not in replicas:
                rejected.append(
                    f"single/{n.node_id}: model {model_bytes/1e9:.1f}GB + KV does not fit "
                    f"(vram {n.free_vram_bytes()/1e9:.1f}GB, ram {n.ram_free_bytes/1e9:.1f}GB)")
        for n in (replicas if executable(Strategy.SINGLE) else []):
            link = graph.link(client_node_id, n.node_id)
            ttft, tps, tr = self._predict_single(n, link, prompt_tokens, gen_tokens)
            # queue wait: requests already on this node each take ~one representative
            # response at this node's rate before yours starts.
            q = load.get(n.node_id, 0)
            queue_wait = q * (256 / tps if tps > 0 else 0.0)
            score = self._score(ttft, tps, [n], graph, [n.node_id], queue_wait_s=queue_wait)
            if q:
                tr = tr + [f"  queue: {q} ahead of you (~{queue_wait:.1f}s wait) on this replica"]
            candidates.append(Candidate(Strategy.SINGLE, [n.node_id], ttft, tps, score, tr))

        # Candidate: PIPELINE across the 2 highest-VRAM compatible nodes (proves the
        # split path). Only offered when the model needs it OR to compare cost.
        splitters = [n for n in compatible if self._can(n, "pipeline")]
        if len(splitters) >= 2 and executable(Strategy.PIPELINE):
            ranked = sorted(splitters, key=lambda n: n.free_vram_bytes(), reverse=True)[:2]
            chain = [n.node_id for n in ranked]
            # a split only works if the chain can JOINTLY hold weights + KV; two
            # 8GB nodes do not add up to a 500GB model.
            combined = sum(self._usable_bytes(n) for n in ranked)
            needed = model_bytes + self._kv_allowance(model_bytes, context_length)
            if combined < needed:
                rejected.append(
                    f"pipeline/{chain[0]}+{chain[1]}: combined memory "
                    f"{combined/1e9:.1f} GB < needed {needed/1e9:.1f} GB")
            else:
                ttft, tps, tr = self._predict_pipeline(ranked, graph, graph.link(
                    client_node_id, ranked[0].node_id), prompt_tokens, gen_tokens)
                score = self._score(ttft, tps, ranked, graph, chain)
                candidates.append(Candidate(Strategy.PIPELINE, chain, ttft, tps, score, tr))

        # SPEED candidates: more (close/fast) nodes -> a single request runs FASTER.
        # TENSOR pod: 2+ fitting replicas mutually on a LAN-class link (NVLINK/LAN/PCIe)
        # -> tensor-parallel, single-request decode scales with node count.
        tensor_capable = [n for n in replicas if self._can(n, "tensor")]
        for size in (3, 2):
            pod = tensor_capable[:size]
            if (len(pod) == size and self._mutually_lan_class(graph, [n.node_id for n in pod])
                    and executable(Strategy.TENSOR)):
                ttft, tps, tr = self._predict_tensor(
                    pod, graph, graph.link(client_node_id, pod[0].node_id), prompt_tokens)
                chain = [n.node_id for n in pod]
                # the pod is only as free as its busiest member — a loaded pod
                # loses to an idle far replica, same queue formula as SINGLE.
                q = max(load.get(n.node_id, 0) for n in pod)
                queue_wait = q * (256 / tps if tps > 0 else 0.0)
                score = self._score(ttft, tps, pod, graph, chain, queue_wait_s=queue_wait)
                if q:
                    tr = tr + [f"  queue: {q} ahead of you (~{queue_wait:.1f}s wait) on this pod"]
                candidates.append(Candidate(Strategy.TENSOR, chain, ttft, tps, score, tr))
                break
        # SPECULATIVE: a strong replica + any draft-capable peer within a tolerable RTT
        # of it -> single-stream speedup (extra node accelerates one request).
        if replicas and executable(Strategy.SPECULATIVE):
            strong = max(replicas, key=self._node_decode_tps)
            drafts = [n for n in compatible if n.node_id != strong.node_id
                      and (self._can(n, "speculative") or self._can(n, "draft"))
                      and graph.link(strong.node_id, n.node_id).rtt_ms
                      <= self.cost.speculative_max_rtt_ms]
            if drafts:
                draft = drafts[0]
                ttft, tps, tr = self._predict_speculative(
                    strong, draft, graph, graph.link(client_node_id, strong.node_id),
                    prompt_tokens)
                q = load.get(strong.node_id, 0)
                queue_wait = q * (256 / tps if tps > 0 else 0.0)
                score = self._score(ttft, tps, [strong], graph, [strong.node_id],
                                    queue_wait_s=queue_wait)
                if q:
                    tr = tr + [f"  queue: {q} ahead of you (~{queue_wait:.1f}s wait) on the verifier"]
                candidates.append(Candidate(Strategy.SPECULATIVE,
                                            [strong.node_id, draft.node_id], ttft, tps, score, tr))

        if not candidates:
            raise ValueError(
                "infeasible: no single node fits the model and no viable split found. "
                + " ".join(rejected))

        candidates.sort(key=lambda c: c.score)
        best = candidates[0]

        trace = list(best.trace)
        # capacity note: more replicas => more concurrent sessions AND a better shot
        # at a free/close node. This is the "more people = faster" scale-out property.
        if len(replicas) >= 1:
            busy = sum(1 for n in replicas if load.get(n.node_id, 0) > 0)
            trace.insert(0, f"replica pool: {len(replicas)} node(s) can serve this model "
                            f"({busy} busy); routed to the best replica for this request; "
                            f"network serves up to {len(replicas)} concurrent session(s), "
                            f"more nodes => less queueing + faster routing.")
        trace.insert(0, f"selected {best.strategy.value} "
                        f"(score {best.score:.2f}s: ttft {best.predicted_ttft_s:.2f}s, "
                        f"decode {best.predicted_decode_tps:.1f} tok/s)")
        for c in candidates[1:]:
            trace.append(f"rejected {c.strategy.value} chain={c.peer_chain}: score "
                         f"{c.score:.2f}s (ttft {c.predicted_ttft_s:.2f}s, "
                         f"decode {c.predicted_decode_tps:.1f} tok/s)")

        assignment = self._assign(best, model)
        return ExecutionPlan(
            plan_id=plan_id,
            model_manifest_hash=manifest_hash,
            model_id=model.id,
            strategy=best.strategy,
            peer_chain=best.peer_chain,
            shard_assignment=assignment,
            context_length=context_length,
            predicted_ttft_s=round(best.predicted_ttft_s, 3),
            predicted_decode_tps=round(best.predicted_decode_tps, 2),
            decision_trace=trace,
            rejected_alternatives=rejected,
        )

    def _assign(self, cand: Candidate, model: ModelManifest) -> list[ShardAssignment]:
        backend = model.runtime.backends[0] if model.runtime.backends else ""
        if cand.strategy == Strategy.SINGLE:
            return [ShardAssignment(node_id=cand.peer_chain[0], role=Strategy.SINGLE,
                                    backend=backend)]
        if cand.strategy == Strategy.TENSOR:
            # all pod members hold every layer, tensor-split (backend does the split)
            return [ShardAssignment(node_id=nid, role=Strategy.TENSOR, backend=backend)
                    for nid in cand.peer_chain]
        if cand.strategy == Strategy.SPECULATIVE:
            return [ShardAssignment(node_id=cand.peer_chain[0], role=Strategy.SINGLE, backend=backend),
                    ShardAssignment(node_id=cand.peer_chain[1], role=Strategy.SPECULATIVE, backend=backend)]
        # pipeline: split 64 layers (or configured) across the chain evenly.
        layers = 64
        n = len(cand.peer_chain)
        per = layers // n
        out = []
        for i, nid in enumerate(cand.peer_chain):
            start = i * per
            end = layers if i == n - 1 else (i + 1) * per
            out.append(ShardAssignment(node_id=nid, role=Strategy.PIPELINE,
                                       layer_start=start, layer_end=end))
        return out
