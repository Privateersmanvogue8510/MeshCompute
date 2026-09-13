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
    def _quant_bytes(self, manifest: ModelManifest, prefer_mtp: bool = True) -> int:
        best = None
        for q in manifest.quantizations:
            total = sum(f.size_bytes for f in q.files)
            if total == 0:
                continue
            if prefer_mtp and q.mtp and best is not None:
                # prefer an mtp quant of similar size; keep the smaller feasible one
                pass
            if best is None or total < best:
                best = total
        return best or 0

    def _fits_single(self, node: CapabilityRecord, model_bytes: int, ctx: int) -> bool:
        # weights + a KV allowance. KV ~ ctx * layers * heads is model-specific;
        # use a coarse 20% of weights per 32k ctx as the POC allowance.
        kv_allow = int(model_bytes * 0.20 * max(ctx, 1) / 32768)
        need = model_bytes + kv_allow
        vram = node.free_vram_bytes() or node.total_vram_bytes()
        # CPU-only nodes advertise vram 0 but can run from RAM (slow) — allow if RAM fits.
        if vram == 0:
            return node.ram_free_bytes >= need
        return vram >= need

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

    def _score(self, ttft: float, decode_tps: float, nodes: list[CapabilityRecord],
               graph: TopologyGraph, chain: list[str]) -> float:
        """Lower is better. Total time to produce a representative response +
        reliability penalty. Objective = user-visible latency."""
        REPRESENTATIVE_GEN = 256
        total = ttft + (REPRESENTATIVE_GEN / decode_tps if decode_tps > 0 else 1e9)
        # reliability: expected restart probability from worst link stability + node.
        worst = self._worst_link(graph, chain)
        p_fail = 1.0 - worst.stability
        total += p_fail * self.cost.reliability_penalty_s
        return total

    # --- main entry ----------------------------------------------------------
    def plan(self, *, model: ModelManifest, nodes: dict[str, CapabilityRecord],
             graph: TopologyGraph, client_node_id: str, context_length: int,
             prompt_tokens: int = 512, gen_tokens: int = 256,
             plan_id: str = "plan_0") -> ExecutionPlan:
        """Enumerate candidates, predict, score, pick best, and record why.

        Raises ValueError if no feasible plan exists (infeasible topology).
        """
        model_bytes = self._quant_bytes(model)
        manifest_hash = model.manifest_hash()
        candidates: list[Candidate] = []
        rejected: list[str] = []

        compatible = [n for n in nodes.values()
                      if any(b in model.runtime.backends for b in n.backends)]
        if not compatible:
            raise ValueError(
                f"no node advertises a backend in {model.runtime.backends}; "
                f"nodes have {[n.backends for n in nodes.values()]}")

        # Candidate 1..k: SINGLE on each node that fits.
        for n in compatible:
            if not self._fits_single(n, model_bytes, context_length):
                rejected.append(
                    f"single/{n.node_id}: model {model_bytes/1e9:.1f}GB + KV does not fit "
                    f"(vram {n.free_vram_bytes()/1e9:.1f}GB, ram {n.ram_free_bytes/1e9:.1f}GB)")
                continue
            link = graph.link(client_node_id, n.node_id)
            ttft, tps, tr = self._predict_single(n, link, prompt_tokens, gen_tokens)
            score = self._score(ttft, tps, [n], graph, [n.node_id])
            candidates.append(Candidate(Strategy.SINGLE, [n.node_id], ttft, tps, score, tr))

        # Candidate: PIPELINE across the 2 highest-VRAM compatible nodes (proves the
        # split path). Only offered when the model needs it OR to compare cost.
        if len(compatible) >= 2:
            ranked = sorted(compatible, key=lambda n: n.free_vram_bytes(), reverse=True)[:2]
            ttft, tps, tr = self._predict_pipeline(ranked, graph, graph.link(
                client_node_id, ranked[0].node_id), prompt_tokens, gen_tokens)
            chain = [n.node_id for n in ranked]
            score = self._score(ttft, tps, ranked, graph, chain)
            candidates.append(Candidate(Strategy.PIPELINE, chain, ttft, tps, score, tr))

        if not candidates:
            raise ValueError(
                "infeasible: no single node fits the model and no viable split found. "
                + " ".join(rejected))

        candidates.sort(key=lambda c: c.score)
        best = candidates[0]

        trace = list(best.trace)
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
        if cand.strategy == Strategy.SINGLE:
            return [ShardAssignment(node_id=cand.peer_chain[0], role=Strategy.SINGLE,
                                    backend=model.runtime.backends[0] if model.runtime.backends else "")]
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
