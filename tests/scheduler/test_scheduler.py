"""Scheduler invariants (INSTRUCTIONS §3, §8): topology-aware plan selection.

Mirrors the "powerhouse behind a 255ms WAN link" scenario documented at the top of
scheduler.py: the scheduler must pick a single low-latency pod over a pipeline split
that crosses a slow inter-peer link, must load-balance replicas by queue depth, and
must reject infeasible topologies instead of guessing.
"""

from __future__ import annotations

import pytest
from meshcompute_control.scheduler import Scheduler
from meshcompute_control.topology import LinkMetrics, TopologyGraph
from meshcompute_protocol import Strategy

CLIENT = "client"


def test_single_low_latency_pod_beats_slow_wan_pipeline(manifest_factory, node_factory):
    manifest = manifest_factory(backends=("lmstudio",), size_bytes=20_000_000_000, context=8192)
    near = node_factory("nd_near", vram_bytes=24_000_000_000, free_vram_bytes=24_000_000_000,
                        decode_tps=50, prefill_tps=400)
    far = node_factory("nd_far_pod", vram_bytes=24_000_000_000, free_vram_bytes=24_000_000_000,
                       decode_tps=50, prefill_tps=400)

    graph = TopologyGraph()
    graph.set_link(CLIENT, "nd_near", LinkMetrics(rtt_ms=5, throughput_mbps=1000))
    graph.set_link(CLIENT, "nd_far_pod", LinkMetrics(rtt_ms=255, throughput_mbps=200))
    # the two nodes themselves are also far apart (a WAN pipeline split), which is
    # what should make PIPELINE lose even though both nodes individually fit.
    graph.set_link("nd_near", "nd_far_pod", LinkMetrics(rtt_ms=255, throughput_mbps=50))

    plan = Scheduler().plan(
        model=manifest, nodes={"nd_near": near, "nd_far_pod": far}, graph=graph,
        client_node_id=CLIENT, context_length=8192, prompt_tokens=200, gen_tokens=256,
    )

    assert plan.strategy == Strategy.SINGLE
    assert plan.peer_chain == ["nd_near"]
    assert plan.decision_trace  # scheduler must log why (INSTRUCTIONS §8)
    assert any("selected single" in line for line in plan.decision_trace)
    assert any("pipeline" in line for line in plan.decision_trace)  # rejected alt is logged


def test_load_aware_routing_prefers_idle_near_replica(manifest_factory, node_factory):
    manifest = manifest_factory(backends=("lmstudio",), size_bytes=1_000_000_000, context=4096)
    busy_near = node_factory("nd_busy_near", vram_bytes=8_000_000_000, free_vram_bytes=8_000_000_000,
                             decode_tps=50, prefill_tps=400)
    idle_near = node_factory("nd_idle_near", vram_bytes=8_000_000_000, free_vram_bytes=8_000_000_000,
                             decode_tps=50, prefill_tps=400)
    idle_far = node_factory("nd_idle_far", vram_bytes=8_000_000_000, free_vram_bytes=8_000_000_000,
                            decode_tps=50, prefill_tps=400)

    graph = TopologyGraph()
    graph.set_link(CLIENT, "nd_busy_near", LinkMetrics(rtt_ms=5, throughput_mbps=1000))
    graph.set_link(CLIENT, "nd_idle_near", LinkMetrics(rtt_ms=10, throughput_mbps=1000))
    graph.set_link(CLIENT, "nd_idle_far", LinkMetrics(rtt_ms=200, throughput_mbps=200))

    plan = Scheduler().plan(
        model=manifest,
        nodes={"nd_busy_near": busy_near, "nd_idle_near": idle_near, "nd_idle_far": idle_far},
        graph=graph, client_node_id=CLIENT, context_length=4096,
        load={"nd_busy_near": 5, "nd_idle_near": 0, "nd_idle_far": 0},
    )

    assert plan.strategy == Strategy.SINGLE
    assert plan.peer_chain == ["nd_idle_near"]
    # capacity note: more replicas => scale-out, and it should call out who's busy.
    assert any("replica pool" in line and "busy" in line for line in plan.decision_trace)
    assert any("nd_busy_near" in line for line in plan.decision_trace)  # rejected, logged


def test_infeasible_topology_raises_value_error(manifest_factory, node_factory):
    manifest = manifest_factory(backends=("lmstudio",), size_bytes=500_000_000_000, context=4096)
    tiny = node_factory("nd_tiny", vram_bytes=8_000_000_000, free_vram_bytes=8_000_000_000,
                        ram_free_bytes=8_000_000_000, decode_tps=50)

    with pytest.raises(ValueError, match="infeasible"):
        Scheduler().plan(model=manifest, nodes={"nd_tiny": tiny}, graph=TopologyGraph(),
                         client_node_id=CLIENT, context_length=4096)


def test_no_compatible_backend_raises_value_error(manifest_factory, node_factory):
    manifest = manifest_factory(backends=("lmstudio",), size_bytes=1_000_000_000)
    incompatible = node_factory("nd_other", backends=("some-other-backend",),
                                vram_bytes=8_000_000_000, free_vram_bytes=8_000_000_000)

    with pytest.raises(ValueError, match="no node advertises a backend"):
        Scheduler().plan(model=manifest, nodes={"nd_other": incompatible}, graph=TopologyGraph(),
                         client_node_id=CLIENT, context_length=4096)


def test_pipeline_not_proposed_with_a_single_node(manifest_factory, node_factory):
    manifest = manifest_factory(backends=("lmstudio",), size_bytes=1_000_000_000)
    solo = node_factory("nd_solo", vram_bytes=8_000_000_000, free_vram_bytes=8_000_000_000,
                        decode_tps=50, prefill_tps=400)

    plan = Scheduler().plan(model=manifest, nodes={"nd_solo": solo}, graph=TopologyGraph(),
                            client_node_id=CLIENT, context_length=4096)

    assert plan.strategy == Strategy.SINGLE
    assert not any("pipeline" in line.lower() for line in plan.decision_trace)


LAN = dict(rtt_ms=1, throughput_mbps=10000, path_type="direct")
POD_KW = dict(vram_bytes=24_000_000_000, free_vram_bytes=24_000_000_000,
              decode_tps=50, prefill_tps=400)


def _lan_graph(node_ids):
    """Every pair (and the client) on a LAN-class link, in BOTH directions —
    the precondition _mutually_lan_class() checks for a tensor pod."""
    graph = TopologyGraph()
    for i, a in enumerate([CLIENT, *node_ids]):
        for b in [CLIENT, *node_ids][i + 1:]:
            graph.set_link(a, b, LinkMetrics(**LAN))
            graph.set_link(b, a, LinkMetrics(**LAN))
    return graph


def test_executable_strategies_limits_candidates(manifest_factory, node_factory):
    """The caller says what it can actually run (the Phase-1 gateway drives one
    OpenAI-compatible backend, so it sends ["single"]). A plan the caller cannot
    execute is worse than no plan, so those strategies must never be proposed."""
    manifest = manifest_factory(size_bytes=1_000_000_000)
    nodes = {nid: node_factory(nid, strategies=("single", "replica", "tensor"), **POD_KW)
             for nid in ("nd_a", "nd_b")}

    plan = Scheduler().plan(model=manifest, nodes=nodes, graph=_lan_graph(list(nodes)),
                            client_node_id=CLIENT, context_length=4096,
                            executable_strategies=["single"])

    assert plan.strategy == Strategy.SINGLE
    assert any("tensor: not executable by caller" in r for r in plan.rejected_alternatives)


def test_pipeline_rejected_when_chain_cannot_hold_the_model(manifest_factory, node_factory):
    """Two 8GB nodes do not add up to a 500GB model: splitting layers only works
    if the chain JOINTLY holds weights + KV, otherwise it is still infeasible."""
    manifest = manifest_factory(size_bytes=500_000_000_000, context=4096)
    nodes = {nid: node_factory(nid, vram_bytes=8_000_000_000, free_vram_bytes=8_000_000_000,
                               ram_free_bytes=8_000_000_000, decode_tps=50)
             for nid in ("nd_tiny_a", "nd_tiny_b")}

    with pytest.raises(ValueError, match="infeasible") as exc:
        Scheduler().plan(model=manifest, nodes=nodes, graph=TopologyGraph(),
                         client_node_id=CLIENT, context_length=4096)
    assert "combined memory" in str(exc.value)


def test_tensor_pod_requires_advertised_tensor_strategy(manifest_factory, node_factory):
    """A node that never advertised "tensor" cannot be drafted into a tensor pod —
    the scheduler must respect CapabilityRecord.strategies, not just the topology."""
    manifest = manifest_factory(size_bytes=1_000_000_000)

    def pod(strategies):
        return {nid: node_factory(nid, strategies=strategies, **POD_KW)
                for nid in ("nd_a", "nd_b")}

    graph = _lan_graph(["nd_a", "nd_b"])
    without = Scheduler().plan(model=manifest, nodes=pod(("single", "replica")), graph=graph,
                               client_node_id=CLIENT, context_length=4096)
    assert without.strategy != Strategy.TENSOR
    assert not any("tensor" in line for line in without.decision_trace)

    with_tensor = Scheduler().plan(model=manifest, nodes=pod(("single", "replica", "tensor")),
                                   graph=graph, client_node_id=CLIENT, context_length=4096)
    assert with_tensor.strategy == Strategy.TENSOR
    assert with_tensor.peer_chain == ["nd_a", "nd_b"]


def test_tensor_pod_score_includes_queue_wait(manifest_factory, node_factory):
    """A fast pod that 50 requests are already queued on is NOT the fast option.
    Tensor/speculative must pay the same queue-wait term SINGLE does, or the
    scheduler sends everyone to the same busy pod."""
    manifest = manifest_factory(size_bytes=1_000_000_000)
    nodes = {nid: node_factory(nid, strategies=("single", "replica", "tensor"), **POD_KW)
             for nid in ("nd_pod_a", "nd_pod_b", "nd_idle_far")}

    graph = _lan_graph(["nd_pod_a", "nd_pod_b"])
    graph.set_link(CLIENT, "nd_idle_far", LinkMetrics(rtt_ms=200, throughput_mbps=200))

    plan = Scheduler().plan(model=manifest, nodes=nodes, graph=graph, client_node_id=CLIENT,
                            context_length=4096,
                            load={"nd_pod_a": 50, "nd_pod_b": 50, "nd_idle_far": 0})

    assert plan.strategy == Strategy.SINGLE
    assert plan.peer_chain == ["nd_idle_far"]
