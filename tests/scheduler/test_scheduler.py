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
