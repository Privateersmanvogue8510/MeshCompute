from __future__ import annotations

import pytest
from meshcompute_protocol import BenchmarkResult, CapabilityRecord, GpuInfo


def make_node(node_id, *, backends=("lmstudio",), vram_bytes=0, free_vram_bytes=0,
              ram_free_bytes=0, decode_tps=None, prefill_tps=None,
              strategies=("single", "replica", "pipeline")) -> CapabilityRecord:
    gpus = []
    if vram_bytes or free_vram_bytes:
        gpus = [GpuInfo(vendor="nvidia", model="test-gpu", vram_bytes=vram_bytes,
                        free_vram_bytes=free_vram_bytes, backend_support=list(backends))]
    return CapabilityRecord(
        node_id=node_id, backends=list(backends), strategies=list(strategies),
        gpus=gpus, ram_free_bytes=ram_free_bytes,
        benchmark=BenchmarkResult(decode_tokens_per_sec=decode_tps,
                                  prefill_tokens_per_sec=prefill_tps),
    )


@pytest.fixture
def node_factory():
    return make_node
