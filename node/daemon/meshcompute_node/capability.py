"""Real node capability measurement (INSTRUCTIONS §7, §26 rule 7): never trust
self-reported performance. This runs the actual benchmark against the actual
running backend and reads real GPU/RAM state instead of a claimed number.
"""

from __future__ import annotations

import os
import platform
import subprocess
from datetime import datetime, timezone

from meshcompute_protocol import BenchmarkResult, CapabilityRecord, ContributionPolicy, GpuInfo
from meshcompute_runtime.backends.base import BackendAdapter, BackendCapabilities, TokenChunk


def _detect_gpu(backend_names: list[str]) -> GpuInfo:
    """Parse nvidia-smi if present; otherwise a cpu GpuInfo with vram 0 (rule 9:
    advertise measured capability, never assume a backend from a device name)."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,memory.free,compute_cap",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5)
        if out.returncode == 0 and out.stdout.strip():
            name, total_mib, free_mib, cc = (s.strip() for s in
                                              out.stdout.strip().splitlines()[0].split(","))
            return GpuInfo(vendor="nvidia", model=name,
                            vram_bytes=int(float(total_mib)) * (1 << 20),
                            free_vram_bytes=int(float(free_mib)) * (1 << 20),
                            compute_capability=cc, backend_support=[*backend_names, "cuda"])
    except (OSError, subprocess.SubprocessError, ValueError):
        pass
    return GpuInfo(vendor="cpu", model="cpu", vram_bytes=0, free_vram_bytes=0,
                    backend_support=backend_names)


def ram_free_bytes() -> int:
    """psutil is intentionally not a dependency for one number: /proc/meminfo
    covers Linux, os.sysconf covers the portable POSIX fallback (macOS/BSD)."""
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    try:
        return os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
    except (ValueError, OSError, AttributeError):
        return 0


async def measure_capability(identity, backend: BackendAdapter, config: dict | None) -> CapabilityRecord:
    """Build a signed-ready CapabilityRecord for `identity` from a live,
    running `backend`. `config` carries contribution_policy overrides
    (CONFIGURATION.md: idle_only, max_vram_percent, ...).
    """
    caps = backend.capabilities()
    models = await backend.list_models()
    model = models[0] if models else caps.id
    # Steady-state decode tok/s over the full streaming window (not just TTFT) —
    # a reasoning model's <think> block would otherwise make TTFT-to-content
    # look like decode is slow when it isn't. BackendAdapter.benchmark()'s
    # default already measures this way: decode = tokens / (total - ttft).
    bench = await backend.benchmark(model)

    policy_cfg = (config or {}).get("contribution_policy", {})
    policy = ContributionPolicy(
        idle_only=policy_cfg.get("idle_only", True),
        max_gpu_percent=policy_cfg.get("max_gpu_percent", 90),
        max_vram_percent=policy_cfg.get("max_vram_percent", 85),
        max_cpu_percent=policy_cfg.get("max_cpu_percent", 50),
        allow_public_pool=policy_cfg.get("allow_public_pool", True),
    )

    strategies = ["replica"]
    if caps.supports_pipeline_split:
        strategies.append("pipeline")

    return CapabilityRecord(
        node_id=identity.node_id,
        os=platform.system().lower(),
        arch=platform.machine(),
        gpus=[_detect_gpu(caps.backends)],
        ram_free_bytes=ram_free_bytes(),
        backends=caps.backends,
        strategies=strategies,
        contribution_policy=policy,
        benchmark=BenchmarkResult(
            decode_tokens_per_sec=bench.get("decode_tokens_per_sec"),
            # TODO(phase-1.5): base BackendAdapter.benchmark() only measures
            # decode today; a separate prefill-tok/s pass is a Phase-1.5 add.
            prefill_tokens_per_sec=None,
            measured_model=model,
            measured_at=datetime.now(timezone.utc).isoformat(),
        ),
    )


if __name__ == "__main__":
    import asyncio

    class _FakeIdentity:
        node_id = "nd_selfcheck00000000000000000000"

    class _FakeBackend(BackendAdapter):
        def capabilities(self) -> BackendCapabilities:
            return BackendCapabilities(id="fake", backends=["fake"], supports_pipeline_split=True)

        async def health(self) -> bool:
            return True

        async def list_models(self) -> list[str]:
            return ["fake-model"]

        def chat_stream(self, req):
            async def _gen():
                for ch in "hello world":
                    yield TokenChunk(text=ch)
            return _gen()

    async def _demo() -> None:
        record = await measure_capability(
            _FakeIdentity(), _FakeBackend(),
            {"contribution_policy": {"idle_only": False, "max_vram_percent": 50}})
        assert record.node_id == "nd_selfcheck00000000000000000000"
        assert record.backends == ["fake"]
        assert "pipeline" in record.strategies
        assert record.contribution_policy.idle_only is False
        assert record.contribution_policy.max_vram_percent == 50
        assert record.ram_free_bytes > 0, "expected a real /proc/meminfo (or sysconf) reading"
        assert record.benchmark.decode_tokens_per_sec is not None
        assert record.gpus[0].vendor in ("nvidia", "cpu")
        print(f"capability.py self-check PASSED: ram_free={record.ram_free_bytes}, "
              f"gpu={record.gpus[0].vendor}, decode={record.benchmark.decode_tokens_per_sec}")

    asyncio.run(_demo())
