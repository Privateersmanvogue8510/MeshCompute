"""Real node capability measurement (INSTRUCTIONS §7, §26 rule 7): never trust
self-reported performance. This runs the actual benchmark against the actual
running backend and reads real GPU/RAM state instead of a claimed number.
"""

from __future__ import annotations

import os
import platform
import re
import subprocess
from datetime import datetime, timezone

from meshcompute_protocol import BenchmarkResult, CapabilityRecord, ContributionPolicy, GpuInfo
from meshcompute_runtime.backends.base import BackendAdapter, BackendCapabilities, TokenChunk

from .contribution import parse_gpu_devices


def _cpu_gpu(backend_names: list[str]) -> GpuInfo:
    return GpuInfo(vendor="cpu", model="cpu", vram_bytes=0, free_vram_bytes=0,
                    backend_support=backend_names)


def _detect_nvlink_indices() -> set[int]:
    """Best-effort: which GPU indices report an active NVLINK peer, via
    `nvidia-smi nvlink --status` (prints a "GPU N:" block per card with
    "Link M: ... <Speed> ... " lines when a link is up). Absent tool, no
    NVLINK hardware, or an unparsable format just means "none detected" —
    this never blocks capability reporting.
    TODO(phase-1.5): confirm the exact `nvlink --status` text on a real 2x3090
    NVLINK box — untestable here (no nvidia-smi at all on this CPU-only box).
    """
    try:
        out = subprocess.run(["nvidia-smi", "nvlink", "--status"],
                              capture_output=True, text=True, timeout=5)
    except (OSError, subprocess.SubprocessError):
        return set()
    if out.returncode != 0 or not out.stdout.strip():
        return set()
    linked: set[int] = set()
    current_gpu: int | None = None
    for line in out.stdout.splitlines():
        line = line.strip()
        m = re.match(r"GPU (\d+):", line)
        if m:
            current_gpu = int(m.group(1))
            continue
        if current_gpu is not None and line.startswith("Link") and "inactive" not in line.lower():
            # a "Link N: <speed> GB/s" line under a GPU block means that link
            # is up; an explicit "inactive"/"<inactive>" line means it's not.
            linked.add(current_gpu)
    return linked


def _parse_gpu_query_csv(csv_lines: list[str], backend_names: list[str],
                          selected: set[int] | None, nvlink_indices: set[int]) -> list[GpuInfo]:
    """Pure parser (no subprocess): turn `nvidia-smi --query-gpu=index,name,
    memory.total,memory.free,compute_cap --format=csv,noheader,nounits` output
    lines into one GpuInfo per selected card. Unit-tested directly with a
    captured multi-GPU sample — no GPU required to exercise this logic.

    GpuInfo has no dedicated `index` field (that's a protocol-package change,
    out of scope here) — the physical index is embedded in `model` (human
    readable) and as a "index:N" tag in backend_support (machine-parseable),
    so a multi-GPU record still lets a caller recover which card is which.
    TODO(phase-1.5): a real `index: int` field on GpuInfo would be cleaner.
    """
    gpus: list[GpuInfo] = []
    for line in csv_lines:
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 5:
            continue
        idx_s, name, total_mib, free_mib, cc = parts
        try:
            idx = int(idx_s)
        except ValueError:
            continue
        if selected is not None and idx not in selected:
            continue
        support = [*backend_names, "cuda", f"index:{idx}"]
        if idx in nvlink_indices:
            support.append("nvlink")
        gpus.append(GpuInfo(vendor="nvidia", model=f"{name} [gpu{idx}]",
                             vram_bytes=int(float(total_mib)) * (1 << 20),
                             free_vram_bytes=int(float(free_mib)) * (1 << 20),
                             compute_capability=cc, backend_support=support))
    return gpus


def detect_gpus(backend_names: list[str],
                 gpu_devices: str | list[int] | None = "auto") -> list[GpuInfo]:
    """One GpuInfo per nvidia GPU (rule 9: real enumeration, never assumed
    from a device name), honoring an explicit gpu_devices selection
    (per-GPU on/off: advertise + sum VRAM only for the enabled indices).
    Falls back to a single cpu GpuInfo when nvidia-smi is absent (this box),
    reports nothing, or the selection excludes every detected card.
    """
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,memory.total,memory.free,compute_cap",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5)
    except (OSError, subprocess.SubprocessError):
        return [_cpu_gpu(backend_names)]
    if out.returncode != 0 or not out.stdout.strip():
        return [_cpu_gpu(backend_names)]

    selected = parse_gpu_devices(gpu_devices)
    nvlink_indices = _detect_nvlink_indices()
    gpus = _parse_gpu_query_csv(out.stdout.strip().splitlines(), backend_names,
                                 selected, nvlink_indices)
    return gpus or [_cpu_gpu(backend_names)]


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

    gpu_devices = (config or {}).get("gpu_devices", "auto")
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
        gpus=detect_gpus(caps.backends, gpu_devices),
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

    # --- multi-GPU parser: fixture from a captured 2x RTX 3090 nvidia-smi
    # sample (no real GPU needed — this is the pure-parser proof). ---
    _SAMPLE_2GPU_CSV = [
        "0, NVIDIA GeForce RTX 3090, 24576, 23000, 8.6",
        "1, NVIDIA GeForce RTX 3090, 24576, 22500, 8.6",
    ]
    both = _parse_gpu_query_csv(_SAMPLE_2GPU_CSV, ["llamacpp"], selected=None, nvlink_indices={0, 1})
    assert len(both) == 2
    assert both[0].vram_bytes == 24576 * (1 << 20) and both[1].vram_bytes == 24576 * (1 << 20)
    assert sum(g.vram_bytes for g in both) == 2 * 24576 * (1 << 20)
    assert "index:0" in both[0].backend_support and "index:1" in both[1].backend_support
    assert "nvlink" in both[0].backend_support and "nvlink" in both[1].backend_support

    only_0 = _parse_gpu_query_csv(_SAMPLE_2GPU_CSV, ["llamacpp"], selected={0}, nvlink_indices=set())
    assert len(only_0) == 1 and "index:0" in only_0[0].backend_support
    assert "nvlink" not in only_0[0].backend_support

    only_0_via_detect = _parse_gpu_query_csv(_SAMPLE_2GPU_CSV, ["llamacpp"],
                                              selected=parse_gpu_devices([0]), nvlink_indices=set())
    assert len(only_0_via_detect) == 1
    print(f"multi-GPU parser self-check PASSED: 2-GPU total={sum(g.vram_bytes for g in both)}, "
          f"gpu_devices=[0] -> {len(only_0)} GPU")

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
