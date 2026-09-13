""""Donate a percentage of the PC" — contribution policy enforcement
(CONFIGURATION.md "compute"/"availability"/"thermal", INSTRUCTIONS §7).

Two jobs:
  1. Idle-awareness: an async ContributionController that watches user-input
     idle time, CPU/GPU load and AC power, and flips ACTIVE/PAUSED so the
     daemon can gate network availability on it (see daemon.py's heartbeat
     loop: it simply skips POSTing while PAUSED — the control plane's own
     heartbeat timeout then ages the node to offline, i.e. "advertise
     unavailable", with no protocol change needed).
  2. Resource budgets: turn max_cpu_percent/max_vram_percent/max_ram_gb into
     concrete numbers (thread count, VRAM bytes, a rough ctx-size cap) the
     daemon applies to the actual llama-server launch and to the advertised
     capability, so contributors share only the % they chose and the
     scheduler never over-places work on them.

Every raw signal degrades gracefully: no crash if a tool/file is missing,
just "unknown" (None), which the gating logic treats as "don't block on it."
"""

from __future__ import annotations

import asyncio
import contextlib
import enum
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from meshcompute_protocol import ContributionPolicy

# --------------------------------------------------------------------------- raw signals


def user_idle_seconds() -> float | None:
    """Seconds since the last user input, or None if no signal is available
    (headless box, or a Wayland session where XWayland input doesn't route
    through xprintidle).
    TODO(phase-1.5): a real Wayland idle signal (ext-idle-notify-v1) and a
    Windows/macOS equivalent; xprintidle only covers a real X11 session.
    """
    if os.environ.get("DISPLAY") and shutil.which("xprintidle"):
        try:
            out = subprocess.run(["xprintidle"], capture_output=True, text=True, timeout=3)
            if out.returncode == 0 and out.stdout.strip():
                return int(out.stdout.strip()) / 1000.0
        except (OSError, subprocess.SubprocessError, ValueError):
            pass
    return None


def cpu_load_percent() -> float:
    """Recent CPU business as a % of all cores, from the 1-minute load
    average — stdlib only, no /proc/stat delta-sampling needed."""
    try:
        load1, _, _ = os.getloadavg()
    except (OSError, AttributeError):
        return 0.0
    ncores = os.cpu_count() or 1
    return min(100.0, 100.0 * load1 / ncores)


def gpu_util_percent() -> float | None:
    """nvidia-smi utilization.gpu, or None when there's no nvidia GPU (never
    blocks contribution on a signal that doesn't exist on this box)."""
    if not shutil.which("nvidia-smi"):
        return None
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5)
        if out.returncode == 0 and out.stdout.strip():
            return float(out.stdout.strip().splitlines()[0])
    except (OSError, subprocess.SubprocessError, ValueError):
        pass
    return None


def gpu_temperature_c() -> float | None:
    """nvidia-smi GPU temperature, or None when there's no nvidia GPU."""
    if not shutil.which("nvidia-smi"):
        return None
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5)
        if out.returncode == 0 and out.stdout.strip():
            return float(out.stdout.strip().splitlines()[0])
    except (OSError, subprocess.SubprocessError, ValueError):
        pass
    return None


def parse_gpu_devices(gpu_devices: str | list[int] | None) -> set[int] | None:
    """The "donate a percentage of the PC" per-GPU on/off selection: None
    return = "auto" (every detected GPU enabled); otherwise the explicit set
    of physical indices a contributor chose to donate. Shared by capability.py
    (enumeration) and the budget helpers below (single source of truth for
    what "selected" means)."""
    if gpu_devices is None or gpu_devices == "auto":
        return None
    if isinstance(gpu_devices, str):
        return {int(x) for x in gpu_devices.split(",") if x.strip() != ""}
    return {int(x) for x in gpu_devices}


def per_gpu_free_vram_bytes(gpu_devices: str | list[int] | None = "auto") -> dict[int, int]:
    """{physical index -> free VRAM bytes} for the SELECTED nvidia GPUs, or
    {} on a CPU-only box / no nvidia-smi. Index order here is what
    backends/llamacpp.py's gpu_launch_args uses to build CUDA_VISIBLE_DEVICES
    and --tensor-split."""
    if not shutil.which("nvidia-smi"):
        return {}
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5)
    except (OSError, subprocess.SubprocessError):
        return {}
    if out.returncode != 0 or not out.stdout.strip():
        return {}
    selected = parse_gpu_devices(gpu_devices)
    result: dict[int, int] = {}
    for line in out.stdout.strip().splitlines():
        try:
            idx_s, mib_s = (p.strip() for p in line.split(","))
            idx = int(idx_s)
        except ValueError:
            continue
        if selected is not None and idx not in selected:
            continue
        result[idx] = int(float(mib_s)) * (1 << 20)
    return result


def total_vram_bytes(gpu_devices: str | list[int] | None = "auto") -> int:
    """Total VRAM across the SELECTED nvidia GPUs (per-GPU on/off), or 0
    (CPU-only box / no nvidia-smi, or a selection that excludes every card)."""
    if not shutil.which("nvidia-smi"):
        return 0
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5)
    except (OSError, subprocess.SubprocessError):
        return 0
    if out.returncode != 0 or not out.stdout.strip():
        return 0
    selected = parse_gpu_devices(gpu_devices)
    total = 0
    for line in out.stdout.strip().splitlines():
        try:
            idx_s, mib_s = (p.strip() for p in line.split(","))
            idx = int(idx_s)
        except ValueError:
            continue
        if selected is not None and idx not in selected:
            continue
        total += int(float(mib_s)) * (1 << 20)
    return total


def total_ram_bytes() -> int:
    """Total physical RAM. /proc/meminfo on Linux, os.sysconf portable
    fallback (macOS/BSD) — mirrors capability.py's ram_free_bytes() but for
    the total instead of the currently-available figure."""
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    try:
        return os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
    except (ValueError, OSError, AttributeError):
        return 0


def on_ac_power() -> bool | None:
    """True/False from /sys/class/power_supply, or None when there's no
    battery at all (a desktop with no power_supply entries — like this box —
    is never blocked by require_ac_power)."""
    base = Path("/sys/class/power_supply")
    if not base.is_dir():
        return None
    entries = list(base.iterdir())
    if not entries:
        return None
    for entry in entries:
        with contextlib.suppress(OSError):
            if (entry / "type").read_text().strip() == "Mains":
                return (entry / "online").read_text().strip() == "1"
    for entry in entries:  # no Mains node — infer from a battery's charge status
        with contextlib.suppress(OSError):
            return (entry / "status").read_text().strip() != "Discharging"
    return None


@dataclass
class IdleSnapshot:
    user_idle_s: float | None
    cpu_percent: float
    gpu_percent: float | None
    on_ac: bool | None

    def is_idle(self, policy: ContributionPolicy, *, check_user_activity: bool = True) -> bool:
        if policy.require_ac_power and self.on_ac is False:
            return False
        if self.cpu_percent > policy.max_cpu_percent:
            return False
        if self.gpu_percent is not None and self.gpu_percent > policy.max_gpu_percent:
            return False
        if check_user_activity and self.user_idle_s is not None:
            if self.user_idle_s < policy.idle_minutes_before_start * 60:
                return False
        return True


def read_idle_snapshot() -> IdleSnapshot:
    return IdleSnapshot(user_idle_seconds(), cpu_load_percent(), gpu_util_percent(), on_ac_power())


# --------------------------------------------------------------------------- resource budgets


def cpu_thread_budget(max_cpu_percent: int, ncores: int | None = None) -> int:
    """max_cpu_percent of the core count, floored at 1 thread."""
    ncores = ncores or os.cpu_count() or 1
    return max(1, round(ncores * max_cpu_percent / 100))


def vram_budget_bytes(total_vram: int, max_vram_percent: int) -> int:
    return int(total_vram * max_vram_percent / 100)


def ram_budget_bytes(total_ram: int, max_ram_gb: int | None) -> int:
    """max_ram_gb caps the absolute share; unset means "no explicit cap" (use
    all of `total_ram`) per CONFIGURATION.md."""
    if max_ram_gb is None:
        return total_ram
    cap = max_ram_gb * (1 << 30)
    return min(total_ram, cap) if total_ram else cap


# Rough, deliberately conservative constants for the two "bound X by budget"
# heuristics below. Neither has the real per-model numbers available at the
# point the daemon must pick launch flags (before the engine has parsed the
# GGUF), so both are safety backstops, not a precise memory model.
# TODO(phase-1.5): read real KV-cache-bytes-per-token and layer count from the
# GGUF header (available once ensure_model resolves it) instead of constants.
_APPROX_KV_BYTES_PER_TOKEN = 128 * 1024
_DEFAULT_LAYER_COUNT_HINT = 32


def bound_ctx_size(requested_ctx: int, ram_budget: int, min_ctx: int = 512) -> int:
    """Coarse RAM-aware --ctx-size cap. ponytail: constant-bytes-per-token
    estimate, not a real per-model memory model — see module TODO above."""
    if ram_budget <= 0:
        return requested_ctx
    budget_ctx = ram_budget // _APPROX_KV_BYTES_PER_TOKEN
    if budget_ctx <= 0:
        return min_ctx
    return max(min_ctx, min(requested_ctx, budget_ctx))


def bound_n_gpu_layers(requested_ngl: int, vram_budget: int, model_file_bytes: int,
                        total_layers_hint: int = _DEFAULT_LAYER_COUNT_HINT) -> int:
    """Coarse VRAM-aware -ngl cap from a whole-file-size / layer-count-hint
    estimate. Untestable end to end on this CPU-only box regardless of
    accuracy — see module TODO above."""
    if requested_ngl <= 0 or vram_budget <= 0 or model_file_bytes <= 0:
        return requested_ngl
    bytes_per_layer = model_file_bytes / max(1, total_layers_hint)
    ngl_budget = int(vram_budget // bytes_per_layer)
    return max(0, min(requested_ngl, ngl_budget))


# --------------------------------------------------------------------------- controller


class ContributionState(enum.Enum):
    ACTIVE = "active"
    PAUSED = "paused"


class ContributionController:
    """Owns the idle/active gate for one running daemon.

    Usage: `await controller.start()` once the backend is up, check
    `.is_available()` (or `.state`) before advertising/accepting network work,
    `await controller.stop()` on shutdown.

    Phase 1 has no in-flight mesh-request queue to drain yet (the daemon only
    registers/heartbeats and serves its own local endpoint; routed mesh
    inference sessions land in a later milestone) — so today PAUSED means
    "stop advertising availability" (daemon.py's heartbeat loop skips POSTing
    while paused). CONFIGURATION.md's "drain or cancel current work" applies
    once there IS in-flight mesh work to drain.
    TODO(phase-1.5): actually cancel/drain in-flight *mesh* sessions on pause
    once session dispatch exists; the local personal endpoint is deliberately
    NOT paused — idle_only governs network contribution, not the user's own
    local use of their own daemon.
    """

    def __init__(self, policy: ContributionPolicy, *, poll_interval_s: float = 15.0) -> None:
        self.policy = policy
        self.poll_interval_s = poll_interval_s
        self.state = ContributionState.ACTIVE
        self.last_snapshot: IdleSnapshot | None = None
        self._task: asyncio.Task | None = None

    def is_available(self) -> bool:
        return self.state == ContributionState.ACTIVE

    def _tick(self) -> ContributionState:
        if not self.policy.idle_only:
            self.state = ContributionState.ACTIVE
            return self.state
        snap = read_idle_snapshot()
        self.last_snapshot = snap
        if self.state == ContributionState.PAUSED:
            # Resuming always requires the full idle bar, including user
            # input idle time (idle_minutes_before_start) — that's the
            # "start" condition CONFIGURATION.md describes.
            if snap.is_idle(self.policy, check_user_activity=True):
                self.state = ContributionState.ACTIVE
        else:  # ACTIVE
            # pause_on_user_activity=False means: once contributing, renewed
            # user input alone doesn't interrupt it — only CPU/GPU/AC still can.
            if not snap.is_idle(self.policy, check_user_activity=self.policy.pause_on_user_activity):
                self.state = ContributionState.PAUSED
        return self.state

    async def _loop(self) -> None:
        while True:
            await asyncio.sleep(self.poll_interval_s)
            self._tick()

    async def start(self) -> None:
        self._tick()  # establish a real initial state before the daemon registers/prints it
        self._task = asyncio.create_task(self._loop())

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
            self._task = None


if __name__ == "__main__":
    snap = read_idle_snapshot()
    print(f"idle snapshot: user_idle_s={snap.user_idle_s} cpu%={snap.cpu_percent:.1f} "
          f"gpu%={snap.gpu_percent} on_ac={snap.on_ac}")
    assert snap.cpu_percent >= 0.0

    ncores = os.cpu_count() or 1
    assert cpu_thread_budget(50, ncores) == max(1, round(ncores * 0.5))
    assert cpu_thread_budget(0, ncores) == 1, "must never budget zero threads"
    assert vram_budget_bytes(10_000, 50) == 5_000
    assert ram_budget_bytes(10_000, None) == 10_000
    assert ram_budget_bytes(10_000, 1) == min(10_000, 1 << 30)
    assert bound_ctx_size(4096, 0) == 4096, "no RAM budget known -> don't second-guess the request"
    assert bound_n_gpu_layers(999, 0, 100) == 999, "no VRAM budget known -> don't second-guess the request"

    assert parse_gpu_devices("auto") is None
    assert parse_gpu_devices(None) is None
    assert parse_gpu_devices("0,1") == {0, 1}
    assert parse_gpu_devices([1]) == {1}
    # this box has no nvidia-smi -> both degrade to empty/zero, never crash
    assert per_gpu_free_vram_bytes("auto") == {}
    assert total_vram_bytes([0]) == 0

    always_on = ContributionController(ContributionPolicy(idle_only=False))
    assert always_on._tick() == ContributionState.ACTIVE

    never_idle = ContributionController(
        ContributionPolicy(idle_only=True, max_cpu_percent=0, idle_minutes_before_start=999))
    assert never_idle._tick() == ContributionState.PAUSED, "max_cpu_percent=0 must never be satisfiable"

    print(f"contribution.py self-check PASSED (ncores={ncores}, "
          f"total_vram={total_vram_bytes()}, total_ram={total_ram_bytes()})")
