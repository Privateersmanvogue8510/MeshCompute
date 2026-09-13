"""Cross-platform host signals for the contributor node daemon (Linux/macOS/
Windows). stdlib only — no psutil. Mirrors contribution.py/capability.py's
Linux-only /proc + xprintidle signals but dispatches on platform.system() so
the daemon can run its idle-awareness and resource-budget logic on any host.

Every function degrades gracefully: a missing tool, file or API means
"unknown" (0/None), never an exception — the gating logic in contribution.py
already treats those as "don't block on it."
"""

from __future__ import annotations

import ctypes
import os
import platform
import re
import shutil
import subprocess
from pathlib import Path

# --------------------------------------------------------------------------- memory


def total_ram_bytes() -> int:
    """Total physical RAM, 0 if unknown."""
    system = platform.system()
    if system == "Linux":
        try:
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        return int(line.split()[1]) * 1024
        except (OSError, ValueError, IndexError):
            pass
        return 0
    if system == "Darwin":
        out = _run(["sysctl", "-n", "hw.memsize"])
        if out:
            try:
                return int(out.strip())
            except ValueError:
                pass
        return 0
    if system == "Windows":
        mem = _windows_memory_status()
        return mem.ullTotalPhys if mem is not None else 0
    return 0


def free_ram_bytes() -> int:
    """"Available" memory (not just literally free — reclaimable cache counts
    as available), 0 if unknown."""
    system = platform.system()
    if system == "Linux":
        try:
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemAvailable:"):
                        return int(line.split()[1]) * 1024
        except (OSError, ValueError, IndexError):
            pass
        return 0
    if system == "Darwin":
        return _macos_free_ram_bytes()
    if system == "Windows":
        mem = _windows_memory_status()
        return mem.ullAvailPhys if mem is not None else 0
    return 0


def _macos_free_ram_bytes() -> int:
    out = _run(["vm_stat"])
    if not out:
        return 0
    lines = out.splitlines()
    if not lines:
        return 0
    m = re.search(r"page size of (\d+) bytes", lines[0])
    if not m:
        return 0
    page_size = int(m.group(1))
    pages = 0
    try:
        for key in ("Pages free", "Pages inactive", "Pages speculative"):
            for line in lines:
                if line.startswith(key):
                    pages += int(line.split(":")[1].strip().rstrip("."))
                    break
    except (ValueError, IndexError):
        return 0
    return pages * page_size


# --------------------------------------------------------------------------- disk


def disk_free_bytes(path: str | os.PathLike = ".") -> int:
    """Free space on the filesystem containing `path`, 0 on error."""
    try:
        return shutil.disk_usage(path).free
    except OSError:
        return 0


# --------------------------------------------------------------------------- idle input


def user_idle_seconds() -> float | None:
    """Seconds since the last keyboard/mouse input, or None if no signal is
    available on this platform/session."""
    system = platform.system()
    if system == "Linux":
        if os.environ.get("DISPLAY") and shutil.which("xprintidle"):
            out = _run(["xprintidle"])
            if out and out.strip():
                try:
                    return int(out.strip()) / 1000.0
                except ValueError:
                    pass
        return None
    if system == "Darwin":
        out = _run(["ioreg", "-c", "IOHIDSystem", "-d", "4"])
        if out:
            m = re.search(r'"HIDIdleTime"\s*=\s*(\d+)', out)
            if m:
                return int(m.group(1)) / 1_000_000_000.0
        return None
    if system == "Windows":
        return _windows_idle_seconds()
    return None


# --------------------------------------------------------------------------- power


def on_ac_power() -> bool | None:
    """True/False, or None when there's no battery to read (desktop) or the
    signal can't be determined."""
    system = platform.system()
    if system == "Linux":
        return _linux_on_ac_power()
    if system == "Darwin":
        out = _run(["pmset", "-g", "batt"])
        if not out:
            return None
        lines = out.splitlines()
        if not lines:
            return None
        first = lines[0]
        if "'AC Power'" in first:
            return True
        if "'Battery Power'" in first:
            return False
        return None
    if system == "Windows":
        return _windows_on_ac_power()
    return None


def _linux_on_ac_power() -> bool | None:
    base = Path("/sys/class/power_supply")
    try:
        if not base.is_dir():
            return None
        entries = list(base.iterdir())
    except OSError:
        return None
    if not entries:
        return None
    for entry in entries:
        try:
            if (entry / "type").read_text().strip() == "Mains":
                return (entry / "online").read_text().strip() == "1"
        except OSError:
            continue
    for entry in entries:  # no Mains node — infer from a battery's charge status
        try:
            return (entry / "status").read_text().strip() != "Discharging"
        except OSError:
            continue
    return None


# --------------------------------------------------------------------------- cpu load


def cpu_load_percent() -> float:
    """Recent CPU business as a % of all cores, 0.0 if unknown (Windows has
    no cheap stdlib CPU%; the daemon then gates on idle-input time instead)."""
    try:
        load1 = os.getloadavg()[0]
    except (OSError, AttributeError):
        return 0.0
    ncores = os.cpu_count() or 1
    return min(100.0, 100.0 * load1 / ncores)


# --------------------------------------------------------------------------- subprocess helper


def _run(cmd: list[str]) -> str | None:
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=3, check=False)
    except (OSError, subprocess.SubprocessError, ValueError):
        return None
    if out.returncode != 0:
        return None
    return out.stdout


# --------------------------------------------------------------------------- Windows ctypes


class _MEMORYSTATUSEX(ctypes.Structure):
    _fields_ = [
        ("dwLength", ctypes.c_ulong),
        ("dwMemoryLoad", ctypes.c_ulong),
        ("ullTotalPhys", ctypes.c_ulonglong),
        ("ullAvailPhys", ctypes.c_ulonglong),
        ("ullTotalPageFile", ctypes.c_ulonglong),
        ("ullAvailPageFile", ctypes.c_ulonglong),
        ("ullTotalVirtual", ctypes.c_ulonglong),
        ("ullAvailVirtual", ctypes.c_ulonglong),
        ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
    ]


class _LASTINPUTINFO(ctypes.Structure):
    _fields_ = [
        ("cbSize", ctypes.c_uint),
        ("dwTime", ctypes.c_uint),
    ]


class _SYSTEM_POWER_STATUS(ctypes.Structure):
    _fields_ = [
        ("ACLineStatus", ctypes.c_byte),
        ("BatteryFlag", ctypes.c_byte),
        ("BatteryLifePercent", ctypes.c_byte),
        ("SystemStatusFlag", ctypes.c_byte),
        ("BatteryLifeTime", ctypes.c_ulong),
        ("BatteryFullLifeTime", ctypes.c_ulong),
    ]


def _windows_memory_status() -> _MEMORYSTATUSEX | None:
    try:
        mem = _MEMORYSTATUSEX()
        mem.dwLength = ctypes.sizeof(mem)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(mem)):  # type: ignore[attr-defined]
            return None
        return mem
    except (AttributeError, OSError, ValueError):
        return None


def _windows_idle_seconds() -> float | None:
    try:
        info = _LASTINPUTINFO()
        info.cbSize = ctypes.sizeof(info)
        if not ctypes.windll.user32.GetLastInputInfo(ctypes.byref(info)):  # type: ignore[attr-defined]
            return None
        tick_count = ctypes.windll.kernel32.GetTickCount()  # type: ignore[attr-defined]
        idle_ms = (tick_count - info.dwTime) & 0xFFFFFFFF
        return idle_ms / 1000.0
    except (AttributeError, OSError, ValueError):
        return None


def _windows_on_ac_power() -> bool | None:
    try:
        status = _SYSTEM_POWER_STATUS()
        if not ctypes.windll.kernel32.GetSystemPowerStatus(ctypes.byref(status)):  # type: ignore[attr-defined]
            return None
        if status.BatteryFlag & 128:  # no system battery
            return None
        if status.ACLineStatus == 1:
            return True
        if status.ACLineStatus == 0:
            return False
        return None
    except (AttributeError, OSError, ValueError):
        return None


if __name__ == "__main__":
    total = total_ram_bytes()
    free = free_ram_bytes()
    disk = disk_free_bytes(".")
    idle = user_idle_seconds()
    ac = on_ac_power()
    load = cpu_load_percent()

    print(f"platform: {platform.system()}")
    print(f"total_ram_bytes: {total}")
    print(f"free_ram_bytes: {free}")
    print(f"disk_free_bytes('.'): {disk}")
    print(f"user_idle_seconds: {idle}")
    print(f"on_ac_power: {ac}")
    print(f"cpu_load_percent: {load:.1f}")

    assert total > 0, "expected a real total-RAM reading on a supported host"
    assert free >= 0
    if total > 0:
        assert free <= total, "free RAM can't exceed total RAM"
    assert disk > 0, "expected real free disk space for '.'"
    assert 0.0 <= load <= 100.0

    print("sysinfo.py self-check PASSED")
