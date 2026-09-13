"""Self-installing inference engine (INSTRUCTIONS §2.1: standalone, no third-party
service like LM Studio/Ollama).

We embed llama.cpp the way an app bundles ffmpeg: download a prebuilt release
binary, verify it runs, cache it, and drive it ourselves (backends/llamacpp.py
spawns it as a child process). Compiling from source is a last resort, only
used when no prebuilt asset exists for this platform+accel.

Also handles GGUF model acquisition: peer swarm leech first (reuses
meshcompute_runtime.swarm, already BLAKE3-verified there), HF origin direct
download as the fallback, with resume support.
"""

from __future__ import annotations

import dataclasses
import platform
import shutil
import subprocess
import tarfile
from pathlib import Path

import blake3
import httpx

DEFAULT_RUNTIME_CACHE = Path.home() / ".mesh" / "runtime"
DEFAULT_MODEL_CACHE = Path.home() / ".mesh" / "models"

# Plain release list (NOT /releases/latest): llama.cpp's actual GitHub "latest"
# marker points at a non-binary release because every b-numbered CI build is
# flagged prerelease. /releases (sorted newest-first) is the real feed.
_GITHUB_RELEASES = "https://api.github.com/repos/ggml-org/llama.cpp/releases"
_HF_HOST = "https://huggingface.co"
_CHUNK = 1 << 20


@dataclasses.dataclass
class ModelSpec:
    """Minimal model-acquisition spec — enough to fetch one GGUF file without a
    full signed ModelManifest. Used for the CPU-box smoke default; a real
    meshcompute_protocol.ModelManifest works too (see ensure_model)."""
    repo: str                     # HF repo id, e.g. "bartowski/SmolLM2-360M-Instruct-GGUF"
    filename: str                 # e.g. "SmolLM2-360M-Instruct-Q4_K_M.gguf"
    revision: str = "main"
    artifact_root_hash: str = "PENDING-unpinned-smoke-model"
    size_bytes: int = 0


# ~258MiB, well-known repo, runs in seconds on CPU — the CPU-box smoke default.
# The real target model (public/qwen3.8-27b-fable) is resolved from the
# control-plane catalog instead; see daemon.py's _resolve_model.
SMOKE_MODEL = ModelSpec(
    repo="bartowski/SmolLM2-360M-Instruct-GGUF",
    filename="SmolLM2-360M-Instruct-Q4_K_M.gguf",
    revision="main",
)


def detect_accel() -> str:
    """"cuda" if an nvidia GPU is present, "metal" on macOS, else "cpu"."""
    if shutil.which("nvidia-smi") is not None:
        return "cuda"
    if platform.system() == "Darwin":
        return "metal"
    return "cpu"


def _arch_token(machine: str) -> str:
    return "arm64" if machine in ("arm64", "aarch64") else "x64"


def _expected_asset_suffix(system: str, accel: str, machine: str) -> str | None:
    """llama.cpp release asset filenames are `llama-<tag>-bin-<platform>.tar.gz`;
    since the tag isn't known until the release list is fetched, match by this
    suffix instead of a full name. Suffix (not substring) matters: it rules out
    e.g. "...-bin-ubuntu-vulkan-x64.tar.gz" when we want the plain CPU build.
    Returns None if upstream ships no such prebuilt (caller falls back to a
    source build).
    """
    arch = _arch_token(machine)
    if system == "Linux":
        if accel == "cuda":
            # TODO(phase-1.5): as of the current llama.cpp CI matrix there is no
            # prebuilt Linux+CUDA asset (only Windows gets CUDA prebuilts; Linux
            # offers cpu/vulkan/rocm/sycl/openvino). Verified by inspecting real
            # release asset lists. Falls through to _build_from_source below.
            return None
        return f"bin-ubuntu-{arch}.tar.gz"
    if system == "Darwin":
        return f"bin-macos-{arch}.tar.gz"  # Metal is built into the macOS release build
    return None  # Windows/other: not a target for this daemon yet


def _find_binary(root: Path, name: str) -> Path | None:
    if not root.is_dir():
        return None
    for p in root.rglob(name):
        if p.is_file():
            return p
    return None


def _verify_binary(path: Path) -> bool:
    try:
        out = subprocess.run([str(path), "--version"], capture_output=True, timeout=10)
        return out.returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


async def _latest_releases(client: httpx.AsyncClient) -> list[dict]:
    resp = await client.get(_GITHUB_RELEASES, params={"per_page": 5})
    resp.raise_for_status()
    releases = resp.json()
    if not releases:
        raise RuntimeError("no llama.cpp releases found on GitHub")
    return releases


async def ensure_llama_server(cache_dir: str | Path = DEFAULT_RUNTIME_CACHE,
                               accel: str | None = None) -> str:
    """Return a path to a working local `llama-server` binary, downloading and
    caching a prebuilt release the first time. Reuse is fully offline: once
    cached, later calls never touch the network.
    """
    accel = accel or detect_accel()
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    system = platform.system()
    machine = platform.machine()

    marker = cache_dir / f"current-{system}-{accel}-{machine}.txt"
    if marker.exists():
        cached_bin = Path(marker.read_text().strip())
        if cached_bin.is_file() and _verify_binary(cached_bin):
            return str(cached_bin)

    asset_suffix = _expected_asset_suffix(system, accel, machine)
    if asset_suffix is not None:
        async with httpx.AsyncClient(timeout=httpx.Timeout(30.0, read=120.0),
                                      follow_redirects=True) as client:
            for release in await _latest_releases(client):
                asset = next((a for a in release.get("assets", [])
                              if a["name"].endswith(asset_suffix)), None)
                if asset is None:
                    continue
                tag = release["tag_name"]
                extract_dir = cache_dir / f"{tag}-{system}-{accel}-{machine}"
                bin_path = _find_binary(extract_dir, "llama-server")
                if bin_path is None or not _verify_binary(bin_path):
                    archive = cache_dir / asset["name"]
                    async with client.stream("GET", asset["browser_download_url"]) as r:
                        r.raise_for_status()
                        with open(archive, "wb") as f:
                            async for chunk in r.aiter_bytes(_CHUNK):
                                f.write(chunk)
                    extract_dir.mkdir(parents=True, exist_ok=True)
                    with tarfile.open(archive) as tf:
                        tf.extractall(extract_dir)  # trusted upstream GitHub release asset
                    archive.unlink(missing_ok=True)
                    bin_path = _find_binary(extract_dir, "llama-server")
                if bin_path is not None and _verify_binary(bin_path):
                    marker.write_text(str(bin_path))
                    return str(bin_path)
                break  # found the asset but it never produced a working binary; don't loop releases

    # Last resort: build from source. Real for the general case, but the CUDA
    # variant is untestable on this CPU-only dev box.
    # TODO(phase-1.5): verify -DGGML_CUDA=ON build on a real CUDA box.
    return _build_from_source(cache_dir, accel, marker)


def _build_from_source(cache_dir: Path, accel: str, marker: Path) -> str:
    src_dir = cache_dir / "src"
    if src_dir.exists():
        shutil.rmtree(src_dir)
    subprocess.run(["git", "clone", "--depth", "1",
                     "https://github.com/ggml-org/llama.cpp", str(src_dir)],
                    check=True, timeout=300)
    build_dir = src_dir / "build"
    cmake_configure = ["cmake", "-S", str(src_dir), "-B", str(build_dir),
                        "-DCMAKE_BUILD_TYPE=Release"]
    if accel == "cuda":
        cmake_configure.append("-DGGML_CUDA=ON")
    # This box shares resources with other sessions; queue/cap the build (see AGENTS.md).
    subprocess.run(["aw", "run", "--class", "build", "--"] + cmake_configure,
                    check=True, timeout=600)
    subprocess.run(["aw", "run", "--class", "build", "--",
                     "cmake", "--build", str(build_dir), "--config", "Release"],
                    check=True, timeout=3600)
    bin_path = _find_binary(build_dir, "llama-server")
    if bin_path is None or not _verify_binary(bin_path):
        raise RuntimeError("source build did not produce a working llama-server")
    marker.write_text(str(bin_path))
    return str(bin_path)


def _blake3_file(path: Path) -> str:
    h = blake3.blake3()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(_CHUNK), b""):
            h.update(block)
    return h.hexdigest()


def _shard_from_manifest(manifest) -> tuple[str, str, str, object]:
    """(repo, revision, manifest_hash, ShardFile) from a real ModelManifest —
    first quantization's first file.
    TODO(phase-1.5): real quant-selection policy (VRAM fit, contribution
    policy) instead of "first listed"; belongs with the scheduler, not here.
    """
    q = manifest.quantizations[0]
    shard = q.files[0]
    return manifest.upstream.repository, manifest.upstream.revision, manifest.manifest_hash(), shard


async def _download_hf_file(repo: str, revision: str, filename: str, dest: Path,
                             root_hash: str) -> None:
    url = f"{_HF_HOST}/{repo}/resolve/{revision}/{filename}"
    part = dest.with_suffix(dest.suffix + ".part")
    resume_from = part.stat().st_size if part.exists() else 0
    headers = {"Range": f"bytes={resume_from}-"} if resume_from else {}
    async with httpx.AsyncClient(timeout=httpx.Timeout(30.0, read=120.0),
                                  follow_redirects=True) as client:
        async with client.stream("GET", url, headers=headers) as r:
            if resume_from and r.status_code == 200:
                resume_from = 0  # server ignored Range; restart clean rather than corrupt
            r.raise_for_status()
            with open(part, "ab" if resume_from else "wb") as f:
                async for chunk in r.aiter_bytes(_CHUNK):
                    f.write(chunk)
    if not root_hash.startswith("PENDING"):
        digest = _blake3_file(part)
        if digest != root_hash:
            part.unlink(missing_ok=True)
            raise RuntimeError(f"downloaded {filename} blake3 {digest} != expected {root_hash}")
    part.replace(dest)


async def ensure_model(manifest_or_spec, cache_dir: str | Path = DEFAULT_MODEL_CACHE,
                        swarm=None, peers=None) -> str:
    """Acquire a local GGUF path. Acquisition order:
      (a) peer swarm leech, if `peers` (already-open Streams for this shard) and
          `swarm` (a meshcompute_runtime.swarm.ChunkCache) are given —
          content-addressed, BLAKE3-verified there;
      (b) HF origin fallback: direct download with resume + BLAKE3 verification
          against the manifest's artifact_root_hash (skipped when it's the
          "PENDING..." placeholder).

    `manifest_or_spec` is either a ModelSpec (the smoke default) or a real
    meshcompute_protocol.ModelManifest.

    TODO(phase-1.5): `peers` here is single-shard/single-source-at-a-time (try
    each stream in turn) — rarity-aware multi-source selection across many
    peers is swarm.py's own documented Phase-1.5 TODO, not reimplemented here.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    if isinstance(manifest_or_spec, ModelSpec):
        repo, revision, filename = manifest_or_spec.repo, manifest_or_spec.revision, manifest_or_spec.filename
        root_hash, size_hint = manifest_or_spec.artifact_root_hash, manifest_or_spec.size_bytes
        manifest_hash, shard = revision, None
    elif hasattr(manifest_or_spec, "quantizations"):
        repo, revision, manifest_hash, shard = _shard_from_manifest(manifest_or_spec)
        filename, root_hash, size_hint = shard.rfilename, shard.artifact_root_hash, shard.size_bytes
    else:
        raise TypeError(f"ensure_model: unsupported manifest_or_spec type {type(manifest_or_spec)!r}")

    dest = cache_dir / repo.replace("/", "--") / filename
    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.is_file() and (size_hint == 0 or dest.stat().st_size == size_hint):
        if root_hash.startswith("PENDING") or _blake3_file(dest) == root_hash:
            return str(dest)  # already cached + verified

    if peers and swarm is not None and shard is not None:
        from meshcompute_runtime.swarm import ChunkVerifyError, leech_file
        for stream in peers:
            try:
                await leech_file(stream, dest, swarm, manifest_hash, shard)
                return str(dest)
            except (ChunkVerifyError, EOFError, TimeoutError, OSError):
                continue  # try the next peer; HF origin fallback below if all fail

    await _download_hf_file(repo, revision, filename, dest, root_hash)
    return str(dest)


if __name__ == "__main__":
    import asyncio

    async def _demo() -> None:
        accel = detect_accel()
        print(f"detected accel: {accel}")
        path = await ensure_llama_server()
        assert Path(path).is_file(), f"expected a real binary at {path}"
        assert _verify_binary(Path(path)), "llama-server --version failed"
        # second call must be a pure cache hit (no network) — prove reuse works.
        path2 = await ensure_llama_server()
        assert path2 == path, "cached ensure_llama_server() call returned a different path"
        print(f"runtime_installer.py self-check PASSED: llama-server at {path}")

    asyncio.run(_demo())
