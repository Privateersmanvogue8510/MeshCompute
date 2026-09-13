"""Self-installing inference engine + model acquisition (INSTRUCTIONS §2.1, §10).

Engine: llama.cpp is embedded the way an app bundles ffmpeg — download the
upstream prebuilt release for this OS/arch/accelerator, verify it runs, cache
it, drive it as a child process (backends/llamacpp.py). Prebuilts exist for
every target except Linux+CUDA, where the cascade is:
  CUDA source build (only when asked, needs cmake+nvcc+git)  ->  Vulkan prebuilt
  (runs on the NVIDIA driver, no toolkit)  ->  CPU prebuilt.

Model: one GGUF quantization (all its files) resolved from a signed
ModelManifest or a ModelSpec. Peer swarm first (BLAKE3-verified chunks over
QUIC, swarm.py), Hugging Face origin as the fallback, both resumable, both
verified against the manifest's artifact_root_hash. The cache keeps a small
state.json so the daemon knows which channel/version each file belongs to,
can seed it to peers, and can delete it when the channel moves on.
"""

from __future__ import annotations

import dataclasses
import json
import os
import platform
import shutil
import subprocess
import tarfile
import zipfile
from pathlib import Path
from typing import Awaitable, Callable

import blake3
import httpx

# MESH_HOME relocates everything the node stores (engine, models, identity,
# logs); default ~/.mesh. Lets one box run several nodes (tests, demos).
MESH_HOME = Path(os.environ.get("MESH_HOME") or (Path.home() / ".mesh"))
DEFAULT_RUNTIME_CACHE = MESH_HOME / "runtime"
DEFAULT_MODEL_CACHE = MESH_HOME / "models"

# Plain release list (NOT /releases/latest): every b-numbered llama.cpp CI build
# is flagged prerelease, so "latest" points at a non-binary release.
_GITHUB_RELEASES = "https://api.github.com/repos/ggml-org/llama.cpp/releases"
_HF_HOST = "https://huggingface.co"
_CHUNK = 1 << 20
_WIN_CUDA = "12.4"          # broadest driver compatibility of the shipped CUDA builds
_DISK_MARGIN = 1 << 30      # keep 1 GiB free after any download
_BIN = "llama-server.exe" if platform.system() == "Windows" else "llama-server"


def _default_log(msg: str) -> None:
    print(f"[mesh] {msg}", flush=True)


@dataclasses.dataclass
class ModelSpec:
    """Minimal model-acquisition spec — one GGUF file, no signed manifest.
    Used for the CPU smoke default; a real ModelManifest is the normal path."""
    repo: str                     # HF repo id
    filename: str
    revision: str = "main"
    artifact_root_hash: str = "PENDING-unpinned"   # BLAKE3 of the file, or PENDING-*
    size_bytes: int = 0
    alias: str = "smoke/smollm2-360m"
    version: int = 1

    @property
    def manifest_hash(self) -> str:
        return f"spec:{self.repo}@{self.revision}/{self.filename}"


# ~258 MiB, runs in seconds on CPU — the smoke default when no model is chosen.
# Hash + size pinned from the real file (blake3 over 270590880 bytes).
SMOKE_MODEL = ModelSpec(
    repo="bartowski/SmolLM2-360M-Instruct-GGUF",
    filename="SmolLM2-360M-Instruct-Q4_K_M.gguf",
    revision="main",
    artifact_root_hash="6cb0ba7f2d5e00ad4445c162c9e7e3f2d2e620f10656b86b1df1f6b1797e509d",
    size_bytes=270590880,
)


# --------------------------------------------------------------------------- engine
@dataclasses.dataclass
class Engine:
    path: str          # llama-server binary
    accel: str         # effective accelerator: cuda | vulkan | metal | cpu
    tag: str           # upstream release tag (or "source")


def detect_accel() -> str:
    """Requested accelerator: MESH_ACCEL env, else nvidia-smi -> cuda, macOS ->
    metal, else cpu. The EFFECTIVE accelerator may be lower (see _cascade)."""
    forced = os.environ.get("MESH_ACCEL", "").strip().lower()
    if forced:
        return forced
    if shutil.which("nvidia-smi") is not None:
        return "cuda"
    if platform.system() == "Darwin":
        return "metal"
    return "cpu"


def _arch_token(machine: str) -> str:
    return "arm64" if machine.lower() in ("arm64", "aarch64") else "x64"


def asset_suffix(system: str, accel: str, machine: str) -> str | None:
    """Upstream asset name suffix for (OS, accel, arch), or None when upstream
    ships no such prebuilt. Suffix (not substring) match: rules out e.g.
    "...-bin-ubuntu-vulkan-x64.tar.gz" when the plain CPU build is wanted."""
    arch = _arch_token(machine)
    if system == "Linux":
        return {"cpu": f"bin-ubuntu-{arch}.tar.gz",
                "vulkan": f"bin-ubuntu-vulkan-{arch}.tar.gz"}.get(accel)
    if system == "Darwin":
        return f"bin-macos-{arch}.tar.gz" if accel in ("metal", "cpu") else None
    if system == "Windows":
        if accel == "cpu":
            return f"bin-win-cpu-{arch}.zip"
        if accel == "cuda" and arch == "x64":
            return f"bin-win-cuda-{_WIN_CUDA}-x64.zip"
        if accel == "vulkan" and arch == "x64":
            return "bin-win-vulkan-x64.zip"
    return None


def _cascade(system: str, accel: str) -> list[str]:
    """Accelerators to try, best first, given what the user asked for."""
    if system == "Darwin":
        return ["metal"]
    if accel == "cuda":
        # Linux: no upstream CUDA prebuilt -> source build only if explicitly
        # wanted (cuda-build), else Vulkan on the NVIDIA driver, else CPU.
        return ["cuda", "vulkan", "cpu"] if system == "Windows" else ["vulkan", "cpu"]
    if accel == "cuda-build":
        return ["cuda-build", "vulkan", "cpu"] if system == "Linux" else ["cuda", "cpu"]
    if accel == "vulkan":
        return ["vulkan", "cpu"]
    return ["cpu"]


def _find_binary(root: Path, name: str = _BIN) -> Path | None:
    if not root.is_dir():
        return None
    for p in root.rglob(name):
        if p.is_file():
            return p
    return None


def _verify_binary(path: Path) -> bool:
    try:
        out = subprocess.run([str(path), "--version"], capture_output=True, timeout=20)
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


async def _download(client: httpx.AsyncClient, url: str, dest: Path) -> None:
    part = dest.with_suffix(dest.suffix + ".part")
    async with client.stream("GET", url) as r:
        r.raise_for_status()
        with open(part, "wb") as f:
            async for chunk in r.aiter_bytes(_CHUNK):
                f.write(chunk)
    part.replace(dest)


def _extract(archive: Path, into: Path) -> None:
    into.mkdir(parents=True, exist_ok=True)
    if archive.suffix == ".zip":
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(into)
    else:
        with tarfile.open(archive) as tf:
            tf.extractall(into, filter="data")  # trusted upstream asset; still no path escapes


async def _install_prebuilt(client: httpx.AsyncClient, release: dict, suffix: str,
                            cache_dir: Path, system: str, accel: str, machine: str) -> Path | None:
    asset = next((a for a in release.get("assets", []) if a["name"].endswith(suffix)), None)
    if asset is None:
        return None
    tag = release["tag_name"]
    extract_dir = cache_dir / f"{tag}-{system}-{accel}-{machine}"
    bin_path = _find_binary(extract_dir)
    if bin_path is None or not _verify_binary(bin_path):
        archive = cache_dir / asset["name"]
        await _download(client, asset["browser_download_url"], archive)
        _extract(archive, extract_dir)
        archive.unlink(missing_ok=True)
        bin_path = _find_binary(extract_dir)
        if bin_path is not None and system == "Windows" and accel == "cuda":
            # The CUDA build needs the CUDA runtime DLLs next to the exe; upstream
            # ships them as a separate cudart-*.zip in the same release.
            cudart = next((a for a in release.get("assets", [])
                           if a["name"].startswith("cudart-") and a["name"].endswith(suffix)), None)
            if cudart is not None:
                cz = cache_dir / cudart["name"]
                await _download(client, cudart["browser_download_url"], cz)
                _extract(cz, bin_path.parent)
                cz.unlink(missing_ok=True)
    if bin_path is not None and _verify_binary(bin_path):
        return bin_path
    return None


async def ensure_llama_server(cache_dir: str | Path = DEFAULT_RUNTIME_CACHE,
                              accel: str | None = None,
                              log: Callable[[str], None] = _default_log) -> Engine:
    """Return a working local llama-server, downloading and caching the best
    available build for this machine the first time. Later calls are offline
    cache hits. Never raises just because the requested accelerator is
    unavailable — it degrades down the cascade and reports what it used."""
    accel = (accel or detect_accel()).lower()
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    system, machine = platform.system(), platform.machine()

    marker = cache_dir / f"current-{system}-{accel}-{machine}.json"
    if marker.exists():
        try:
            cached = Engine(**json.loads(marker.read_text()))
            if Path(cached.path).is_file() and _verify_binary(Path(cached.path)):
                return cached
        except (json.JSONDecodeError, TypeError):
            pass

    errors: list[str] = []
    for step in _cascade(system, accel):
        if step == "cuda-build":
            try:
                eng = _build_from_source(cache_dir, "cuda")
                marker.write_text(json.dumps(dataclasses.asdict(eng)))
                return eng
            except (OSError, subprocess.SubprocessError, RuntimeError) as e:
                errors.append(f"cuda source build failed: {e}")
                continue
        suffix = asset_suffix(system, step, machine)
        if suffix is None:
            errors.append(f"no upstream prebuilt for {system}/{step}/{machine}")
            continue
        async with httpx.AsyncClient(timeout=httpx.Timeout(30.0, read=300.0),
                                     follow_redirects=True) as client:
            for release in await _latest_releases(client):
                try:
                    bin_path = await _install_prebuilt(client, release, suffix, cache_dir,
                                                       system, step, machine)
                except (httpx.HTTPError, OSError, zipfile.BadZipFile, tarfile.TarError) as e:
                    errors.append(f"{step}@{release.get('tag_name')}: {e}")
                    continue
                if bin_path is not None:
                    eng = Engine(path=str(bin_path), accel=step, tag=release["tag_name"])
                    marker.write_text(json.dumps(dataclasses.asdict(eng)))
                    if step != accel:
                        log(f"accelerator '{accel}' unavailable here; using '{step}' build")
                    return eng
                errors.append(f"{step}@{release.get('tag_name')}: asset missing or binary "
                              f"does not run (missing driver/runtime library?)")
    raise RuntimeError("could not obtain a working llama-server: " + "; ".join(errors))


def _build_from_source(cache_dir: Path, accel: str) -> Engine:
    """Last resort / opt-in: `git clone` + cmake. Needs git, cmake, a C++
    toolchain, and for CUDA the toolkit (nvcc). Build once, cached after."""
    for tool in ("git", "cmake"):
        if shutil.which(tool) is None:
            raise RuntimeError(f"{tool} not on PATH (required for a source build)")
    if accel == "cuda" and shutil.which("nvcc") is None and not os.environ.get("CUDACXX"):
        raise RuntimeError("nvcc not on PATH (CUDA toolkit required for -DGGML_CUDA=ON)")
    src_dir = cache_dir / "src"
    if src_dir.exists():
        shutil.rmtree(src_dir)
    subprocess.run(["git", "clone", "--depth", "1", "https://github.com/ggml-org/llama.cpp",
                    str(src_dir)], check=True, timeout=600)
    build_dir = src_dir / "build"
    configure = ["cmake", "-S", str(src_dir), "-B", str(build_dir), "-DCMAKE_BUILD_TYPE=Release",
                 "-DLLAMA_CURL=OFF"]
    if accel == "cuda":
        configure.append("-DGGML_CUDA=ON")
    subprocess.run(configure, check=True, timeout=900)
    subprocess.run(["cmake", "--build", str(build_dir), "--config", "Release", "--target",
                    "llama-server", "-j", str(max(1, (os.cpu_count() or 2) // 2))],
                   check=True, timeout=7200)
    bin_path = _find_binary(build_dir)
    if bin_path is None or not _verify_binary(bin_path):
        raise RuntimeError("source build did not produce a working llama-server")
    return Engine(path=str(bin_path), accel=accel, tag="source")


# --------------------------------------------------------------------------- model cache state
def blake3_file(path: str | Path) -> str:
    h = blake3.blake3()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(_CHUNK), b""):
            h.update(block)
    return h.hexdigest()


def _state_path(cache_dir: Path) -> Path:
    return cache_dir / "state.json"


def load_state(cache_dir: str | Path | None = None) -> dict:
    """{"selected_alias": str|None, "channels": {alias: {"manifest_hash", "version",
    "files": [{"rfilename","path","size_bytes","artifact_root_hash","chunk_bytes"}]}}}"""
    p = _state_path(Path(cache_dir or DEFAULT_MODEL_CACHE))
    try:
        data = json.loads(p.read_text())
    except (OSError, json.JSONDecodeError):
        data = {}
    data.setdefault("selected_alias", None)
    data.setdefault("channels", {})
    return data


def save_state(state: dict, cache_dir: str | Path | None = None) -> None:
    cache_dir = Path(cache_dir or DEFAULT_MODEL_CACHE)
    cache_dir.mkdir(parents=True, exist_ok=True)
    tmp = _state_path(cache_dir).with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2))
    tmp.replace(_state_path(cache_dir))


def installed_bytes(state: dict) -> int:
    return sum(f.get("size_bytes", 0) for ch in state["channels"].values() for f in ch["files"])


def seed_index(state: dict) -> dict[tuple[str, str], tuple[Path, dict]]:
    """(manifest_hash, rfilename) -> (path, shard-dict) for every file this node
    holds — what swarm.serve_chunk_stream serves from. Missing files are skipped."""
    out: dict[tuple[str, str], tuple[Path, dict]] = {}
    for ch in state["channels"].values():
        for f in ch["files"]:
            p = Path(f["path"])
            if p.is_file() and p.stat().st_size == f["size_bytes"]:
                out[(ch["manifest_hash"], f["rfilename"])] = (p, f)
    return out


def record_channel(state: dict, alias: str, manifest_hash: str, version: int,
                   files: list[dict]) -> list[str]:
    """Point `alias` at a new version. Returns paths of the PREVIOUS version's
    files that no other channel still references (the caller deletes them
    once the new engine is healthy — CONFIGURATION.md: no junk left behind)."""
    old = state["channels"].get(alias)
    state["channels"][alias] = {"manifest_hash": manifest_hash, "version": version, "files": files}
    if not old:
        return []
    still_used = {f["path"] for ch in state["channels"].values() for f in ch["files"]}
    return [f["path"] for f in old["files"] if f["path"] not in still_used]


def delete_files(paths: list[str], cache_dir: str | Path | None = None) -> list[str]:
    """Delete model files, refusing anything outside the cache dir. Sidecars
    (.part/.mcprogress.json) go with them. Returns what was actually removed."""
    root = Path(cache_dir or DEFAULT_MODEL_CACHE).resolve()
    removed = []
    for s in paths:
        p = Path(s)
        try:
            if root not in p.resolve().parents:
                continue
            for extra in (p, p.with_suffix(p.suffix + ".part"),
                          p.with_name(p.name + ".mcprogress.json")):
                if extra.is_file():
                    extra.unlink()
            removed.append(s)
            if p.parent != root and p.parent.is_dir() and not any(p.parent.iterdir()):
                p.parent.rmdir()
        except OSError:
            continue
    return removed


# --------------------------------------------------------------------------- model acquisition
@dataclasses.dataclass
class ModelTarget:
    """Everything ensure_model needs, normalised from a ModelSpec or ModelManifest."""
    alias: str
    version: int
    manifest_hash: str
    repo: str
    revision: str
    files: list[dict]          # shard dicts: rfilename,size_bytes,artifact_root_hash,chunk_bytes

    @property
    def total_bytes(self) -> int:
        return sum(f["size_bytes"] for f in self.files)


def pick_quant(manifest, quant_id: str | None = None):
    """The catalog's FIRST listed quantization is its recommendation (the
    scheduler sizes model-fit the same way); `quant_id` overrides."""
    if not manifest.quantizations:
        raise ValueError(f"manifest {manifest.id} lists no quantizations")
    if quant_id:
        for q in manifest.quantizations:
            if q.id == quant_id:
                return q
        raise ValueError(f"manifest {manifest.id} has no quantization {quant_id!r}; "
                         f"have {[q.id for q in manifest.quantizations]}")
    return manifest.quantizations[0]


def model_target(manifest_or_spec, quant_id: str | None = None) -> ModelTarget:
    if isinstance(manifest_or_spec, ModelSpec):
        s = manifest_or_spec
        return ModelTarget(alias=s.alias, version=s.version, manifest_hash=s.manifest_hash,
                           repo=s.repo, revision=s.revision,
                           files=[{"rfilename": s.filename, "size_bytes": s.size_bytes,
                                   "artifact_root_hash": s.artifact_root_hash,
                                   "chunk_bytes": _CHUNK}])
    if hasattr(manifest_or_spec, "quantizations"):
        m = manifest_or_spec
        q = pick_quant(m, quant_id)
        return ModelTarget(alias=m.id, version=m.version, manifest_hash=m.manifest_hash(),
                           repo=m.upstream.repository, revision=m.upstream.revision,
                           files=[{"rfilename": f.rfilename, "size_bytes": f.size_bytes,
                                   "artifact_root_hash": f.artifact_root_hash,
                                   "chunk_bytes": f.chunk_bytes} for f in q.files])
    raise TypeError(f"unsupported model spec type {type(manifest_or_spec)!r}")


def model_dest(cache_dir: Path, target: ModelTarget, rfilename: str) -> Path:
    # Versioned per manifest so a channel update never overwrites the file the
    # running engine has mmap'ed; the old dir is deleted after the swap.
    ver = blake3.blake3(target.manifest_hash.encode("utf-8")).hexdigest()[:12]
    return cache_dir / target.repo.replace("/", "--") / f"v{target.version}-{ver}" / rfilename


def sweep_orphans(state: dict, cache_dir: str | Path | None = None) -> list[str]:
    """Delete model files under the cache that no channel in state.json owns —
    leftovers from a crashed changeover or an older layout. Only touches our
    own file types (.gguf, .part, .mcprogress.json); never anything else."""
    root = Path(cache_dir or DEFAULT_MODEL_CACHE)
    if not root.is_dir():
        return []
    owned = {str(Path(f["path"])) for ch in state["channels"].values() for f in ch["files"]}
    removed: list[str] = []
    for p in sorted(root.rglob("*")):
        if not p.is_file() or str(p) in owned:
            continue
        if p.suffix == ".gguf" or p.name.endswith((".gguf.part", ".mcprogress.json")):
            try:
                p.unlink()
                removed.append(str(p))
            except OSError:
                continue
    for d in sorted((d for d in root.rglob("*") if d.is_dir()), key=lambda d: -len(d.parts)):
        try:
            if not any(d.iterdir()):
                d.rmdir()
        except OSError:
            continue
    return removed


def _verified(path: Path, shard: dict) -> bool:
    if not path.is_file():
        return False
    if shard["size_bytes"] and path.stat().st_size != shard["size_bytes"]:
        return False
    root = shard["artifact_root_hash"]
    return root.startswith("PENDING") or blake3_file(path) == root


def check_storage(cache_dir: Path, need_bytes: int, state: dict,
                  max_storage_bytes: int | None) -> None:
    """Refuse a download that would overfill the disk or the contributor's
    storage budget (CONFIGURATION.md storage.max_cache_gb)."""
    free = shutil.disk_usage(cache_dir).free
    if need_bytes + _DISK_MARGIN > free:
        raise RuntimeError(f"not enough disk: need {need_bytes/1e9:.1f} GB + 1 GB margin, "
                           f"{free/1e9:.1f} GB free under {cache_dir}")
    if max_storage_bytes is not None and installed_bytes(state) + need_bytes > max_storage_bytes:
        raise RuntimeError(f"storage budget exceeded: {installed_bytes(state)/1e9:.1f} GB "
                           f"installed + {need_bytes/1e9:.1f} GB new > "
                           f"{max_storage_bytes/1e9:.1f} GB (--max-storage)")


async def _download_hf_file(repo: str, revision: str, filename: str, dest: Path,
                            root_hash: str, log: Callable[[str], None]) -> None:
    """Resumable HF origin download, verified before commit. HF_TOKEN honoured
    for gated repos."""
    url = f"{_HF_HOST}/{repo}/resolve/{revision}/{filename}"
    part = dest.with_suffix(dest.suffix + ".part")
    resume_from = part.stat().st_size if part.exists() else 0
    headers = {"Range": f"bytes={resume_from}-"} if resume_from else {}
    token = os.environ.get("HF_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    log(f"downloading {filename} from Hugging Face"
        + (f" (resuming at {resume_from/1e6:.0f} MB)" if resume_from else ""))
    async with httpx.AsyncClient(timeout=httpx.Timeout(30.0, read=300.0),
                                 follow_redirects=True) as client:
        async with client.stream("GET", url, headers=headers) as r:
            if resume_from and r.status_code == 200:
                resume_from = 0  # server ignored Range; restart clean rather than corrupt
            r.raise_for_status()
            with open(part, "ab" if resume_from else "wb") as f:
                async for chunk in r.aiter_bytes(_CHUNK):
                    f.write(chunk)
    if not root_hash.startswith("PENDING"):
        digest = blake3_file(part)
        if digest != root_hash:
            part.unlink(missing_ok=True)
            raise RuntimeError(f"downloaded {filename} blake3 {digest} != expected {root_hash}")
    part.replace(dest)


# swarm_fetch(dest, manifest_hash, shard_dict) -> True if the file was fully
# fetched from peers and verified; False/exception -> fall back to HF origin.
SwarmFetch = Callable[[Path, str, dict], Awaitable[bool]]


async def ensure_model(manifest_or_spec, cache_dir: str | Path | None = None, *,
                       quant_id: str | None = None, swarm_fetch: SwarmFetch | None = None,
                       max_storage_bytes: int | None = None,
                       log: Callable[[str], None] = _default_log) -> tuple[str, ModelTarget]:
    """Make every file of the chosen quantization present + verified locally.
    Order per file: cached -> peer swarm -> Hugging Face. Returns (path of the
    first file — what llama-server loads; split GGUFs find their siblings next
    to it — , the resolved ModelTarget). Records nothing in state.json: the
    caller does that once the engine actually runs on the files."""
    cache_dir = Path(cache_dir or DEFAULT_MODEL_CACHE)
    cache_dir.mkdir(parents=True, exist_ok=True)
    target = model_target(manifest_or_spec, quant_id)
    state = load_state(cache_dir)

    missing = [f for f in target.files if not _verified(model_dest(cache_dir, target, f["rfilename"]), f)]
    if missing:
        check_storage(cache_dir, sum(f["size_bytes"] for f in missing), state, max_storage_bytes)
    for shard in missing:
        dest = model_dest(cache_dir, target, shard["rfilename"])
        dest.parent.mkdir(parents=True, exist_ok=True)
        got = False
        if swarm_fetch is not None:
            try:
                got = await swarm_fetch(dest, target.manifest_hash, shard)
            except Exception as e:  # any peer failure -> origin; never abort the install
                log(f"swarm fetch of {shard['rfilename']} failed ({e}); using origin")
                got = False
        if got and not _verified(dest, shard):
            log(f"swarm result for {shard['rfilename']} failed verification; using origin")
            got = False
        if not got:
            await _download_hf_file(target.repo, target.revision, shard["rfilename"], dest,
                                    shard["artifact_root_hash"], log)
        else:
            log(f"{shard['rfilename']} obtained from peers (verified)")
    first = model_dest(cache_dir, target, target.files[0]["rfilename"])
    return str(first), target


def channel_files(cache_dir: Path, target: ModelTarget) -> list[dict]:
    """state.json file records for a ModelTarget whose files are present."""
    return [{**f, "path": str(model_dest(Path(cache_dir), target, f["rfilename"]))}
            for f in target.files]


if __name__ == "__main__":
    import asyncio
    import tempfile

    # pure asset matrix — every target OS/accel resolves to a real upstream name or None
    assert asset_suffix("Linux", "cpu", "x86_64") == "bin-ubuntu-x64.tar.gz"
    assert asset_suffix("Linux", "vulkan", "aarch64") == "bin-ubuntu-vulkan-arm64.tar.gz"
    assert asset_suffix("Linux", "cuda", "x86_64") is None
    assert asset_suffix("Darwin", "metal", "arm64") == "bin-macos-arm64.tar.gz"
    assert asset_suffix("Windows", "cuda", "AMD64") == "bin-win-cuda-12.4-x64.zip"
    assert asset_suffix("Windows", "cpu", "ARM64") == "bin-win-cpu-arm64.zip"
    assert _cascade("Linux", "cuda") == ["vulkan", "cpu"]
    assert _cascade("Linux", "cuda-build") == ["cuda-build", "vulkan", "cpu"]
    assert _cascade("Windows", "cuda") == ["cuda", "vulkan", "cpu"]
    assert _cascade("Darwin", "cuda") == ["metal"]

    with tempfile.TemporaryDirectory() as d:
        st = load_state(d)
        t = model_target(SMOKE_MODEL)
        assert record_channel(st, t.alias, t.manifest_hash, 1, channel_files(Path(d), t)) == []
        old = record_channel(st, t.alias, "spec:new", 2, [{**f, "path": "/x/new.gguf"}
                                                          for f in t.files])
        assert old == [str(model_dest(Path(d), t, SMOKE_MODEL.filename))]
        save_state(st, d)
        assert load_state(d)["channels"][t.alias]["version"] == 2
    print("runtime_installer pure self-check PASSED")

    async def _demo() -> None:
        eng = await ensure_llama_server()
        assert Path(eng.path).is_file() and _verify_binary(Path(eng.path))
        eng2 = await ensure_llama_server()
        assert eng2.path == eng.path, "cached call must be a pure cache hit"
        path, target = await ensure_model(SMOKE_MODEL)
        assert _verified(Path(path), target.files[0])
        print(f"engine {eng.accel}@{eng.tag}: {eng.path}\nmodel: {path}\n"
              f"runtime_installer.py self-check PASSED")

    asyncio.run(_demo())
